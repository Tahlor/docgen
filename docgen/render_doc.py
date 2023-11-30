import warnings
import random
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.generic import TextStringObject, NameObject, ContentStream
from PyPDF2._utils import b_
from docgen.localize import generate_localization_from_bytes, generate_localization_from_file
from math import ceil
from docgen.img_tools import *
import numpy as np
from PIL import ImageChops, ImageDraw, PpmImagePlugin
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from typing import List, Tuple, Optional, Union
from docgen.bbox import BBox
from docgen.utils import utils
import logging
from PIL import Image
import dill as pickle
from multiprocessing import Process, Queue
from collections.abc import Iterator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")

def composite_images_numpy(background_image, paste_image, position):
    """
    Args:
        background_image:
        paste_image:
        position:

    Returns:

    """
    if not isinstance(background_image, np.ndarray):
        mode = background_image.mode
        background_image = np.array(background_image)
        if mode=="L":
            background_image = cv2.cvtColor(background_image, cv2.COLOR_GRAY2RGB)

    paste_image = np.asarray(paste_image)

    # Converting to numpy necessitates transposing the positions; y is now first
    x,y = position

    yslice = slice(y, y+get_y(paste_image))
    xslice = slice(x, x+get_x(paste_image))
    if paste_image.ndim < 3:
        paste_image = np.tile(paste_image[:, :, None], 3)

    try:
        background_image[yslice, xslice] = np.minimum(paste_image, background_image[yslice,xslice])
    except Exception as e:
        print(e)
        pass

    return background_image

def composite_images_PIL(background:Union[Image.Image, np.ndarray],
                         paste:Union[Image.Image, np.ndarray],
                         pos,
                         offset=(0,0)):
    """

    Args:
        background:
        paste:
        pos: (x,y) or BBox
        offset: (x,y) offset from pos
                If pos is in absolute coordinates, but pasting into a section,
                offset should be the top left corner of the section

    Returns:

    """
    if isinstance(pos, BBox):
        pos = pos.bbox[:2]
    elif isinstance(pos, (list, tuple)):
        pos = pos[:2]

    pos = int(pos[0]+offset[0]), int(pos[1]+offset[1])

    # # if background is RGB, convert foreground
    # if background.mode == "RGB" and paste.mode != "RGB":
    #     paste = paste.convert("RGB")
    # elif paste.mode == "RGB" and background.mode == "L":
    #     paste = paste.convert("L")

    phrase_img = merge_img(background, paste, pos)

    background.paste(phrase_img, pos)
    return background


def merge_img(bck, fore, pos):
    """ Preserves formlines etc. during pasting
        Crop window from FORM_IMG and merge with word image before pasting

    Args:
        bck:
        fore:
        pos: (x,y) or BBox

    Returns:

    """
    if isinstance(bck, np.ndarray):
        bck = bck.squeeze()
        bck = Image.fromarray(bck.astype(np.uint8), mode="L")
    if isinstance(fore, np.ndarray):
        fore = fore.squeeze()
        fore = Image.fromarray(fore.astype(np.uint8), mode="L")
    bck = bck.crop((pos[0], pos[1], pos[0]+fore.size[0],pos[1]+fore.size[1]))

    if bck.mode == "RGB" and fore.mode != "RGB":
        fore = fore.convert("RGB")
    elif fore.mode == "RGB" and bck.mode == "L":
        fore = fore.convert("L")

    return ImageChops.multiply(fore, bck)

def get_x(item):
    if isinstance(item, np.ndarray):
        return item.shape[1]
    elif isinstance(item, (PpmImagePlugin.PpmImageFile, Image.Image)):
        return item.size[0]
width = get_x

def get_y(item):
    if isinstance(item, np.ndarray):
        return item.shape[0]
    elif isinstance(item, (PpmImagePlugin.PpmImageFile, Image.Image)):
        return item.size[1]
height = get_y

def estimate_space(word_imgs, vertical_space, horizontal_space):
    """ Estimate how much space is needed for horizontal_min_to_fit
        Also scale option...ugh

    Returns:

    """
    widths = [word.shape[0] + horizontal_space for word in word_imgs]
    total_length_concat = np.cumsum(widths)

def identity(x):
    return x
def np_img_to_pil(x):
    if np.max(x) <= 1:
        return Image.fromarray((x.squeeze() * 255).astype(np.uint8))
    else:
        return Image.fromarray(x.squeeze())
def shape(item):
    """
    Args:
        item:
    Returns:
        x, y
    """
    if isinstance(item, np.ndarray):
        return item.shape[1],item.shape[0]
    elif isinstance(item, PpmImagePlugin.PpmImageFile) or isinstance(item, Image.Image):
        return item.size

def convert_to_ocr_format(localization,
                          box="bbox",
                          origin_offset=(0,0),
                          section=0,
                          text_decode_vocab=True):
    """ A localization is a list of BBox objects, which contain the pararaph, line, and word indices
        This function converts this list of BBox objects to a format to a OCR-like format
        {"paragraphs:[{lines:[{words:[...], "bbox":[bbox_for_all_lines]}, "bbox":[bbox_for_all_paragraphs]}

    Args:
        localization:
        box:
        origin_offset:
        section:
        text_decode_vocab: if BBox's have a text_decode_vocab attribute, then include them in the OCR

    Returns:

    """

    def start_next_line(line=None):
        nonlocal current_line_num
        if line:
            current_line_num = word_BBbox.line_number
            line[box] = BBox.get_maximal_box(line[box])
            line["text"] = " ".join(line["text"])
            if "text_decode_vocab" in line:
                line["text_decode_vocab"] = " ".join(line["text_decode_vocab"])
            current_paragraph_dict["lines"].append(line)
            current_paragraph_dict[box].append(line[box])
        out = {box: [], "text": [], "words": []}
        return out

    def start_next_paragraph(paragraph=None):
        nonlocal current_paragraph
        if paragraph:
            current_paragraph = word_BBbox.paragraph_index
            paragraph[box] = BBox.get_maximal_box(paragraph[box])
            ocr_dict_page["paragraphs"].append(paragraph)
            ocr_dict_page[box].append(paragraph[box])
        return {"lines": [], box: []}

    ocr_dict_page = {"paragraphs": [], box: []}
    current_paragraph_dict = start_next_paragraph()
    current_line_dict = start_next_line()
    current_line_num = 0
    current_paragraph = 0

    for word_BBbox in localization:

        # Start new line
        if word_BBbox.line_number!=current_line_num:
            current_line_dict = start_next_line(current_line_dict)

        # Start a new paragraph
        if word_BBbox.paragraph_index>current_paragraph:
            current_paragraph_dict = start_next_paragraph(current_paragraph_dict)

        current_line_dict["text"].append(word_BBbox.text)


        if hasattr(word_BBbox, "text_decode_vocab") and text_decode_vocab:
            if "text_decode_vocab" not in current_line_dict:
                current_line_dict["text_decode_vocab"] = []
            current_line_dict["text_decode_vocab"].append(word_BBbox.text_decode_vocab)

        current_word = {box: word_BBbox.offset_origin(origin_offset[0],
                                                origin_offset[1]).bbox,
                        "text": word_BBbox.text,
                        "id": [section,
                               current_paragraph,
                               word_BBbox.line_number,
                               word_BBbox.line_word_index]}
        current_line_dict["words"].append(current_word)
        current_line_dict[box].append(word_BBbox.bbox)

    # Finish
    start_next_line(current_line_dict)
    start_next_paragraph(current_paragraph_dict)
    ocr_dict_page[box] = BBox.get_maximal_box(ocr_dict_page[box])
    return ocr_dict_page

def calculate_median(img):
    np.median(np.linalg.norm(img[:,:2], axis=1))

def skew_image_and_localization(image, localization, orthogonal_bbox=True):
    """

    Args:
        image:
        localization:
        orthogonal_bbox (bool): BBox is orthogonal WRT to image

    Returns:

    """
    pass

def fill_box_with_random_lines(word_imgs,
                               bbox,
                               text_list: List[str],
                               number_of_lines=(1,3),
                               words_per_line=(1,3),
                               **kwargs):
    lines = random.randint(*number_of_lines)
    for line in range(lines):
        words = random.randint(*words_per_line)
        img, localization = fill_area_with_words(word_imgs, bbox, text_list, max_words=words)
        # skew would have to happen here

def flip(prob=.5):
    return random.random() < prob

def hasattr_not_none(obj, attr):
    return hasattr(obj, attr) and getattr(obj, attr) is not None

class BoxFiller:
    """ A re-usable object for filling in a box with words
        #   All word BBox's are relative to the box (i.e., assumes 0,0 origin) until pasted onto the image
            Initiated using some metaparameters (what will the SD horiz/vert linespacing be etc.)
            For each new section bbox:
                * mean of vert/horiz linespacing chosen
                * mean of slope chosen
                * mean of word size chosen
            #Line generation IS aware of the paragraph at all times through self.properties
            returns a list of BBox objects which contain text, paragraph/line indices

            # Options:
                Take in EITHER a list of word images and a list of text OR a generator that does both
                Take just a list in, put into boxes, if space is leftover, WRAP it

        # NOT (fully) IMPLEMENTED:
            self.channels
    """
    def __init__(self,
                 img_text_word_dict=None,
                 word_img_gen=None,
                 text_gen=None,
                 channels=3,
                 slope_sd=0.0015,
                 font_size_sd=0.05,
                 random_word_idx=False,
                 default_font_size=32,
                 default_max_words=sys.maxsize,
                 default_max_lines=sys.maxsize,
                 default_error_mode="ignore",
                 word_img_format="auto",
                 terminate_on_new_style=True,
                 indent_after_newline_character_probability=0.8,
                 random_horizontal_offset_sd=0.2,
                 use_random_vertical_offset=True,
                 ):
        """ Initiate the BoxFiller object

        Args:
            img_text_word_dict: A generator that yields a dict with at least {img: PIL.Image, text: the text of the word}
            word_img_gen: A list/generator of word images
            text_gen: A list/generator of text
            channels: Number of channels in the images
            slope_sd: Standard deviation of the slope of a line of text
            font_size_sd: Standard deviation of font
            random_word_idx: Whether to randomly select a word from the word_img_gen/text_gen
            default_font_size: Default font size
            default_max_words: Default maximum number of words to put in a box
            default_max_lines: Default maximum number of lines to put in a box
            word_img_format: Format of the word images; only numpy01, PIL and auto implemented
                                                        numpy01 -> 0-255 PIL
            indent_after_newline_character_probability: Probability of indenting after a newline character \n in a word

            Challenges between dealing with the filler, the layout, and the word generators:
                * The filling is easier to do before the layout is finalized
                * We need to be able to change word generators (font/hw) mid-paragraph
                * We may want ot adjust paragraph filling parameters based on the generator

        """
        self.word_img_format = word_img_format
        self.indent_after_newline_character_probability = indent_after_newline_character_probability
        self.img_text_pair_gen, self.word_img_gen, self.text_gen = img_text_word_dict, word_img_gen, text_gen

        if self.img_text_pair_gen is not None or self.word_img_gen is not None or self.text_gen is not None:
            self.setup_content_gen(img_text_word_dict, word_img_gen, text_gen)

        self.channels = channels
        self.slope_sd = slope_sd
        self.font_size_sd = font_size_sd
        self.random_word_idx = random_word_idx
        self.default_font_size = default_font_size
        self.default_max_words = default_max_words
        self.default_max_lines = default_max_lines
        self.default_error_mode = default_error_mode
        self.horizontal_offset_sd = random_horizontal_offset_sd
        self.use_random_vertical_offset = use_random_vertical_offset
        self.reset_document_level_params()

    def setup_content_gen(self, img_text_pair_gen, word_img_gen, text_gen):
        if img_text_pair_gen is None and word_img_gen is None:
            warnings.warn("No handwritten images provided, output will be black boxes")
            length = 512
            word_img_gen = [np_img_to_pil(np.zeros([32, 64, 1])), np.zeros([33, 128, 1])] * length
            text_gen = ["short", "long"] * length
            self._get_word = self._get_from_separate
            self.dataset_length = length
        elif img_text_pair_gen is not None:
            if isinstance(img_text_pair_gen, Iterator):
                self._get_word = self._get_from_joint_iterator
                self.dataset_length = sys.maxsize
            # elif isinstance(img_text_pair_gen, Queue):
            #     self._get_word = self._get_from_joint_queue
            #     self.dataset_length = sys.maxsize
            else:
                self._get_word = self._get_from_joint
                self.dataset_length = len(img_text_pair_gen)
            if word_img_gen is not None:
                warnings.warn("Handwritten images overspecified, using the img_text_pair_gen generator")

        else:
            self._get_word = self._get_from_separate
            self.dataset_length = len(word_img_gen)

        return img_text_pair_gen, word_img_gen, text_gen

    def override_text_generator(self, func_name, kwargs):
        """ Maybe we need to forcibly change which generator is being used

        Args:
            func_name:
            kwargs:

        Returns:

        """
        # if has func self.img_text_pair_gen func
        if hasattr(self.img_text_pair_gen, func_name):
            func = getattr(self.img_text_pair_gen, func_name)
            if callable(func):
                func(**kwargs)

    def try_to_update_BoxFiller_parameters_based_on_generator(self):
        if hasattr(self.img_text_pair_gen, "you_should_update_object_attributes_that_use_this_generator"):
            kwargs = self.img_text_pair_gen.you_should_update_object_attributes_that_use_this_generator()
            if kwargs:
                self.update_object_attributes(**kwargs)

    def update_object_attributes(self, **kwargs):
        for k,v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def gen_slope(self):
        return random.gauss(0, self.slope_sd)

    def size_variation_gen(self):
        return random.gauss(1, self.font_size_sd)

    def round(self, i):
        return int(round(i))
    def _get_from_joint(self, i):
        """ Should have at least img and text keys

        Args:
            i:

        Returns:

        """
        word_dict = self.img_text_pair_gen[i]
        if isinstance(word_dict, (tuple, list)):
            word_dict = {"img": word_dict[0],
                         "text": word_dict[1]}
        word_dict["img"] = self.convert_img_format(word_dict["img"])
        if "style" in word_dict:
            self.styles.add(word_dict["style"])
        return word_dict

    def _get_from_joint_iterator(self, i):
        word_dict = next(self.img_text_pair_gen)
        if isinstance(word_dict, (tuple, list)):
            word_dict = {"img": word_dict[0],
                         "text": word_dict[1]}
        word_dict["img"]  = self.convert_img_format( word_dict["img"])
        if "style" in word_dict:
            self.styles.add(word_dict["style"])
        return word_dict

    def _get_from_joint_queue(self, i):
        word_dict = self.img_text_pair_gen.get()
        word_dict["img"]  = self.convert_img_format( word_dict["img"])
        if "style" in word_dict:
            self.styles.add(word_dict["style"])
        return word_dict

    def _get_from_separate(self, i):
        return {"img":self.convert_img_format(self.word_img_gen[i]),
                "text":self.text_gen[i]
                }

    def convert_img_format(self, img):
        if self.word_img_format == "auto":
            if isinstance(img, np.ndarray):
                return np_img_to_pil(img)
            else:
                return img
        elif self.word_img_format == "numpy01":
            return np_img_to_pil(img)
        elif self.word_img_format == "PIL":
            return img
        else:
            raise NotImplementedError("Only numpy01, auto, and PIL implemented for word_img_format")
        return img


    def get_word(self, i=None):
        """ Get a word image and text
        Args:
            i (int): The index of the word to get

        Returns:
            (PIL.Image, str): The word image and text
        """
        if i is None:
            i = self.input_word_idx
        i = i % self.dataset_length

        self.try_to_update_BoxFiller_parameters_based_on_generator()

        word_dict = self._get_word(i)
        img, self.last_word_text = word_dict["img"], word_dict["text"]

        if "text_decode_vocab" in word_dict:
            self.last_word_text_decode_vocab = word_dict["text_decode_vocab"]

        to_font_size_factor = self.font_size / height(img) if self.font_size else 1
        if self.bbox.height < self.y1 + self.font_size*1.2:
            decimal_percent = to_font_size_factor * random.uniform(.9,1) # make the last line smaller
        else:
            decimal_percent = self.size_variation_gen() * to_font_size_factor
        self.last_word_img = self.rescale(img, decimal_percent)
        return self.last_word_img,self.last_word_text

    def rescale(self, img, decimal_percent):
        """ Rescale an image

        Args:
            img: A numpy array or a PIL image
            decimal_percent: 1 => no rescale, 0.5 => half size, 2 => double size

        Returns:
            The rescaled image

        """
        if isinstance(img, np.ndarray):
            img = resize_numpy(img, decimal_percent)
        elif isinstance(img, Image.Image):
            img = resize_pil(img, decimal_percent)
        else:
            raise TypeError("img must be a numpy array or a PIL image")
        return img

    def vert_space_max_gen(self):
        return random.uniform(.2, .4)
    def vert_space_min_gen(self):
        return random.uniform(0, .2)

    def reset_document_level_params(self):
        self.vertical_space_min_max = (self.vert_space_min_gen(), self.vert_space_max_gen())

    def reset_box_leveL_params(self, bbox, img=None,
                               max_lines=None,
                               max_words=None,
                               img_text_pair_gen=None,
                               word_img_gen=None,
                               text_gen=None,
                               error_mode=None,
                               font_size_override=None
                               ):
        """ Reset the BoxFiller object, usually done for each new box
            If you have "\n" in a word, then it will move to the next line and indent
        Args:
            bbox: The BBox object to fill
            img: The background image to fill the box with
            max_lines: The maximum number of lines to fill
            max_words: The maximum number of words to fill
            error_mode: How to handle errors when filling the box.
                          "expand" will expand the box to fit the text
                          "ignore" will ignore the out-of-bounds (formerly "force")
                          "terminate" will end the line
                          "skip_" will try the next word, should be combined with one of the others
                            skip_expand will skip and expand
                            skip_ignore will skip and ignore
                            skip_terminate will skip and terminate
            img_text_pair_gen: A generator that yields (img, text) pairs
            OR
            word_img_gen: A list of images to use
            text_gen: A list of text to use


        Returns:

        """
        self.styles = set()
        if not isinstance(bbox, BBox):
            bbox = BBox("ul", bbox, format="XYXY")
        self.bbox = bbox

        if img_text_pair_gen is not None or word_img_gen is not None:
            if self.random_word_idx:
                warnings.warn("Not recommended to use random_word_idx when updating the word_img_gen or img_text_pair_gen every call")
                self.random_word_idx = False
            self.img_text_pair_gen, self.word_img_gen, self.text_gen = self.setup_content_gen(img_text_pair_gen, word_img_gen, text_gen)

        if font_size_override:
            self.font_size = font_size_override
        elif hasattr_not_none(self.bbox, "font_size"):
            self.font_size = self.bbox.font_size
        else:
            self.font_size = self.default_font_size

        if hasattr_not_none(self.bbox, "max_lines"):
            max_lines = bbox.max_lines
        elif max_lines is None:
            max_lines = self.default_max_lines

        self.max_lines = max_lines

        if max_words is None:
            max_words = self.default_max_words
        self.max_words = min(max_words, self.dataset_length)

        self.bbox_list = []

        if img is None:
            self.img = np_img_to_pil(np.ones((bbox.height, bbox.width, self.channels)))
            self.paste_offset = -self.bbox[0], -self.bbox[1]

        else:
            # if not numpy, convert to numpy
            if isinstance(img, np.ndarray):
                img = np_img_to_pil(img)
            self.img = img
            self.paste_offset = 0, 0

        if self.random_word_idx:
            self.input_word_idx = random.randint(0, self.dataset_length-1)
        else:
            self.input_word_idx = 0

        self.words_used = 0
        self.max_vertical_offset_between_words = 2
        self.max_line_pixel_overlap = 5
        self.line_idx = 0
        self.line_word_idx = 0
        self.paragraph_idx = 0

        self.last_word_img = None
        self.last_word_text = ""
        self.last_word_text_decode_vocab = ""
        self.tab_spaces = 5
        if error_mode is not None:
            self.error_mode = error_mode # skip+, ignore, expand, terminate
        else:
            self.error_mode = self.default_error_mode

        self.skip_count = 0
        self.skip_limit = 3
        self.next_is_last_line = False
        self.slope = self.gen_slope()

        self.starting_x1 = self.round(random.uniform(0, self.font_size * .3))
        self.reset_horizontal_line_variation_for_each_new_line()

        if hasattr_not_none(self.bbox, "max_lines") and hasattr_not_none(self.bbox, "vertically_centered"):
            estimated_height = self.bbox.max_lines * (self.font_size* (1+np.mean(self.vertical_space_min_max))) * 1.05
            self.y1 = self.round((self.bbox.height - estimated_height) / 2)
        else:
            self.y1 = self.round(
                random.uniform(0, self.max_line_pixel_overlap))  # this allows for no margin on first line
        self.abs_y2_func_prev_line = lambda x: np.interp(x, [0, self.bbox.width], [self.y1, self.y1])

    def reset_horizontal_line_variation_for_each_new_line(self):
        """ Used as part of reset, to reset some of the idiosyncratic priors of HW

        Args:
            bbox:

        Returns:

        """
        # vert offset
        # horiz space
        # slope
        self.starting_x1 += int(random.gauss(0,self.horizontal_offset_sd) * self.font_size )
        self.starting_x1 = abs(self.starting_x1)
        self.horizontal_space_min_max = (.2, .6)

    def reset_word_variation(self):
        """ Randomize font size between words

        Returns:

        """
        pass


    def gen_valid_vertical_offset(self, x1, word_height):
        """ Generate a valid vertical offset for a word

        Args:
            x1: The x1 coordinate of the word
            word_height: The height of the word

        Returns:
            The vertical offset for the word

        """
        base_offset_from_slope = self.slope * x1

        # Word goes out the bottom of the bbox
        _max = self.bbox.height - self.y1 - word_height

        # Word interferes with previous line OR goes out the top of the bbox
        _min = max(self.abs_y2_func_prev_line(x1) - self.max_line_pixel_overlap - self.y1, -self.y1)

        # allow potentially big negative offset for first word of box if we need more space
        if self.words_used > 0:
            _min = max(_min, -self.max_vertical_offset_between_words)
        _max = min(_max, self.max_vertical_offset_between_words)

        # Word box does not have enough space for valid placement
        if _min > _max:
            return None

        offset = base_offset_from_slope + self.random_vertical_offset(_min, _max)
        offset = max(_min, min(_max, offset))
        offset = self.round(offset)
        assert offset + self.y1 + word_height <= self.bbox.height

        return offset

    def random_vertical_offset(self, _min, _max):
        if self.use_random_vertical_offset:
            return random.uniform(_min, _max)
        else:
            return 0

    def generate_random_total_horizontal_space_for_line_in_px(self, height=32):
        return int(random.uniform(*self.horizontal_space_min_max)*height)

    def generate_random_total_vertical_space_for_line_in_px(self, height=32):
        if self.use_random_vertical_offset:
            return int(random.uniform(*self.vertical_space_min_max)*height)
        else:
            # average vertical space * height
            return int(np.mean(self.vertical_space_min_max)*height)

    def make_line(self):
        """ keep adding words until you can't fit anymore in the line

        Returns:
            str: "newline" or "newline_indent" or "complete", depending what should happen next

        """
        # make sure there are enough words
        # make sure we haven't reached self.max_words
        while self.words_used < self.max_words:
            # if there's enough space, add next word, increment i
            next_word_img, next_word_text = self.get_word(self.input_word_idx)

            if next_word_text == "\n":
                return "newline_indent"

            next_word_width, next_word_height = shape(next_word_img)[:2]
            horizontal_space = self.generate_random_total_horizontal_space_for_line_in_px(next_word_height)

            if hasattr(self.img_text_pair_gen,"get_added_space"):
                horizontal_space += self.img_text_pair_gen.get_added_space() * next_word_height

            proposed_x1_after_adding = self.x1 + horizontal_space + next_word_width

            if proposed_x1_after_adding > self.bbox.width and self.line_word_idx > 0:
                return "newline"

            proposed_vert_offset = self.gen_valid_vertical_offset(self.x1, next_word_height)
            if proposed_vert_offset is None:
                if (self.line_word_idx > 0 or self.line_idx > 0):
                    return "complete" # can't fit more words without overlapping previous line
                else: # the first word is too big, 0 out offsets
                    warnings.warn("No viable space for word")
                    proposed_vert_offset = -self.y1

            # Critical errors - word is too big for entire box before we've even started
            result = self.check_for_errors(next_word_height, next_word_width, proposed_x1_after_adding)
            if result == "try_again":
                continue

            self.y1 += proposed_vert_offset
            y2 = self.y1 + next_word_height
            x2 = self.x1 + next_word_width
            assert self.y1 >= 0, "y1 is negative"
            self.words_used += 1
            self.x1s.append(self.x1)
            self.y2s.append(y2)
            new_box = BBox("ul",
                                       (self.x1, self.y1, x2, y2),
                                       line_index=self.line_idx,
                                       line_word_index=self.line_word_idx,
                                       text=next_word_text,
                                       parent_obj_bbox=self.bbox,
                                       paragraph_index=self.paragraph_idx,
                                       img=next_word_img
                                       )
            if self.last_word_text_decode_vocab:
                new_box.text_decode_vocab = self.last_word_text_decode_vocab

            self.bbox_list.append(new_box)

            # Set for next iteration
            self.x1 = x2 + horizontal_space
            self.line_word_idx += 1
            self.input_word_idx += 1
        return "complete"

    def check_for_errors(self, word_height, word_width, proposed_x1_after_adding):
        """ Make sure the word fits in the box. If not, handle the error

        Args:
            word_height: the height of the word in pixels
            word_width: the width of the word in pixels
            proposed_x1_after_adding: the x1 value of the word if it were added to the line

        Returns:
            None if no error, "try_again" if the word should be skipped, "complete" if the line should be completed
        """
        if (word_height <= self.bbox.height) and (word_width <= self.bbox.width):
            return None

        if "skip" in self.error_mode:
            self.skip_count += 1
            self.input_word_idx += 1
            if self.skip_count < self.skip_limit:
                return "try_again"

        if "ignore" in self.error_mode: # if a background image is supplied, it might not be cropped out
            return None
        elif "expand" in self.error_mode:
            if word_height > self.bbox.height:
                self.bbox.expand_downward(word_height-self.bbox.height)
            if word_width > self.bbox.width:
                self.bbox.expand_rightward(proposed_x1_after_adding-self.bbox.width)
            return None
        elif "terminate" in self.error_mode:
            return "complete"

        raise ValueError("Invalid error mode")

    def is_last_line(self):
        """ Check if the current line is approximately the last line in the box
            Useful for making more believable looking paragraphs
        """
        if self.y1 + 2 * self.font_size > self.bbox.height:
            return True
        elif self.line_idx == self.max_lines - 1:
            return True
        return False

    def update_max_words(self):
        """ Limit the number of words in the last row to make more believable looking paragraphs

        Returns:

        """
        if self.is_last_line():
            last_row_words = self.bbox_list[-1].line_word_index + 1 if self.bbox_list else 10
            self.max_words = min(random.randint(1, last_row_words) + self.words_used, self.max_words)

    def gen_box_layout(self):
        """ Propose words to fill box with, line-by-line
            Localization stored in self.bbox_list
        """
        while self.line_idx < self.max_lines:
            self.reset_horizontal_line_variation_for_each_new_line()
            self.x1 = self.starting_x1

            if self.line_idx == 0 and hasattr_not_none(self.bbox, "indent"):
                self.x1 += self.bbox.indent

            self.x1s, self.y2s = [], []

            # Realistic paragraph ending on last line
            if self.max_lines > 1:
                self.update_max_words()

            result = self.make_line()

            if "newline" in result:
                self.line_idx += 1
                self.line_word_idx = 0
                self.abs_y2_func_prev_line = lambda x, _x1s=tuple(self.x1s), _y2s=tuple(self.y2s),: np.interp(x, _x1s,_y2s)
                self.y1 = max(self.y2s) + self.generate_random_total_vertical_space_for_line_in_px(height=height(self.last_word_img))
            if "indent" in result and flip(self.indent_after_newline_character_probability):
                self.starting_x1 += self.generate_random_total_horizontal_space_for_line_in_px(height=width(self.last_word_img)) * self.tab_spaces
            elif result == "complete":
                return "complete"

    def fill_box(self, bbox, *args, **kwargs):
        """ Fill box orderly, line by line

        Args:
            bbox: BBox object
            *args:
            **kwargs:

        Returns:

        """
        self.reset_box_leveL_params(bbox,
                                    *args, **kwargs)
        self.gen_box_layout()
        self.paste_images()

        return {
                "img":self.img,
                "bbox_list": self.bbox_list,
                "styles":self.styles
                }

    def randomly_fill_box_with_words(self, bbox,
                                     allow_overlap=True,
                                     max_tries=2,
                                     font_size_override_range=None,
                                     *args,
                                     **kwargs):
        """ Fill a box with words, randomly rotating and scaling them, used for paragraph note boxes

        Args:
            bbox:
            *args:
            **kwargs:

        Returns:

        """
        self.reset_box_leveL_params(bbox, *args, **kwargs)
        self.words_attempted = 0
        failures = 0

        while self.words_used < self.max_words and self.words_attempted < self.max_words*2:
            if font_size_override_range:
                self.font_size = random.randint(*font_size_override_range)
            self.words_attempted += 1
            next_word_img, next_word_text = self.get_word(self.input_word_idx)
            # plt.hist(np.random.exponential(4, 1000)); plt.show()
            # Rotate the image by the angle
            if np.random.choice([True, False]):
                new_angle = np.random.exponential(4) * np.random.choice([-1, 1])
                next_word_img = next_word_img.rotate(new_angle, expand=True, fillcolor='white')

            # Generate a random position within the bounding box
            max_valid_x1, max_valid_y1 = bbox.width - width(next_word_img), bbox.height - height(next_word_img)
            x = random.uniform(0, max_valid_x1)
            y = random.uniform(0, max_valid_y1)

            # make sure the word is not outside the box
            if max_valid_x1 < 0 or max_valid_y1 < 0:
                continue

            new_bbox = BBox("ul",
                                       (x,
                                        y,
                                        x + width(next_word_img),
                                        y + height(next_word_img)),
                                       line_index=0,
                                       line_word_index=self.line_word_idx,
                                       text=next_word_text,
                                       parent_obj_bbox=self.bbox,
                                       paragraph_index=self.paragraph_idx,
                                       img=next_word_img
                                       )

            if allow_overlap or not any(new_bbox.intersects(other_bbox) for other_bbox in self.bbox_list):
                    self.bbox_list.append(
                        new_bbox
                    )
                    self.words_used += 1
                    self.line_word_idx += 1
            else:
                failures += 1
                if failures >= max_tries:
                    break

        self.paste_images()

        return {
                "img":self.img,
                "bbox_list": self.bbox_list,
                "styles":self.styles
                }

    def paste_images(self):

        for i, bbox in enumerate(self.bbox_list):

            # update the bbox to be relative to the page
            bbox.offset_origin(offset_y=self.bbox[1],
                                offset_x=self.bbox[0])

            composite_images_PIL(self.img, bbox.img, bbox, offset=self.paste_offset)
            bbox.img = None # free up memory

        return self.img, self.bbox_list


def fill_area_with_words(word_imgs,
                         bbox,
                         text_list: List[str],
                         horizontal_space_min_max=[.2,.5],
                         vertical_space_min_max=[.3,.4],
                         max_vertical_offset_between_words=1,
                         horizontal_min_to_fit=False,
                         vertical_min_to_fit=False,
                         limit_vertical_offset_to_stay_in_box_on_last_line=True,
                         follow_previous_line_slope=False,
                         prevent_line_overlap_from_vertical_drift=True,
                         error_handling: Literal['ignore',
                                                 'skip_bad',
                                                 "expand",
                         ]="ignore",
                         force_crop=True,
                         max_attempts=3,
                         scale=1,
                         max_lines=None,
                         max_words=None,
                         indent_new_paragraph_prob=.5,
                         slope=0,
                         slope_drift=(0,0), ):

    """ DEPRECATED: USE BoxFiller class instead

        TODO: Line can still be too long under skip_bad
        TODO: Slope not reliable, warp should be a postprocessing step
                    Since succeeding lines should have the same slope as the previous
                    Also can't hvae massive white space between lines
        Takes a bounding box, list of images, and a list of words and fills the bounding box with the word images.
        x1,y1 is the top left corner of the box
        x2,y2 is the bottom right corner of the box
    Args:
        word_imgs:
        bbox (x1,y1,x2,y2):
        text_list List[str]: list of text words to put in, necessary because all may not fit and needs to return actual list
        horizontal_space_min_max (0,1): spacing as a % of image height
        vertical_space_min_max (0,1): spacing as a % of image height
        TODO: horizontal_min_to_fit (bool): use minimum horizontal space if it probably won't fit
        TODO: vertical_min_to_fit (bool): use minimum vertical space if it probably won't fit
        TODO: vertical_drift: how much vertical space can change between words
        error_handling (str): skip_bad - skip this word/line and move to next one; set max_attempts to give up
                              ignore - paste AT LEAST 1 word in the line, even if it exceeds the boundary of the box, crop later
                              expand - make canvas bigger (line wider, box taller, etc.)
                                          # when 1 word is too big for a line
                                          # when the lines are too tall for the box
                                          # will not continually add lines however
        scale=1,
        max_lines=None,
        max_words=None,
        max_attempts (int): if doing skip_bad, will try multiple times
        slope:
        slope_drift:
        TODO: rotate words to align with slope
        TODO: option to return 8 parameter bboxes OR 4 parameter bboxes

    Returns:

    - mcmc - choose offset from previous line
           - sum offsets
           - find vocab_size/min
           - this is the line thickness

    TODO: Handle words that are too wide for bbox
    """
    if not text_list is None:
        word_text_list = list(zip(word_imgs, text_list))
    else:
        word_text_list = word_imgs

    if max_words is None:
        max_words = len(word_text_list)

    # A function that determines the bottom of the line
    abs_y2_func_prev_line = lambda x: np.interp(x, [0, max_line_width], [0, 0])

    word_attempts = 0
    line_attempts = 0
    paragraph_number = -1

    text = ""
    bbox_list = []
    page_lines = []
    current_line = []
    cum_y1_pos = 0 # start at 0, increase as we add lines

    # Ys attributes of lines at box level
    line_height_list = [0]
    vertical_line_spaces = [0]
    starting_height_with_vertical_space = [0]

    # Xs,Ys within a line
    cumulative_height_list = []
    y1s_within_line = []
    x1s_within_line = []
    bbox_list = []

    x1_start = x_end = 0
    #rescale_func = lambda img: np.array(img) / 255.0 if np.max(np.array(word_imgs[0][0])) > 1 else lambda img:img
    rescale_func = lambda img: np.array(img) / 255.0 if np.max(np.array(word_imgs[0][0])) > 1 else np.asarray(img)

    resize_func = resize_numpy if scale != 1 else lambda w, scale:w

    out_text = []

    # Just use a very large BBox, user must specify word limit
    if bbox is None:
        bbox = BBox([0,0,100000,100000])
        assert max_words

    max_line_width = int(bbox[2]-bbox[0])
    max_height = int(bbox[3]-bbox[1])

    def random_horizontal_space(height):
        return int(random.uniform(*horizontal_space_min_max)*height)

    def end_of_the_line():
        nonlocal abs_y2_func_prev_line, y1s_within_line, current_line, x1s_within_line, line_attempts, x1_start, x_end, cum_y1_pos, slope

        # Make sure non-trivial line
        if not y1s_within_line:
            return
        # Line can drift up/down, shift so min=0 offset; line can be squashed later?
        y1s_cum_offset = np.cumsum(y1s_within_line)
        y1s_cum_offset -= np.min(y1s_cum_offset)

        y2s_within_line = [w.shape[0] + y1s_cum_offset[i] for i, w in enumerate(current_line)]
        y_height_for_current_line = max(y2s_within_line)

        # Add to list of functions to estimate the bottom of the line; YOU MUST COPY THEM FIRST
        # _xs, _ys = x1s_within_line.copy(), y2s_within_line.copy()
        abs_y2_func_prev_line = lambda x,x1s=x1s_within_line,y2s=y2s_within_line,cum_y1=cum_y1_pos: np.interp(x, x1s, y2s) + cum_y1

        #### ALWAYS GENERATE 1 LINE EVEN IF TOO BIG ####
        # Make sure first line is not too tall!!!
        # if y_height_for_current_line > max_height: # mostly happens for first line, so try again or quit
        #     warnings.warn("Line was too tall for box")
        #     if "skip_bad" in error_handling:
        #         line_attempts += 1
        #         if line_attempts >= max_attempts:
        #             warnings.warn(f"First line too tall despite {line_attempts} attempts")
        #             return "break"
        #         return "continue"
        #     else:
        #         logger.debug(f"Too tall on line {len(page_lines)}")
        #         return "break"  # no reason to generate more lines, just end

        line_height_list.append(y_height_for_current_line)
        line_tensor = np.ones([y_height_for_current_line, x_end], dtype=np.float32)
        for i, _word in enumerate(current_line):
            line_tensor[y1s_cum_offset[i]:y1s_cum_offset[i] + _word.shape[0], x1s_within_line[i]:x1s_within_line[i] + _word.shape[1]] = _word

        # Naively concatenate spaces and words; instead, we define array and replace it to handle vertical space
        # page_lines.append(np.concatenate(current_line, 1))
        page_lines.append(line_tensor)

        current_line = []

        cumulative_height_list.append(y1s_cum_offset)

        x1s_within_line, y1s_within_line, y2s_within_line = [], [], []
        x1_start = 0
        # Ys
        line_height_list.append(y_height_for_current_line)
        vert_space = int(random.uniform(*vertical_space_min_max))
        starting_height_with_vertical_space.append(vert_space + y_height_for_current_line)
        vertical_line_spaces.append(vert_space)

        cum_y1_pos += starting_height_with_vertical_space[-1]
        # If the next line puts us over the limit
        # if starting_height_with_vertical_space[-1] > max_height:
        # Doesn't account for vertical offsets; as long as these are small, shouldn't matter;
            # Since image size is enforced with skip_bad/force, these would just be cropped slightly

        if cum_y1_pos + word_img.shape[0] > max_height or end_of_page:
            # out of space for new line, end it!
            # if it was the last word, don't worry about it
            return "end_of_page"

        line_attempts = 0
        slope += random.uniform(*slope_drift)
    def add_next_word_bbox():
        nonlocal x1_start, x_end

        if not follow_previous_line_slope:
            # Use a predefined slope
            offset_vertical = slope * x1_start
        else:
            # Use the previous line's slope
            offset_vertical = abs_y2_func_prev_line(x1_start) - cum_y1_pos

        # Choose random offset
        offset_vertical += random.uniform(-max_vertical_offset_between_words, max_vertical_offset_between_words)
        offset_vertical = int(round(offset_vertical))

        if word_img.shape[0] > max_height:
            warnings.warn("Word taller than maximum allowable height in box")
            offset_vertical = 0
        else:
            override_vertical_overlap = override_end_of_box = False
            if limit_vertical_offset_to_stay_in_box_on_last_line:
                if word_img.shape[0] + offset_vertical + cum_y1_pos > max_height:
                    offset_vertical = max(0, max_height - word_img.shape[0] - cum_y1_pos)
                    override_vertical_overlap = True
            if prevent_line_overlap_from_vertical_drift:
                # get the minimum y1 without overlapping previous line
                min_y1 = abs_y2_func_prev_line(x1_start)
                if offset_vertical + cum_y1_pos < min_y1:
                    offset_vertical = int(round(min_y1 - cum_y1_pos))
                    override_end_of_box = True
            if override_end_of_box and override_vertical_overlap:
                return False

        current_line.append(word_img)
        x1s_within_line.append(x1_start)
        y1s_within_line.append(offset_vertical)
        out_text.append(word)

        x_end = x1_start + word_img.shape[1]

        bbox_list.append( # Y-positions to be calcuated/updated at the end
            BBox("ul",
                 [x1_start,0,x_end, word_img.shape[0]],
                 line_index=len(page_lines),
                 line_word_index=len(current_line)-1,  # word was just added
                 text=out_text[-1],
                 parent_obj_bbox=bbox,
                 paragraph_index=paragraph_number)
        )

        # x_start for new word - must be after localization
        horizontal_space = random_horizontal_space(word_img.shape[0])
        x1_start = x_end + horizontal_space
        return True

    def new_paragraph(word_img):
        nonlocal x1_start, paragraph_number
        if utils.flip(indent_new_paragraph_prob):
            horizontal_space = random_horizontal_space(word_img.shape[0]) * random.randint(0,6)
            x1_start += min(horizontal_space, int(max_line_width/4))
        paragraph_number += 1

    """
    lines
        [{box,text,words:[{box,text,id:[0,0,0,0]}]
                        page,paragraph,line,word
    """

    end_of_page = False
    # WORD BUILDING LOOP
    for ii,(word_img,word) in enumerate(word_text_list):
        """
        For Y's:
            # line starting position -- can't be known until all offsets are calculated
            # where the previous line left off
            # 
        """
        word_img = resize_func(rescale_func(word_img), scale)

        # First paragraph
        if len(page_lines)==x1_start==0:
            new_paragraph(word_img)

        last_word = ii+1 >= max_words

        # Skip word if it is wider than max line width
        if "skip_bad" in error_handling:
            if word_img.shape[1] > max_line_width or word_img.shape[0] > max_height:
                word_attempts += 1
                if word_attempts >= max_attempts:
                    warnings.warn(
                        f"Tried {word_attempts} times, but specified line too narrow for words, e.g. {word}")
                    break
                continue

        word_attempts = 0

        # Check Line/Page End Conditions
        end_of_line = x1_start + word_img.shape[1] > max_line_width
        end_of_page = (max_lines and len(page_lines) >= max_lines)

        end_of_page = ""
        if end_of_line: # Don't add next word if it would make the line too long
            end_of_page = end_of_the_line()
        elif out_text and out_text[-1] and out_text[-1][-1] == "\n":
            end_of_page = end_of_the_line()
            new_paragraph(word_img)

        if end_of_page == "end_of_page":
            break

        success = add_next_word_bbox()
        if not success:
            break

    end_of_the_line()

    # if line is too long, max width equals that line, part of the "expand" paradigm
    # this should only happen if a single word is too long for a line
    # this could expand off of the page
    if error_handling=="expand":
        actual_max_line_width = int(max(max_line_width, *[line.shape[1] for line in page_lines]))
        if actual_max_line_width > max_line_width:
            if isinstance(bbox, BBox):
                bbox.expand_rightward(actual_max_line_width - max_line_width)
            else:
                bbox[2] += actual_max_line_width - max_line_width
            max_line_width = actual_max_line_width
    cum_height = np.cumsum(starting_height_with_vertical_space)

    # Put lines together
    page_img = []
    for i, line in enumerate(page_lines):
        vertical_line_space = np.ones([vertical_line_spaces[i], max_line_width])
        if error_handling == "ignore" and line.shape[1] > max_line_width:
            line = line[:,:max_line_width]
        right_pad_ = np.ones([line.shape[0], max_line_width - line.shape[1]])
        page_img.append(np.concatenate([line, right_pad_], 1))
        page_img.append(vertical_line_space)

    # Update the Y-positions in the localization, now that lines are spaced etc.
    for ii, _bbox in enumerate(bbox_list):
        if _bbox.line_number >= len(page_lines): # words were added, but line was ulitmately excluded
            bbox_list = bbox_list[:ii] # truncate lines not added
            break
        intra_line_offset = cumulative_height_list[_bbox.line_number][_bbox.line_word_index]
        final_y = int(cum_height[_bbox.line_number]+intra_line_offset)

        # Adjust for final y-position & offset initial bbox location
        _bbox.offset_origin(offset_y=final_y+bbox[1],
                            offset_x=bbox[0])

    if page_img:
        page = np.concatenate(page_img, 0)

        if force_crop:
            page = page[:max_height,:max_line_width]

        return page * 255, bbox_list
    else:
        return np.array([]), []

def resize_image_to_bbox(img:Image.Image,
                         bbox:Tuple[float, ...],
                         resize_method:Literal["fit","height","none"]="fit"):
    resample=Image.BICUBIC
    new_size_x, new_size_y = img.size
    old_x, old_y = bbox[2] - bbox[0], bbox[3] - bbox[1]
    scale_factor_width = old_x / new_size_x
    scale_factor_height = old_y / new_size_y
    if resize_method == "fit":
        scale_factor = min(scale_factor_width, scale_factor_height)
        img = img.resize([int(old_x * scale_factor),
                         int(old_y * scale_factor)],
                         resample=resample)
    elif resize_method == "width_only":
        img = img.resize([int(new_size_x * scale_factor_width),
                         int(new_size_y * scale_factor_width)],
                         resample=resample)
    elif resize_method == "height_only":
        img = img.resize([int(new_size_x * scale_factor_height),
                         int(new_size_y * scale_factor_height)],
                         resample=resample)
    return img

def resize_numpy(img, decimal_percent):

    width = ceil(img.shape[1] * decimal_percent)
    height = ceil(img.shape[0] * decimal_percent)
    dim = (width, height)
    if width > 0 and height > 0:
        return cv2.resize(img, dim)
    else:
        return img.reshape(height,width)

def resize_pil(img, decimal_percent):
    return img.resize((ceil(img.size[0] * decimal_percent), ceil(img.size[1] * decimal_percent)))

def draw_bbox(img, bboxs):
    img1 = ImageDraw.Draw(img)
    for bbox in bboxs:
        img1.rectangle(bbox)
    return img1

def offset_bboxes(bboxes, origin):
    for i, bbox in enumerate(bboxes):
        bboxes[i]=bbox + origin

if __name__=='__main__':
    pickled_data = pickle.dumps(BoxFiller())

