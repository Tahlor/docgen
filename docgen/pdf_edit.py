import warnings
import random
from io import BytesIO

import cv2
from PyPDF2 import PdfFileReader, PdfFileWriter
from PyPDF2.generic import TextStringObject, NameObject, ContentStream
from PyPDF2._utils import b_
from docgen.localize import generate_localization_from_bytes, generate_localization_from_file
from math import ceil
from docgen.img_tools import *
import numpy as np
from PIL import ImageChops, ImageDraw, PpmImagePlugin
from typing import Literal, List
from docgen.bbox import BBox
from docgen import utils
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")

FONT_DIR = r"C:\Users\tarchibald\github\brian\fslg_documents\data\fonts\fonts"
CLEAR_FONTS = r"C:\Users\tarchibald\github\brian\fslg_documents\data\fonts\clear_fonts.csv"


def delete_all_content(input_bytes_io,
                       output_path=None,
                       first_page_only=True,
                       exclusion_mark=None):
    def delete():
        pass

    reader = PdfFileReader(input_bytes_io)
    writer = PdfFileWriter()
    last_page_index = 1 if first_page_only else reader.getNumPages()
    for page_idx in range(0, last_page_index): # !!! loop through all pages

        # Get the current page and it's contents
        page = reader.getPage(page_idx)

        content_object = page["/Contents"].getObject()
        content = ContentStream(content_object, reader)
        #print_all(content, reset=True)
        for operands, operator in content.operations:
            # print(operator, operands)
            if operator == b_("BDC"):
                #operands[1][NameObject("/Contents")] = TextStringObject("xyz")
                if operands:
                    # delete = exclusion_mark is None or exclusion_mark in operands[0]
                    # print(operands)
                    # if delete:
                    operands[1][NameObject("/Contents")] = TextStringObject("")

            if operator == b_("TJ"):
                if operands:
                    # delete = exclusion_mark is None or exclusion_mark in operands[0]
                    # # Delete the text!
                    # if delete:
                    operands[0] = TextStringObject("")
                    # b'TJ' [['T', 6, 'A', -6, 'Y', 5, 'L', -5, 'OR']]
        page[NameObject('/Contents')] = content
        writer.addPage(page)

    # Write the stream
    with BytesIO() as output_stream:
        writer.write(output_stream)
        output_bytes = output_stream.getvalue()
        if output_path:
            with open(output_path, "wb") as fp:
                fp.write(output_bytes)
    return output_bytes

def edit(input_bytes_io,
                       output_path=None,
                       first_page_only=True,
                       exclusion_mark=None):
    reader = PdfFileReader(input_bytes_io)
    writer = PdfFileWriter()
    last_page_index = 1 if first_page_only else reader.getNumPages()
    for page_idx in range(0, last_page_index): # !!! loop through all pages

        # Get the current page and it's contents
        page = reader.getPage(page_idx)
        contents = page.getContents()


        content_object = page["/Contents"].getObject()
        content = ContentStream(content_object, reader)
        #print_all(content, reset=True)
        for operands, operator in content.operations:
            # print(operator, operands)
            if operator == b_("BDC"):
                #operands[1][NameObject("/Contents")] = TextStringObject("xyz")
                if operands:
                    # delete = exclusion_mark is None or exclusion_mark in operands[0]
                    # print(operands)
                    # if delete:
                    operands[1][NameObject("/Contents")] = TextStringObject("")

            if operator == b_("TJ"):
                if operands:
                    # delete = exclusion_mark is None or exclusion_mark in operands[0]
                    # # Delete the text!
                    # if delete:
                    operands[0] = TextStringObject("")
                    # b'TJ' [['T', 6, 'A', -6, 'Y', 5, 'L', -5, 'OR']]
        page[NameObject('/Contents')] = content
        writer.addPage(page)

    # Write the stream
    with BytesIO() as output_stream:
        writer.write(output_stream)
        output_bytes = output_stream.getvalue()
        if output_path:
            with open(output_path, "wb") as fp:
                fp.write(output_bytes)
    return output_bytes



def create_new_textbox(page,
                       rect,
                       text="THIS IS MY NEW TEXT",
                       font_name ="Times-Roman",
                       font_size = 14,
                       draw_box=True,
                       box_color=(.25,1,.25)):
    """

    Args:
        page:
        text:
        rect: x1,y1,x2,y2
        font_name:
        font_size:
        draw_box:
        box_color:

    Returns:

    """

    if draw_box:
        page.draw_rect(rect,color=box_color)

    rc = page.insert_textbox(rect, text,
                             fontsize=font_size,
                             fontname=font_name,
                             align=1)

    return rc

class ScaleOptions:
    """ max_upscale_factor (float):
        max_downscale_factor (float):
        scale_strategy (fit_x, fit_y, None, random): scale image to fit destination based on which dimension?
        max_threshold_x: do not scale beyond e.g. 110% reference box
        max_threshold_y
        scale_jitter_factor (.9,1.1: applied after scaling has happened
        random_pad: if smaller than destination box, apply random padding
    """

    def resize(self, source_dim, dest_dim):
        pass

def composite_images(background_image, paste_image, position):
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

def composite_images2(background, paste, pos):
    """

    Args:
        background:
        paste:
        pos: (x,y) or BBox

    Returns:

    """
    if isinstance(pos, BBox):
        pos = pos.bbox[:2]

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
        bck = Image.fromarray(bck.astype(np.uint8), mode="L")
    if isinstance(fore, np.ndarray):
        fore = Image.fromarray(fore.astype(np.uint8), mode="L")
    bck = bck.crop((pos[0], pos[1], pos[0]+fore.size[0],pos[1]+fore.size[1]))
    if bck.mode == "RGB" and fore.mode != "RGB":
        fore = fore.convert("RGB")
    elif fore.mode == "RGB" and bck.mode != "RGB":
        bck = bck.convert("RGB")

    return ImageChops.multiply(fore, bck)

def shape(item):
    if isinstance(item, np.ndarray):
        return item.shape
    elif isinstance(item, (PpmImagePlugin.PpmImageFile,Image.Image)):
        return item.size

def get_x(item):
    if isinstance(item, np.ndarray):
        return item.shape[1]
    elif isinstance(item, PpmImagePlugin.PpmImageFile):
        return item.size[0]

def get_y(item):
    if isinstance(item, np.ndarray):
        return item.shape[0]
    elif isinstance(item, PpmImagePlugin.PpmImageFile):
        return item.size[1]


def estimate_space(word_imgs, vertical_space, horizontal_space):
    """ Estimate how much space is needed for horizontal_min_to_fit
        Also scale option...ugh

    Returns:

    """
    widths = [word.shape[0] + horizontal_space for word in word_imgs]
    total_length_concat = np.cumsum(widths)


def convert_to_ocr_format(localization, box="bbox", origin_offset=(0,0), section=0):

    def start_next_line(line=None):
        nonlocal current_line_num
        if line:
            current_line_num = word.line_number
            line[box] = BBox.get_maximal_box(line[box])
            line["text"] = " ".join(line["text"])
            current_paragraph_dict["lines"].append(line)
            current_paragraph_dict[box].append(line[box])
        return {box: [], "text": [], "words": []}

    def start_next_paragraph(paragraph=None):
        nonlocal current_paragraph
        if paragraph:
            current_paragraph = word.paragraph_index
            paragraph[box] = BBox.get_maximal_box(paragraph[box])
            ocr_dict_page["paragraphs"].append(paragraph)
            ocr_dict_page[box].append(paragraph[box])
        return {"lines": [], box: []}

    ocr_dict_page = {"paragraphs": [], box: []}
    current_paragraph_dict = start_next_paragraph()
    current_line_dict = start_next_line()
    current_line_num = 0
    current_paragraph = 0

    for word in localization:

        # Start new line
        if word.line_number!=current_line_num:
            current_line_dict = start_next_line(current_line_dict)

        # Start a new paragraph
        if word.paragraph_index>current_paragraph:
            current_paragraph_dict = start_next_paragraph(current_paragraph_dict)

        current_line_dict["text"].append(word.text)
        current_word = {box: word.offset_origin(origin_offset[0],
                                                origin_offset[1]).bbox,
                        "text": word.text,
                        "id": [section,
                               current_paragraph,
                               word.line_number,
                               word.line_word_index]}
        current_line_dict["words"].append(current_word)
        current_line_dict[box].append(word.bbox)

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


def fill_area_with_words(word_imgs,
                         bbox,
                         text_list: List[str],
                         horizontal_space_min_max=[.2,.5],
                         vertical_space_min_max=[.3,.4],
                         max_intraline_vertical_space_offset=1,
                         horizontal_min_to_fit=False,
                         vertical_min_to_fit=False,
                         error_handling: Literal['force',
                                                 'skip_bad',
                                                 "expand",
                                                 "force_hard",
                                                 "skip_bad_hard",
                         ]="skip_bad",
                         max_attempts=3,
                         scale=1,
                         max_lines=None,
                         max_words=None,
                         indent_new_paragraph_prob=.5,
                         slope=0,
                         slope_drift=(0,0)):

    """ TODO: Line can still be too long under skip_bad
        TODO: Slope not reliable, warp should be a postprocessing step
        Takes a bounding box, list of images, and a list of words and fills the bounding box with the word images.

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
                              force - paste word, even if it execeeds the boundary of the box
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
        word_text_list = zip(word_imgs, text_list)
    else:
        word_text_list = word_imgs

    if max_words is None:
        max_words = len(word_text_list)

    word_attempts = 0
    line_attempts = 0
    paragraph_number = -1

    text = ""
    localization = []
    page_lines = []
    current_line = []
    cum_y_pos = 0

    # Ys attributes of lines at box level
    y_start_lines = [0]
    vertical_line_spaces = [0]
    starting_height_with_vertical_space = [0]

    # Xs,Ys within a line
    list_of_y_sums = []
    ys_within_line = []
    xs_within_line = []
    localization = []

    x_start = x_end = 0
    rescale_func = lambda img: np.array(img) / 255.0 if np.max(np.array(word_imgs[0][0]))>1 else lambda img:img
    resize_func = resize if scale != 1 else lambda w,scale:w

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
        nonlocal ys_within_line, current_line, xs_within_line, line_attempts, x_start, x_end, cum_y_pos, slope

        # Make sure non-trivial line
        if not ys_within_line:
            return
        ys_sum = np.cumsum(ys_within_line)
        ys_sum -= np.min(ys_sum)

        y_height_for_current_line = max([w.shape[0] + ys_sum[i] for i, w in enumerate(current_line)])

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

        y_start_lines.append(y_height_for_current_line)
        line_tensor = np.ones([y_height_for_current_line, x_end], dtype=np.float32)
        for i, _word in enumerate(current_line):
            line_tensor[ys_sum[i]:ys_sum[i] + _word.shape[0], xs_within_line[i]:xs_within_line[i] + _word.shape[1]] = _word

        # Naively concatenate spaces and words; instead, we define array and replace it to handle vertical space
        # page_lines.append(np.concatenate(current_line, 1))
        page_lines.append(line_tensor)

        current_line = []

        list_of_y_sums.append(ys_sum)
        xs_within_line, ys_within_line = [], []
        x_start = 0
        # Ys
        y_start_lines.append(y_height_for_current_line)
        vert_space = int(random.uniform(*vertical_space_min_max))
        starting_height_with_vertical_space.append(vert_space + y_height_for_current_line)
        vertical_line_spaces.append(vert_space)

        cum_y_pos += starting_height_with_vertical_space[-1]
        # If the next line puts us over the limit
        # if starting_height_with_vertical_space[-1] > max_height:
        # Doesn't account for vertical offsets; as long as these are small, shouldn't matter;
            # Since image size is enforced with skip_bad/force, these would just be cropped slightly

        if cum_y_pos + word_img.shape[0] > max_height or end_of_page:
            # out of space for new line, end it!
            # if it was the last word, don't worry about it
            return "break"

        line_attempts = 0
        slope += random.uniform(*slope_drift)
    def add_next_word():
        nonlocal x_start, x_end

        # Use a predefined slope
        offset_vertical = slope * x_start

        # Choose random offset
        offset_vertical += random.uniform(-max_intraline_vertical_space_offset, max_intraline_vertical_space_offset)
        offset_vertical = int(round(offset_vertical))

        if word_img.shape[0] > max_height:
            warnings.warn("Word taller than maximum allowable height in box")
        current_line.append(word_img)
        xs_within_line.append(x_start)
        ys_within_line.append(offset_vertical)
        out_text.append(word)

        x_end = x_start + word_img.shape[1]

        localization.append( # Y-positions to be calcuated/updated at the end
            BBox("ul",
                 [x_start,0,x_end, word_img.shape[0]],
                 line_number=len(page_lines),
                 line_word_index=len(current_line)-1,  # word was just added
                 text=out_text[-1],
                 parent_obj_bbox=bbox,
                 paragraph_index=paragraph_number)
        )

        # x_start for new word - must be after localization
        horizontal_space = random_horizontal_space(word_img.shape[0])
        x_start = x_end + horizontal_space

    def new_paragraph(word_img):
        nonlocal x_start, paragraph_number
        if utils.flip(indent_new_paragraph_prob):
            horizontal_space = random_horizontal_space(word_img.shape[0]) * random.randint(0,6)
            x_start += min(horizontal_space, int(max_line_width/4))
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
        if len(page_lines)==x_start==0:
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

        end_of_line = x_start + word_img.shape[1] > max_line_width
        end_of_page = (max_lines and len(page_lines) >= max_lines)

        result = ""
        if end_of_line: # Don't add next word if it would make the line too long
            result = end_of_the_line()
        elif out_text and out_text[-1] and out_text[-1][-1] == "\n":
            result = end_of_the_line()
            new_paragraph(word_img)

        if result == "break":
            break
        elif result == "continue":
            continue
        add_next_word()

    end_of_the_line()

    # if line is too long, max width equals that line, part of the "expand" paradigm
    if error_handling=="expand":
        max_line_width = int(max(max_line_width, *[line.shape[1] for line in page_lines]))

    cum_height = np.cumsum(starting_height_with_vertical_space)

    # Put lines together
    page_ = []
    for i, line in enumerate(page_lines):
        vertical_line_space = np.ones([vertical_line_spaces[i], max_line_width])
        if error_handling == "force" and line.shape[1] > max_line_width:
            line = line[:,:max_line_width]
        right_pad_ = np.ones([line.shape[0], max_line_width - line.shape[1]])
        page_.append(np.concatenate([line, right_pad_], 1))
        page_.append(vertical_line_space)

    # Update the Y-positions in the localization, now that lines are spaced etc.
    for ii, _bbox in enumerate(localization):
        if _bbox.line_number >= len(page_lines): # words were added, but line was ulitmately excluded
            localization = localization[:ii] # truncate lines not added
            break
        intra_line_offset = list_of_y_sums[_bbox.line_number][_bbox.line_word_index]
        final_y = int(cum_height[_bbox.line_number]+intra_line_offset)

        # Adjust for final y-position & offset initial bbox location
        _bbox.offset_origin(offset_y=final_y+bbox[1],
                            offset_x=bbox[0])

    if page_:
        page = np.concatenate(page_, 0)

        if "hard" in error_handling:
            page = page[:max_height,:max_line_width]

        return page * 255, localization
    else:
        return np.array([]), []

def resize_image_to_bbox(img:Image.Image,
                         bbox:tuple[float, ...],
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

def resize(img, decimal_percent):

    width = ceil(img.shape[1] * decimal_percent)
    height = ceil(img.shape[0] * decimal_percent)
    dim = (width, height)
    if width > 0 and height > 0:
        return cv2.resize(img, dim)
    else:
        return img.reshape(height,width)


class PDF:
    def __init__(self, renderer,
                 ignore_chars=[],
                 mark_for_replacement_char=None):
        """

        Args:
            renderer:
            remove_chars: characters to ignore when transcribing
            mark_for_replacement_char:
        """
        self.RT = renderer
        self.ignore_chars = ignore_chars + [chr(96)]
        self.mark_for_replacement = mark_for_replacement_char

    def clnstr(self, txt):
        return txt.translate(None, ''.join(self.ignore_chars))

    def gen_word_image(self, word):
        return self.RT.render_word(word)

    def replace_text_with_images(self,
                                 pdf,
                                 new_words=None,
                                 localization=None,
                                 delete_text=True,
                                 vertical_inversion=True,
                                 composite=True,
                                 request_same_word=False,
                                 resize_words: Literal["fit","height_only","width_only","none"]="fit",
                                 first_page_only=True,
                                 use_replace_flag=False
                                 ):
        """

        Args:
            pdf:
            new_words:
            delete_text:
            vertical_inversion (bool): PDF localization origin is Lower Left, image origin is Upper Left
            composite (bool): True: composite images instead of pasting, i.e., merge/take darkest pixel
            resize_words: fit-make sure new word height/width will fit
                          height-only: make sure it is not taller than original word
                          width-only: make sure it is not wider than original word
                          none: no resize
                          truncate: NOT IMPLEMENTED; use with height-only-truncate or something
            first_page_only: only use first page of PDF
            use_replace_flag: only replace text where "replace:True" in localization
        Returns:

        """
        # User must supply words to be replaced, or it will replace it with the same one already there
        assert not new_words is None or request_same_word

        # Perform localization
        if isinstance(pdf, str) or isinstance(pdf, Path):
            with Path(pdf).open('rb') as _p:
                # For each text box, replace it with HWR resized to fit the same space
                pdf_obj = _p.read()
        else:
            pdf_obj = pdf

        pdf_io = BytesIO(pdf_obj)

        if localization is None:
            localization = generate_localization_from_bytes(pdf_io,
                                                            first_page_only=first_page_only,
                                                            mark_for_replacement=self.mark_for_replacement)
        elif first_page_only:
            localization = {0: localization[0]}

        new_localization = {}

        # delete text
        if delete_text:
            pdf_obj = delete_all_content(pdf_io,
                                         first_page_only=first_page_only,
                                         exclusion_mark=self.mark_for_replacement)

        images = convert_pdf_bytes_to_img_bytes(pdf_obj)
        if first_page_only:
            images = [images[0]]

        # Paste new word images
        for page in localization:
            if isinstance(page,int):
                new_localization[page] = {}
                img_w = images[page].size[0]
                img_h = images[page].size[1]
                new_localization[page]["localization_word"] = []
                for i, textbox in enumerate(localization[page]["localization_word"]):
                    if use_replace_flag:
                        if "replace" in textbox and not textbox["replace"]:
                            continue
                    if request_same_word:
                        # Replace existing text with same text, new font/handwriting
                        phrase_text = textbox["text"]
                    else:
                        # Use generator to find text
                        phrase_text = new_words[i]

                    # Generate new image
                    phrase_obj = self.gen_word_image(phrase_text)

                    # Convert from numpy to PIL
                    if isinstance(phrase_obj["image"], np.ndarray):
                        phrase_img = Image.fromarray(phrase_obj["image"])
                    else:
                        phrase_img = phrase_obj["image"]

                    x = textbox["norm_bbox"][0]

                    # Need to invert y coordinates and use "lower" coordinate instead of upper coordinate
                    y = 1-textbox["norm_bbox"][3] if vertical_inversion else textbox["norm_bbox"][1]
                    pos = int(x * get_x(images[page])), int(y * get_y(images[page]))

                    abs_pixel_bbox = textbox["norm_bbox"][0]*img_w,textbox["norm_bbox"][1]*img_h,textbox["norm_bbox"][2]*img_w,textbox["norm_bbox"][3]*img_h
                    if resize_words != "none":
                        phrase_img = resize_image_to_bbox(phrase_img, abs_pixel_bbox, resize_method=resize_words)

                    if composite: # i.e., something closer to taking darkest pixel
                        # My numpy version - dangerous, just keep everything in PIL or everything will get screwed up
                        # images[page] = composite_images(images[page], phrase_img, pos)

                        # PIL only
                        phrase_img = merge_img(images[page], phrase_img, pos)
                        images[page].paste(phrase_img, pos)

                    else: # naive paste
                        images[page].paste(phrase_img, pos)

                    new_localization[page]["localization_word"].append({"text": phrase_text,
                                                                        "bbox": BBox.img_and_pos_to_bbox(phrase_img,pos),
                                                                        })

                if isinstance(images[page], np.ndarray):
                    images[page] = Image.fromarray(np.uint8(images[page]))

        # Paste word image

        # loop through bounding boxes
            # get the original data for these bounding boxes???
            # render them as images
            # resize
            # paste the images
        return images, new_localization


def draw_bbox(img, bboxs):
    img1 = ImageDraw.Draw(img)
    for bbox in bboxs:
        img1.rectangle(bbox)
    return img1

def offset_bboxes(bboxes, origin):
    for i, bbox in enumerate(bboxes):
        bboxes[i]=bbox + origin

def delete_text_test():
    input_path = Path("../temp/TEMPLATE.pdf")
    output_path = input_path.parent / (input_path.stem + "_modded.pdf")
    delete_all_content(input_path, output_path)



def localization_test(input_path):
    """ Example of creating new textbox on PDF

    Returns:

    """
    return generate_localization_from_file(input_path)



if __name__ == '__main__':
    input_path = "../temp/TEMPLATE.pdf"
    output_path = "../temp/TEMPLATE_with_textbox.pdf"
    input_path = "../temp/french_census_12.pdf"
    output_path = "../temp/french_census_12_MOD.pdf"

    #drawing_test(input_path, output_path)
    localization_test(input_path)

    #### TOMORROW:
        # fix logic:
            # want to check if end of the line at the beginning, so we don't add another word to the line
            # BUT we need to add the last word to the last line too
            # Proposal:
                # have end the line be after, but have a quick check to see if the word should be added
                # THIS won't work, because then we go to the next word
            ### OR:
                # have the last item in the list be a dummy item; when we get to the dummy, end the line
                # or just call the function again
            # break/continue won't work with subfunction; just return something that relays this to the loop

    ### MAKE FORMS
    ### MAKE DATA FOR DESSURT FOR IAM