import warnings
from collections import defaultdict
import random
from collections import defaultdict
from math import ceil
import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from typing import Union, Optional, Tuple, Callable
from docgen.dataset_utils import ocr_dataset_to_coco
from PIL import Image

from docgen.bbox import BBox
from docgen.render_doc import fill_area_with_words, composite_images_PIL, BoxFiller
from docgen.utils.utils import display
from docgen.render_doc import convert_to_ocr_format

def flip(prob=.5):
    return random.random() < prob

class DocBox(BBox):
    """
    An abstraction of a bounding box that can be used to represent different parts of a document.
    ALL BBOXes are ASSUMED TO BE IN XYXY FORMAT

    All DocBoxes have a BBox which defines the outer limit of the box, but also a writable BBox, which is includes some
        interior margin/padding, where text is written if the BBox is in a writable category. HOWEVER, margins can be negative.

    """
    def __init__(self,
                 bbox,
                 bbox_writable=None,
                 category="",
                 children=None,
                 parent=None,
                 root=None,
                 origin: Literal['ul', 'll'] = "ul",
                 font_size=None,
                 id=None,
                 max_lines=None,
                 uncle=None,
                 nephews=None,
                 vertically_centered=None):
        """

        Args:
            bbox:
            bbox_writable:
            category:
            children:
            parent:
            origin:
            line_number:
            line_word_index:
            text:
            parent_obj_bbox:
            paragraph_index:
            font_size:
            id:
            max_lines (int): The maximum number of lines that can be written in this box
            uncle: E.g., a margin note is a descendant of the corresponding paragraph, but an uncle of the MARGIN
                    We recursive generate text based on parent-chlid relationships, but we track layouts by uncle/nephew relationships
        """
        super().__init__(origin,bbox)
        self.category = category
        self.children = children if not children is None else []
        self.nephews = nephews if not nephews is None else []
        self.parent=parent
        self.font_size = font_size
        self.id = id
        self.max_lines = max_lines
        self.uncle = uncle
        self.vertically_centered = vertically_centered
        self.root = root

        if self.root is None:
            if self.parent:
                if self.parent.root is None:
                    self.root = self.parent
                else:
                    self.root = self.parent.root

        if not parent is None:
            parent.children.append(self)

            if parent.vertically_centered is not None:
                self.vertically_centered = parent.vertically_centered

        if not uncle is None:
            uncle.nephews.append(self)


        if bbox_writable is None:
            self.bbox_writable = self
        else:
            self.bbox_writable = BBox("ul",bbox=tuple(int(x) for x in bbox_writable))

    def prune_child(self, child):
        self.children.remove(child)
        child.parent = None

    def shorten(self, amount):
        self.expand_downward(-amount)
        self.bbox_writable.expand_downward(-amount)

class SectionTemplate:
    def __init__(self,
                 top_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 bottom_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 left_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 right_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 ignore_margins:bool=False,
                 lines_rng:Union[Tuple[float, float], None]=(1,10),
                 words_rng:Union[Tuple[float, float], None]=None,
                 font_scale_factor_rng:Union[Tuple[float, float], None]=(.05, None),
                 height_as_percent_of_page_rng:Union[Tuple[float, float], None]=None,
                 min_height_pixels=30,
                 probability_existence=1.0,
                 probability_blank=0.0,
                 probability_vertically_centered=0,
                 ):
        """ Define possible ranges of font-rescaling factors, number of lines, margins, etc.

            If multiple ranges are specified (words, lines, etc.), it will terminate at the first stop condition.

        Args:
            top_margin (Tuple[float, float]): The range of possible top margins, as a fraction of the box height
            bottom_margin (Tuple[float, float]): The range of possible bottom margins, as a fraction of the box height
            left_margin (Tuple[float, float]): The range of possible left margins, as a fraction of the box width
            right_margin (Tuple[float, float]): The range of possible right margins, as a fraction of the box width
            ignore_margins (bool): If True, ignore the margins and just generate a section box that fills the parent box
            lines_rng (Tuple[float, float]): The range of possible number of lines, can be less if it runs out of words
            words_rng (Tuple[float, float]): The range of possible number of words per line
            font_scale_factor_rng (Tuple[float, float]): The range of possible font rescaling factors
            height_as_percent_of_page_rng (Tuple[float, float]): The range of possible height of the section box, as a fraction of the page height
            min_height_pixels (int): The minimum height of the section, in pixels
            probability_existence (float): The probability that this section generally exists at all on the PAGE (e.g., maybe no margin notes)
            probability_blank (float): The probability that a section box will be blank, i.e. not exist
            probability_vertically_centered (float): The probability that this section box will be vertically centered

        """
        self.lines_rng = lines_rng
        self.words_rng = words_rng
        self.font_scale_factor_rng = font_scale_factor_rng
        self.min_height_pixels = min_height_pixels
        self.height_as_percent_of_page_rng = height_as_percent_of_page_rng
        self.probability_existence = probability_existence
        self.probability_blank = probability_blank
        self.probability_vertically_centered = probability_vertically_centered

        if ignore_margins:
            self.generate_margin_box = self.naive
        else:
            self.top_margin = top_margin
            self.bottom_margin = bottom_margin
            self.left_margin = left_margin
            self.right_margin = right_margin

    def gen_max_words(self):
        if self.words_rng is None:
            return None
        else:
            return self._sample_int(self.words_rng)
    def gen_max_lines(self):
        return self._sample_int(self.lines_rng)

    def generate_height_in_pixels(self, page_height=None):
        """ Maybe add a method to take the max of the two methods
            NOT USED -> Preferring linespaces-inspired box sizing
        Args:
            page_height:
            font_size:
            method:

        Returns:

        """
        return self._sample_value(self.height_as_percent_of_page_rng) * page_height

    def max_lines_and_height(self, font_size, max_height=None):
        if max_height is not None and max_height < self.min_height_pixels:
            return None, None
        random_max_line = self.gen_max_lines()
        height = max(font_size * 1.15 * random_max_line, self.min_height_pixels)
        if max_height is not None:
            height = min(height, max_height)
        return random_max_line, height

    def font_size_add_noise(self, font_size):
        if self.font_scale_factor_rng is None:
            return font_size
        else:
            factor = self._sample_value(self.font_scale_factor_rng)
            return round(factor * font_size)

    def height_as_percent_of_page(self):
        return self._sample_value(self.height_as_percent_of_page_rng)

    @staticmethod
    def _sample_value(rng):
        if rng is None:
            return None
        else:
            return random.uniform(*rng)
    @staticmethod
    def _sample_int(rng):
        if rng is None:
            return None
        elif rng[0]<=rng[1]:
            return random.randint(rng[0], rng[1])

    def naive(self, bbox, *args, **kwargs):
        return bbox 

    def gen_margin(self, width, height) -> tuple:
        """ Generate (random) margins for a box of given width and height

        Args:
            width: width of box
            height: width of box

        Returns:

        """
        top_margin = random.uniform(*self.top_margin) * height if self.top_margin is not None else 0
        bottom_margin = random.uniform(*self.bottom_margin) * height if self.bottom_margin is not None else 0
        left_margin = random.uniform(*self.left_margin) * width if self.left_margin is not None else 0
        right_margin = random.uniform(*self.right_margin) * width if self.right_margin is not None else 0
        return left_margin, top_margin, right_margin, bottom_margin

    def generate_margin_box(self, bbox, font_size=32, buffer=2, root_bbox=None):
        """ Generate a box with margins

        Args:
            bbox:
            font_size:
            buffer:
            root_bbox:

        Returns:

        """
        font_size += buffer
        width, height = BBox._get_dim(bbox)
        left_margin, top_margin, right_margin, bottom_margin = self.gen_margin(width, height)

        # If box is too small for even 1 line, no top/bottom margin
        if height <= font_size:
            top_margin=bottom_margin=0

        # If box+margins is too small for one line, guarantee room for at least one line
        elif height - top_margin - bottom_margin < font_size:
            max_margin = height - font_size
            current_margin = top_margin+bottom_margin
            top_margin,bottom_margin = int(max_margin * top_margin/current_margin), \
                                       int(max_margin * bottom_margin/current_margin)
         
        box = [bbox[0]+left_margin, bbox[1]+top_margin, bbox[2]-right_margin, bbox[3]-bottom_margin]

        output_bbox = [round(max(b,0)) for b in box]

        if hasattr(bbox, "root") and bbox.root and not root_bbox:
            root_bbox = bbox.root
        if root_bbox:
            output_bbox = [output_bbox[0], output_bbox[1], min(output_bbox[2], root_bbox[2]), min(output_bbox[3],root_bbox[3])]

        if output_bbox[2] <= output_bbox[0] or output_bbox[3] <= output_bbox[1]:
            return None

        return BBox("ul", output_bbox)

    def generate_margin_box_expand(self, bbox, font_size=32):
        """ Expand box VERTICALLY to fit at least one line of text

        Args:
            bbox:
            font_size:

        Returns:

        """
        width, height = BBox._get_dim(bbox)
        if height < font_size:
            new_y0 = int(max(0,bbox[1]-(height-font_size)/2))
            diff = bbox[1]-new_y0
            new_y1 = int(max(bbox[3]+height-diff,bbox[3]+(height-font_size)/2))
            bbox = bbox[0],new_y0,bbox[2],new_y1
            width, height = BBox._get_dim(bbox)

        left_margin, top_margin, right_margin, bottom_margin = self.gen_margin(width, height)

        box = [round(bbox[0] - left_margin), round(bbox[1] - top_margin), round(bbox[2] + right_margin),
               round(bbox[3] + bottom_margin)]
        bbox = [max(b, 0) for b in box]
        return BBox("ul", bbox)

def scale_font(font_size, rng):
    return round(random.uniform(*rng)*font_size)

class LayoutGenerator:
    """ Generate a layout for a document
        ALL BBOXes are ASSUMED TO BE IN XYXY FORMAT

    """

    def  __init__(self,
                  pages_per_image=(1,2),
                  width_rng:int=(1000,2000),
                  height_width_ratio_rng:int=(1,1.5),
                  page_template:Union[SectionTemplate,None]=None,
                  paragraph_template:Union[SectionTemplate,None]=None,
                  margin_notes_template:Union[SectionTemplate,None]=None,
                  page_title_template:Union[SectionTemplate,None]=None,
                  paragraph_note_template:Union[SectionTemplate,None]=None,
                  page_header_template:Union[SectionTemplate,None]=None,
                  margin_notes_width:(float,float)=(.1,.3),
                  stop_page_early_probability:int=.1,
                  font_size_pixels:(int,int)=(20,64),
                  output_path:str=None,
                  save_layout_images=False,
                  degradation_function:Callable=None,
                  text_gen=None,
                  word_img_gen=None,
                  img_text_pair_gen=None,
                  ):
        """ Generates a series of nested DocBoxes, which can be used to generate a document
            Each DocBox proposes a region, applies a margin, and then it is filled with text
            The output localization BBoxes and segmentations are then reduced to fit the rendered text as needed
            ALL BBOXES ARE in XYXY format UNTIL EXPORTING TO COCO

            Categories:
                page_header: the BBox at the top of a page before the title
                page_title: the BBox for the title of the page [FRENCH BMD: WRITABLE]
                paragraph: the BBox for the main paragraph of text, e.g. one containing the BMD [FRENCH BMD: WRITABLE]
                paragraph_note: the BBox for a note AFTER a paragraph of text (e.g. for signatures) [FRENCH BMD: WRITABLE]
                margin: the BBox where margin notes are placed, assumed 1 per page
                margin_note: a BBox within the margin where a note may be that may correspond to a paragraph [FRENCH BMD: WRITABLE]

            document
                page
                    page_header
                    page_title
                    (all_margins_box)
                    all_paragraphs_box
                        paragraph
                            margin_note
                            paragraph_note

        Args:
            pages_per_image: (min,max) number of pages per image
            width_rng: (min,max) width of the image
            height_width_ratio_rng: (min,max) ratio of height to width
            page_template: SectionTemplate for the page, specifying lines ranges, margins, font_sizes, etc.
            paragraph_template: SectionTemplate for the paragraph, specifying lines ranges, margins, font_sizes, etc.
            margin_notes_template: SectionTemplate for the margin notes, specifying lines ranges, margins, font_sizes, etc.
            page_title_template: SectionTemplate for the page title, specifying lines ranges, margins, font_sizes, etc.
            paragraph_note_template: SectionTemplate for the paragraph notes, specifying lines ranges, margins, font_sizes, etc.
            page_header_template: SectionTemplate for the page header, specifying lines ranges, margins, font_sizes, etc.
            margin_notes_width: (min,max) width of the margin notes
            stop_page_early_probability: probability of stopping a page early
            font_size_pixels: (min,max) base font size in pixels, can be modified by the SectionTemplate
            output_path: path to save the layout images to
            save_layout_images: save images of the layout, MUST BE USED WITH OUTPUT_PATH
            degradation_function: function to apply to the layout images
            text_gen: TextGenerator to use for generating text
            word_img_gen: WordImageGenerator to use for generating word images
            img_text_pair_gen: ImageTextPairGenerator to use for generating image-text pairs

        """
        self.pages_per_image = pages_per_image
        self.width_rng = width_rng
        self.height_width_ratio_rng = height_width_ratio_rng

        # These categories will be SHRUNKEN to fit the space the children actually used
        self.writable_categories = ['paragraph',
                                    'margin_note',
                                    'page_title',
                                    'paragraph_note',
                                    'page_header',]



        self.dont_shrinkwrap_categories = ['page',
                                       "document",                                       
                                       ]

        self.output_path = output_path
        self.save_layout_images = save_layout_images
        self.degradation_function = degradation_function

        # Set self.templates
        for template in ["page_template", 
                         "paragraph_template",
                         "margin_notes_template",
                         "page_title_template",
                         "paragraph_note_template",
                         "page_header_template"]:
            # if not locals()[template] is None:
            #     assert isinstance(locals()[template], SectionTemplate)
            if locals()[template] is None:
                warnings.warn(f"Template {template} is None, using default SectionTemplate")
                setattr(self, template, SectionTemplate(ignore_margins=True))
            else:
                setattr(self, template, locals()[template])

        self.margin_notes_width = margin_notes_width

        self.stop_page_early_probability = stop_page_early_probability

        if isinstance(pages_per_image, int):
            self.pages_per_image = [pages_per_image,pages_per_image]
        else:
            self.pages_per_image = pages_per_image

        self.font_size_pixels = font_size_pixels
        self.text_gen = text_gen
        self.word_img_gen = word_img_gen
        self.img_text_pair_gen = img_text_pair_gen


        self.filler = BoxFiller(img_text_word_dict=self.img_text_pair_gen,
                                word_img_gen=self.word_img_gen,
                                text_gen=self.text_gen,
                                random_word_idx=True)

    def make_one_image(self,i):
        """ Generates a single image, layout, and ocr

        Args:
            i(int): the index of the image

        Returns:
            name(str): the name of the image
            ocr(OCR): the ocr, i.e., nested DocBoxes
            image(Image): the image
        """
        name = f"{i:07.0f}"
        image = None

        # Make sure the document is not completely blank
        while image is None:
            layout = self.generate_layout()
            image = self.render_text(layout)

        if self.output_path:
            save_path = self.output_path / f"{name}.jpg"
            if self.degradation_function:
                image = self.degradation_function(image)
            image.save(save_path)

            if self.save_layout_images:
                layout_img = self.draw_doc_boxes(layout)
                layout_img.save(self.output_path / f"{name}_layout.jpg")

        ocr = self.create_ocr(layout, id=i, filename=name)
        return name, ocr, image


    def generate_layout(self) -> DocBox:
        """ Generate the layout for one section, i.e., a DocBox, which might represent a paragraph, margin note, etc.

        Returns:
            layout(DocBox): the layout for a single image

        """
        pages = random.randint(*self.pages_per_image)
        page_width = random.randint(*self.width_rng)
        height = page_width * random.uniform(*self.height_width_ratio_rng)
        width = pages * page_width

        self.font_size = random.randint(*self.font_size_pixels)
        current_x = 0
        layout = DocBox(bbox=(0,0,width,height), category="document")
        for page in range(0,pages):
            self.generate_page(starting_x=current_x, page_width=page_width, page_height=height, parent=layout)
            current_x += page_width

        return layout

    def generate_page(self, starting_x:int, page_width:int, page_height:int, parent=None) -> DocBox:
        """ Generate the layout for one page; this is a DocBox and will potentially include many children DocBoxes.
            The paragraph will be the parent of an associated margin note AND paragraph note, if they exist.

        Args:
            starting_x: the x coordinate of the left edge of the page
            page_width: the width of the page
            page_height: the height of the page
            parent: the parent DocBox

        Returns:
            page(DocBox): the layout for a single page
        """
        page_bbox=(starting_x, 0, starting_x+page_width, page_height)
        page_bbox_with_margins = page_bbox if self.page_template is None else self.page_template.generate_margin_box(page_bbox)
        #full_page = DocBox(bbox=bbox, bbox_writable=bbox_with_margins, parent=parent)
        page = DocBox(bbox=page_bbox,
                      bbox_writable=page_bbox_with_margins,
                      parent=parent,
                      font_size=self.font_size,
                      category="page",
                      )

        current_y = page.bbox_writable[1]

        # Page title
        if flip(self.page_title_template.probability_existence):
            title_box = self.page_title_box(page, current_y)
            current_y = title_box.y2

        # Page Header
        if flip(self.page_header_template.probability_existence):
            header_box = self.page_header(page, current_y)
            current_y = header_box.y2

        paragraph_note_permitted = flip(self.paragraph_note_template.probability_existence)

        # build margin
        if flip(self.margin_notes_template.probability_existence):
            margin_width = round(random.uniform(*self.margin_notes_width) * page.width)
            all_paragraphs_box = DocBox(bbox=[page.bbox_writable[0]+margin_width,current_y,page.bbox_writable[2],page.bbox_writable[3]],
                                        parent=page,
                                        category="all_paragraphs_box",
                                        font_size=self.font_size)
            all_margins_box = DocBox(bbox=[page.bbox_writable[0], page.bbox_writable[1], page.bbox_writable[0]+margin_width, page.bbox_writable[3]],
                                 parent=page,
                                 category="all_margins_box",
                                 font_size=self.font_size)

        else:
            all_paragraphs_box = page
            all_margins_box = None

        stop_early_prob = 0.02 # almost never leave the page without ANY paragraphs
        paragraph_id = 0
        while not flip(stop_early_prob):
            new_paragraph = self.paragraph_box(current_y, all_paragraphs_box, id=paragraph_id)
            if not new_paragraph is None:
                current_y += new_paragraph.height

                if not all_margins_box is None and not flip(self.margin_notes_template.probability_blank):
                    self.margin_note_box(new_paragraph, all_margins_box, id=paragraph_id)

                if paragraph_note_permitted and not flip(self.paragraph_note_template.probability_blank):
                    # Usually make it 1 row
                    # Usually recenter it a bit
                    new_paragraph_note_box = self.paragraph_note_box(new_paragraph, id=paragraph_id)
                    if not new_paragraph_note_box is None:
                        current_y += new_paragraph_note_box.height
                    else:
                        break
                paragraph_id += 1
            else:
                break
            stop_early_prob = self.stop_page_early_probability
        return page

    def shrink_parent(self, parent_doc_box, min_lines=5):
        """ Shrink the parent box to be a bit smaller than the current size, if it is large enough to hold at least min_lines.

        Args:
            parent_doc_box: the parent box to shrink
            min_lines: the minimum number of lines that the parent box should be able to hold

        Returns:

        """
        # probably has some extra space
        if parent_doc_box.height > parent_doc_box.font_size * 1.1 * (min_lines+1):
            parent_doc_box.shorten(parent_doc_box.font_size * 1.5)
            return True
        return False

    def paragraph_note_box(self, paragraph:DocBox, id=None, depth=0)->DocBox:
        """ Generate a paragraph note box, which is a box that is placed BELOW the pargraph and is associated with a paragraph.

        Args:
            paragraph: the paragraph box that this paragraph note is associated with
            id: the id of the paragraph box
            depth: the recursion depth

        Returns:
            paragraph_note(DocBox): the paragraph note box
        """
        max_height = paragraph.parent.y2 - paragraph.y2
        font_size = self.paragraph_note_template.font_size_add_noise(self.font_size)
        lines, height = self.paragraph_note_template.max_lines_and_height(font_size=font_size, max_height=max_height)
        if height is None:
            if depth <=0:
                if self.shrink_parent(paragraph, min_lines=self.paragraph_template.lines_rng[0]):
                    return self.paragraph_note_box(paragraph, id=id, depth=depth+1)
            return None
                #return self.paragraph_note_box(paragraph, id=id, depth=0)

        current_y = paragraph.y2
        bbox = paragraph.x1,current_y,paragraph.x2,current_y+height

        paragraph_note = DocBox(bbox,
                                bbox_writable=self.paragraph_note_template.generate_margin_box(bbox, root_bbox=paragraph.root),
                                parent=paragraph,
                                font_size=font_size,
                                category="paragraph_note",
                                id=id,
                                max_lines=lines
                                )
        return paragraph_note

    def margin_note_box(self, paragraph:DocBox, all_margin_note:DocBox, id=None)->DocBox:
        """ Generate a margin note box, which is a box that is placed in the margin and is associated with a paragraph.
            THE PARENT IS THE PARAGRAPH RIGHT NOW
            THE ALL MARGINS BOX IS THE uncle
        Args:
            paragraph: the paragraph box that this margin note is associated with
            all_margin_note: the box that contains all the margin notes
            id: the id of the paragraph box

        Returns:
            margin_note(DocBox): the margin note box
        """
        # TODO: random height offset
        bbox = all_margin_note.bbox_xyxy[0], paragraph.bbox_xyxy[1], all_margin_note.bbox_xyxy[2], paragraph.bbox_xyxy[3]
        max_lines = self.margin_notes_template.gen_max_lines()
        font_size = self.margin_notes_template.font_size_add_noise(self.font_size)

        return DocBox(bbox=bbox,
                      bbox_writable=self.margin_notes_template.generate_margin_box(bbox, font_size=font_size, root_bbox=paragraph.root),
                      category="margin_note",
                      parent=paragraph,
                      font_size=font_size,
                      id=id,
                      max_lines=max_lines,
                      uncle=all_margin_note
                      )

    def paragraph_box(self, current_y, all_paragraphs_box, id=None):
        """ Generate a paragraph box, which is a box that contains a paragraph of text.

        Args:
            current_y: The current y position to start generating the next paragraph box (layouts generated from top to bottom)
            all_paragraphs_box:
            id: the id of the paragraph box

        Returns:
            paragraph(DocBox): the paragraph box
        """

        available_space = all_paragraphs_box.bbox_writable.y2 - current_y
        font_size = self.paragraph_template.font_size_add_noise(self.font_size)

        lines, height = self.paragraph_template.max_lines_and_height(font_size=font_size, max_height=available_space)
        if lines is None:
            return None
        bbox = all_paragraphs_box.bbox_writable[0], current_y, all_paragraphs_box.bbox_writable[2], current_y+height
        return DocBox(bbox=bbox,
                      bbox_writable=self.paragraph_template.generate_margin_box(bbox, font_size=font_size, root_bbox=all_paragraphs_box.root),
                      category="paragraph",
                      parent=all_paragraphs_box,
                      font_size=font_size,
                      id=id,
                      max_lines=lines,
                      vertically_centered=flip(self.paragraph_template.probability_vertically_centered)
                      )

    def page_title_box(self, pg_box: DocBox, current_y):
        """ Given a page DocBox, generate a page title box, which is a box that contains a title for the page.

        Args:
            pg_box: the page DocBox

        Returns:
            page_title(DocBox): the page title box

        """
        font_size = self.page_title_template.font_size_add_noise(self.font_size)
        max_lines, height = self.page_title_template.max_lines_and_height(font_size=font_size, max_height=pg_box.bbox_writable.y2 - current_y)
        if height is None:
            return None
        bbox = pg_box.bbox_writable[0], current_y, pg_box.bbox_writable[2], current_y + height
        bbox_writable = self.page_title_template.generate_margin_box(bbox, font_size=font_size, root_bbox=pg_box.root)

        return DocBox(bbox=bbox,
               bbox_writable=bbox_writable,
               category="page_title",
               parent=pg_box,
               font_size=font_size,
               max_lines=max_lines
               )

    def page_header(self, pg_box: DocBox, current_y):
        """ Given a page DocBox, return a PageHeader DocBox

        Args:
            pg_box (DocBox):  The page DocBox to add the header to

        Returns:
            page_header(DocBox): the page header box

        """
        font_size = self.page_header_template.font_size_add_noise(self.font_size)
        max_height = pg_box.bbox_writable.y2 - current_y
        max_lines, height = self.page_header_template.max_lines_and_height(font_size=font_size, max_height=max_height)
        bbox = pg_box.bbox_writable[0], current_y, pg_box.bbox_writable[2], current_y + height
        bbox_writable = self.page_header_template.generate_margin_box(bbox, font_size=font_size, root_bbox=pg_box.root)

        return DocBox(bbox=bbox,
                      bbox_writable=bbox_writable,
                      category="page_header",
                      parent=pg_box,
                      font_size=font_size,
                      max_lines=self.page_header_template.gen_max_lines()
                      )

    def _draw_doc_box(self, doc_box, image):
        """ Recursive call from draw_doc_boxes

        Args:
            doc_box: doc_box to draw
            image: PIL image

        Returns:
            image: PIL image with doc_box drawn on it

        """
        for child in doc_box.children:
            image = self._draw_doc_box(child, image)
        image = BBox._draw_box(doc_box.bbox, image, "black")
        if doc_box.category in self.writable_categories:
            image = BBox._draw_center(doc_box.bbox, image, "red")
        if not doc_box.bbox_writable is None:
            image = BBox._draw_box(doc_box.bbox_writable, image, "red")
        return image

    def draw_doc_boxes(self, doc_box, image=None):
        """ Recursively draw bounding boxes on image for debugging purposes

            Args:
                doc_box: DocBox object
                image: PIL image to draw on

            Returns:
                image: PIL image with bounding boxes drawn on it

        """
        size = doc_box.size
        if image is None:
            image = Image.new("L", size, 255)
        image = self._draw_doc_box(doc_box, image)
        return image

    def _render_text(self, background_image, doc_box, **kwargs):
        """ Recursive call from render_text

        Args:
            background_image: PIL image
            doc_box: DocBox object
            **kwargs: keyword arguments to pass to the filler

        Returns:
            background_image: PIL image with text rendered on it

        """
        child_boxes = []

        # Look for nephews; nephews should not have text rendered, but should be incorporated into the layout
        if doc_box.nephews is not None:
            child_boxes = [doc_box.bbox_writable for doc_box in doc_box.nephews]

        for child in doc_box.children:
            self._render_text(background_image, child, **kwargs)
            child_boxes.append(child.bbox_writable)

        if doc_box.category in self.writable_categories:
            if doc_box.bbox_writable:
                    doc_box.update_bbox(doc_box.bbox_writable, format="XYXY")
                    self.blank_page = False
            if doc_box.category == "paragraph_note":
                box_dict = self.filler.randomly_fill_box_with_words(doc_box,
                                                                               max_words=random.randint(1,10),
                                                                               **kwargs)
                image = box_dict["img"]
                localization = box_dict["bbox_list"]
                styles = box_dict["styles"]

            else:
                box_dict = self.filler.fill_box(doc_box, **kwargs)
                image = box_dict["img"]
                localization = box_dict["bbox_list"]
                styles = box_dict["styles"]


            composite_images_PIL(background_image, image, doc_box.bbox_writable[0:2])
            doc_box.bbox_list = localization

        else:
            # aggregate localization of children
            if child_boxes:
                doc_box.bbox_list = child_boxes
            else:
                doc_box.bbox_list = [doc_box.bbox_writable]

        # Shrink BBox to fit actual text / children
        if doc_box.category not in self.dont_shrinkwrap_categories:
            if doc_box.bbox_list is None or not doc_box.bbox_list:
                # Didn't write anything in box, delete it
                # TODO: TEST THIS
                doc_box.parent.prune_child(doc_box)
            else:
                shrinkwrap_bbox = BBox.get_maximal_box(doc_box.bbox_list)
                doc_box.update_bbox(shrinkwrap_bbox, format="XYXY")
                doc_box.bbox_writable = shrinkwrap_bbox


    def render_text(self, doc_box, **kwargs):
        """ Recursively draw text in DocBox's in document

        Args:
            doc_box (DocBox): DocBox object defining where to draw text
            img_text_pair_gen (obj): A generator that returns (image, str) tuples
            **kwargs:

        Returns:
            image: PIL image with text rendered on it

        """
        size = doc_box.size
        image = Image.new("L", size, 255)
        self.blank_page = True
        self._render_text(image, doc_box, **kwargs)
        if self.blank_page:
            return None
        return image

    def _create_ocr(self, doc_box, ocr_format_master, level, ids):
        """ Recursive function called by create_ocr
            Modifies ocr_format_master in place

        Args:
            doc_box: DocBox object
            ocr_format_master: dict to be filled with ocr format
            level: int, level of doc_box in document recursion
            ids: dict, keeps track of how many boxes at each level

        """
        doc_box.id = level, ids[level]
        ids[level] += 1
        #children_categories = []
        for child in doc_box.children:
            self._create_ocr(child, ocr_format_master, level+1, ids)
            #children_categories.append(child.category)

        # A writable category will be converted to paragraphs,lines,words
        # If it is writable, BBox will be determined by words used
        if doc_box.category in self.writable_categories:
            ocr_dict_paragraph = convert_to_ocr_format(doc_box.bbox_list)
        else:
            ocr_dict_paragraph = {"paragraphs":None, "bbox": doc_box.bbox_writable}

        meta_fields = {'level':level,
                       "category": doc_box.category,
                       "id": doc_box.id,
                       }
        if doc_box.parent:
            meta_fields["parent_id"] = doc_box.parent.id

        ocr_dict_paragraph.update(meta_fields)
        ocr_format_master["sections"].append(ocr_dict_paragraph)


    def create_ocr(self, doc_box, id, filename=""):
        """ Recursively creates a dictionary of OCR data for a document which can be converted to COCO etc.

        Args:
            doc_box: DocBox object
            id: int, id of the DocBox
            filename: str, filename of the image

        Returns:

        """
        ocr_out = {"sections": [], "width": doc_box.width, "height": doc_box.height, "id":id, "filename":filename}
        ids = defaultdict(int)
        self._create_ocr(doc_box, ocr_out, level=0, ids=ids)
        return ocr_out

if __name__ == "__main__":
    page_margins = SectionTemplate()

    page_title_margins = SectionTemplate(top_margin=(-.02, .02),
                                         bottom_margin=(-.02,.02),
                                         left_margin=(-.02,.5),
                                         right_margin=(-.02,.5))
    paragraph_margins = SectionTemplate(top_margin=(-.1, .1),
                                        bottom_margin=(-.1,.1),
                                        left_margin=(-.1,.1),
                                        right_margin=(-.1,.1))
    margin_margins = SectionTemplate(top_margin=(-.1, .5),
                                     bottom_margin=(-.1,.1),
                                     left_margin=(-.1,.1),
                                     right_margin=(-.1,.1))
    paragraph_note_margins = SectionTemplate(top_margin=(-.05, .2),
                                             bottom_margin=(-.05, .2),
                                             left_margin=(-.05, .2),
                                             right_margin=(-.05, .2))
    if True:
        lg = LayoutGenerator(paragraph_template=paragraph_margins,
                             page_template=page_margins,
                             margin_notes_template=margin_margins,
                             page_title_template=page_title_margins,
                             paragraph_note_template=paragraph_note_margins,
                             pages_per_image=(1,3)
                             )

    layout = lg.generate_layout()
    image = lg.draw_doc_boxes(layout)
    display(image)
