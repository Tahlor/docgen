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
from typing import Union, Optional, Tuple
from docgen.dataset_utils import ocr_dataset_to_coco
from PIL import Image

from docgen.bbox import BBox
from docgen.render_doc import fill_area_with_words, composite_images2
from docgen.utils import display
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
                 origin: Literal['ul', 'll'] = "ul",
                 font_size=None,
                 id=None,
                 max_lines=None,
                 uncle=None,
                 nephews=None):
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

        if not parent is None:
            parent.children.append(self)
        if not uncle is None:
            uncle.nephews.append(self)

        if bbox_writable is None:
            self.bbox_writable = self._bbox
        else:
            self.bbox_writable = tuple(int(x) for x in bbox)

class SectionTemplate:
    """ Generate a margin box within the parent box """

    def __init__(self,
                 top_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 bottom_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 left_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 right_margin:Union[Tuple[float, float], None]=(0.,0.1),
                 ignore_margins:bool=False,
                 lines_rng:Union[Tuple[float, float], None]=(1,10),
                 font_scale_factor_rng:Union[Tuple[float, float], None]=(.05, None),
                 height_as_percent_of_page_rng:Union[Tuple[float, float], None]=None,
                 min_height_pixels=40,
                 probability_existence=1.0,
                 probability_blank=0.0
                 ):
        self.lines_rng = lines_rng
        self.font_scale_factor_rng = font_scale_factor_rng
        self.min_height_pixels = min_height_pixels
        self.height_as_percent_of_page_rng = height_as_percent_of_page_rng
        self.probability_existence = probability_existence
        self.probability_blank = probability_blank

        if ignore_margins:
            self.generate_margin_box = self.naive
        else:
            self.top_margin = top_margin
            self.bottom_margin = bottom_margin
            self.left_margin = left_margin
            self.right_margin = right_margin
    def max_lines(self):
        return self._sample_int(self.lines_rng)

    def generate_height_in_pixels(self, page_height=None, font_size=None, method=None):
        """ Maybe add a method to take the max of the two methods

        Args:
            page_height:
            font_size:
            method:

        Returns:

        """
        if method=="max":
            return max(self.generate_height_in_pixels(page_height), self.generate_height_in_pixels(font_size))
        elif not self.height_as_percent_of_page_rng is None and not page_height is None:
            return self._sample_value(self.height_as_percent_of_page_rng) * page_height
        elif not self.lines_rng is None and not font_size is None:
            return self._sample_value(self.lines_rng)*font_size*1.1
        else:
            raise ValueError("Must specify either height_as_percent_of_page_rng/page_height or lines_rng/font_size")

    def font_scale_factor(self):
        if self.font_scale_factor_rng is None:
            return 1
        return self._sample_value(self.font_scale_factor_rng)

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
        elif rng[0]<rng[1]:
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
        bottom_margin = random.uniform(*self.bottom_margin) * height if self.top_margin is not None else 0
        left_margin = random.uniform(*self.left_margin) * width if self.top_margin is not None else 0
        right_margin = random.uniform(*self.right_margin) * width if self.top_margin is not None else 0
        return left_margin, top_margin, right_margin, bottom_margin

    def generate_margin_box(self, bbox, font_size=32):
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
         
        box = [round(bbox[0]+left_margin), round(bbox[1]+top_margin), round(bbox[2]-right_margin), round(bbox[3]-bottom_margin)]
        bbox = [max(b,0) for b in box]
        return bbox

    def generate_margin_box_expand(self, bbox, font_size=32):
        """ Expand box VERTICALLY to include margins

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
        return bbox

def scale_font(font_size, rng):
    return round(random.uniform(*rng)*font_size)

class LayoutGenerator:
    """ Generate a layout for a document
        ALL BBOXes are ASSUMED TO BE IN XYXY FORMAT

    """

    def  __init__(self,
                  pages_per_image=(1,2),
                  width_base:int=3200,
                  height_base:int=2400,
                  random_size_factor:float=2,
                  max_page_width_factor:float=1.5,
                  page_template:Union[SectionTemplate,None]=None,
                  paragraph_template:Union[SectionTemplate,None]=None,
                  margin_notes_template:Union[SectionTemplate,None]=None,
                  page_title_template:Union[SectionTemplate,None]=None,
                  paragraph_note_template:Union[SectionTemplate,None]=None,
                  page_header_template:Union[SectionTemplate,None]=None,
                  margin_notes_width:(float,float)=(.1,.3),
                  stop_page_early_probability:int=.1,
                  font_size_pixels:(int,int)=(20,64),
                  ):
        """

        Args:
            pages_per_image:
            pages_per_image_prob_weights:
            width_base:
            height_base:
            random_size_factor:
            max_page_width_factor: page cannot be wider than FACTOR*WIDTH, or smaller than WIDTH/FACTOR
            page_template:
            paragraph_template:
            paragraph_height_min_pixels:
            paragraph_height:
            margin_notes_width:
            stop_page_early_probability:
            max_lines_margin_notes:

            Categories:
                page_header: the BBox at the top of a page before the title
                page_title: the BBox for the title of the page [FRENCH BMD: WRITABLE]
                paragraph: the BBox for the main paragraph of text, e.g. one containing the BMD [FRENCH BMD: WRITABLE]
                paragraph_note: the BBox for a note AFTER a paragraph of text (e.g. for signatures) [FRENCH BMD: WRITABLE]
                margin: the BBox where margin notes are placed, assumed 1 per page
                margin_note: a BBox within the margin where a note may be that may correspond to a paragraph [FRENCH BMD: WRITABLE]

            TODO: make a super MARGIN category, super PARAGRAPH category, super TITLE/HEADING category that produces correct localizations
            document
                page
                    page_header
                    page_title
                    (all_margins_box)
                    all_paragraphs_box
                        paragraph
                            margin_note
                            paragraph_note

        """
        self.writable_categories = ['paragraph', 'margin_note', 'page_title', 'paragraph_note']

        # These categories will be SHRUNKEN to fit the space the children actually used
        # HOWEVER, ALL_MARGINS HAS NO CHILDREN!!!
        # ALSO: paragraphs needs to be shrunken NOT to fit children, ONLY to fit words...
        # WHAT IS HAPPENING RIGHT NOW? BBOX WRITABLE VS BBOX
        self.dont_shrinkwrap_categories = ['page',
                                       "document",                                       
                                       ]

        self.width = width_base
        self.height = height_base
        self.random_size_factor = random_size_factor
        self.max_page_width_factor = max_page_width_factor

        # Set self.templates
        for template in ["page_template", "paragraph_template", "margin_notes_template", "page_title_template", "paragraph_note_template", "page_header_template"]:
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

        self.width_upper = round(self.width * self.random_size_factor)
        self.width_lower = round(self.width / self.random_size_factor)
        self.height_upper = round(self.height * self.random_size_factor)
        self.height_lower = round(self.height / self.random_size_factor)

        self.font_size_pixels = font_size_pixels

    def generate_layout(self) -> DocBox:
        """ Generate the layout for one section, i.e., a DocBox, which might represent a paragraph, margin note, etc.

        Returns:

        """
        width = random.randint(self.width_lower, self.width_upper)
        height = random.randint(self.height_lower, self.height_lower)

        # Determine if 1 OR 2 pages can fit
        min_pages_factor = ceil(width / (self.width * self.max_page_width_factor))
        max_pages_factor = int(width / (self.width / self.max_page_width_factor))
        min_pages = max(min_pages_factor, 1, self.pages_per_image[0])
        max_pages = min(max(min_pages, max_pages_factor), self.pages_per_image[1])
        pages = random.randint(min_pages, max_pages+1)
        self.font_size = random.randint(*self.font_size_pixels)
        current_x = 0
        page_width = round(width / pages)
        layout = DocBox(bbox=(0,0,width,height), category="document")
        for page in range(0,pages):
            self.generate_page(starting_x=current_x, page_width=page_width, page_height=height, parent=layout)
            current_x += page_width

        return layout

    def generate_page(self, starting_x:int, page_width:int, page_height:int, parent=None) -> DocBox:
        """ Generate the layout for one page; this is a DocBox and will potentially include many children DocBoxes.
            The paragraph will be the parent of an associated margin note AND paragraph note, if they exist.

        Args:
            starting_x:
            page_width:
            page_height:
            parent:

        Returns:

        """
        page_bbox=(starting_x, 0, starting_x+page_width, page_height)
        page_bbox_with_margins = page_bbox if self.page_template is None else self.page_template.generate_margin_box(page_bbox)
        #full_page = DocBox(bbox=bbox, bbox_writable=bbox_with_margins, parent=parent)
        page = DocBox(bbox=page_bbox,
                      bbox_writable=page_bbox_with_margins,
                      parent=parent,
                      font_size=self.font_size,
                      category="page")

        current_y = page.bbox_writable[1]

        # Page title
        if flip(self.page_title_template.probability_existence):
            title_box = self.page_title_box(page, current_y)
            current_y += title_box.bbox[3]

        # Page Header
        if flip(self.page_header_template.probability_existence):
            header_box = self.page_header(page, current_y)
            current_y += header_box.bbox[3]

        paragraph_note = flip(self.paragraph_note_template.probability_existence)

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

                if not all_margins_box is None and not flip(self.margin_notes_template.probability_existence):
                    self.margin_note_box(new_paragraph, all_margins_box, id=paragraph_id)

                if paragraph_note and not flip(self.paragraph_note_template.probability_existence):
                    # Usually make it 1 row
                    # Usually recenter it a bit
                    new_paragraph_note_box = self.paragraph_note_box(new_paragraph, id=paragraph_id)
                    current_y += new_paragraph_note_box.height
                paragraph_id += 1
            else:
                break
            stop_early_prob = self.stop_page_early_probability
        return page

    def paragraph_note_box(self, paragraph:DocBox, id=None)->DocBox:
        """ Generate a paragraph note box, which is a box that is placed BELOW the pargraph and is associated with a paragraph.

        Args:
            paragraph:
            id:

        Returns:

        """
        font_size = scale_font(self.font_size, (.9,1.1))
        height = self.paragraph_note_template.generate_height_in_pixels(font_size=font_size)
        bbox = paragraph.bbox_xyxy[0],paragraph.bbox_xyxy[3],paragraph.bbox_xyxy[2],paragraph.bbox_xyxy[3]+height
        paragraph_note = DocBox(bbox,
                                bbox_writable=self.paragraph_note_template.generate_margin_box(bbox),
                                parent=paragraph,
                                font_size=font_size,
                                category="paragraph_note",
                                id=id,
                                max_lines=self.paragraph_note_template.max_lines()
                                )
        return paragraph_note

    def margin_note_box(self, paragraph:DocBox, all_margin_note:DocBox, id=None)->DocBox:
        """ Generate a margin note box, which is a box that is placed in the margin and is associated with a paragraph.
            THE PARENT IS THE PARAGRAPH RIGHT NOW
            THE ALL MARGINS BOX IS THE uncle
        Args:
            paragraph:
            all_margin_note:
            id:

        Returns:

        """
        # TODO: random height offset
        bbox = all_margin_note.bbox_xyxy[0], paragraph.bbox_xyxy[1], all_margin_note.bbox_xyxy[2], paragraph.bbox_xyxy[3]
        font_size = scale_font(self.font_size, (.9,1.1))
        max_lines = self.margin_notes_template.max_lines()

        return DocBox(bbox=bbox,
                      bbox_writable=self.margin_notes_template.generate_margin_box(bbox, font_size=font_size),
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
            id:

        Returns:

        """

        available_space = all_paragraphs_box.bbox_writable[3] - current_y
        font_size = self.paragraph_template.font_scale_factor() * self.font_size
        if available_space < self.paragraph_template.min_height_pixels or available_space < font_size:
            return None

        random_height = round(self.paragraph_template.height_as_percent_of_page() * all_paragraphs_box.height)
        height = max(min(random_height, available_space), self.paragraph_template.min_height_pixels, font_size)

        bbox = all_paragraphs_box.bbox_writable[0], current_y, all_paragraphs_box.bbox_writable[2], current_y+height
        return DocBox(bbox=bbox,
                      bbox_writable=self.paragraph_template.generate_margin_box(bbox, font_size=font_size),
                      category="paragraph",
                      parent=all_paragraphs_box,
                      font_size=font_size,
                      id=id,
                      max_lines=self.paragraph_template.max_lines()
                      )

    def page_title_box(self, pg_box: DocBox, current_y):
        """ Given a page DocBox, generate a page title box, which is a box that contains a title for the page.

        Args:
            pg_box:

        Returns:

        """
        font_size = scale_font(self.font_size, (1,1.3))
        height = self.page_title_template.generate_height_in_pixels(font_size=font_size)
        bbox = pg_box.bbox_writable[0], current_y, pg_box.bbox_writable[2], current_y + height
        bbox_writable = self.page_title_template.generate_margin_box(bbox, font_size=font_size)

        return DocBox(bbox=bbox,
               bbox_writable=bbox_writable,
               category="page_title",
               parent=pg_box,
               font_size=font_size,
               max_lines=self.page_title_template.max_lines()
               )

    def page_header(self, pg_box: DocBox, current_y):
        """ Given a page DocBox, return a PageHeader DocBox

        Args:
            pg_box (DocBox):  The page DocBox to add the header to

        Returns:

        """
        font_size = scale_font(self.font_size, (1, 1.3))
        height = self.page_header_template.generate_height_in_pixels(font_size=font_size, page_height=pg_box.height, method="max")
        bbox = pg_box.bbox_writable[0], current_y, pg_box.bbox_writable[2], current_y + height
        bbox_writable = self.page_title_template.generate_margin_box(bbox, font_size=font_size)

        return DocBox(bbox=bbox,
                      bbox_writable=bbox_writable,
                      category="page_title",
                      parent=pg_box,
                      font_size=font_size
                      )

    def _draw_doc_box(self, doc_box, image):
        """ Recursive call from draw_doc_boxes

        Args:
            doc_box: doc_box to draw
            image: PIL image

        Returns:

        """
        for child in doc_box.children:
            image = self._draw_doc_box(child, image)
        image = BBox._draw_box(doc_box.bbox, image, "black")
        if doc_box.category in ["paragraph","margin_note","page_title","paragraph_note"]:
            image = BBox._draw_center(doc_box.bbox, image, "red")
        if not doc_box.bbox_writable is None:
            image = BBox._draw_box(doc_box.bbox_writable, image, "red")
        return image

    def draw_doc_boxes(self, doc_box, image=None):
        """ Recursively draw bounding boxes on image for debugging purposes

            Args:
                doc_box: DocBox object
                image: PIL image to draw on
        """
        size = doc_box.size
        if image is None:
            image = Image.new("L", size, 255)
        image = self._draw_doc_box(doc_box, image)
        return image

    def _render_text(self, background_image, doc_box, text_generator, **kwargs):
        child_boxes = []

        # Look for nephews; nephews should not have text rendered, but should be incorporated into the layout
        if doc_box.nephews is not None:
            child_boxes = [doc_box.bbox_writable for doc_box in doc_box.nephews]

        for child in doc_box.children:
            self._render_text(background_image, child, text_generator, **kwargs)
            child_boxes.append(child.bbox_writable)

        if doc_box.category in self.writable_categories:
            image, localization = fill_area_with_words(text_generator,
                                                       doc_box.bbox_writable,
                                                       text_list=None,
                                                       max_lines=doc_box.max_lines,
                                                       error_handling="ignore",
                                                       indent_new_paragraph_prob=.2,
                                                       scale=1 if text_generator.font_size is None else doc_box.font_size / text_generator.font_size,
                                                       slope=random.gauss(0,0.001),
                                                       slope_drift=(0, 0.0001),
                                                       **kwargs)
            composite_images2(background_image, image, doc_box.bbox_writable[0:2])
            doc_box.bbox_list = localization

        else:
            # aggregate localization of children
            if child_boxes:
                doc_box.bbox_list = child_boxes
            else:
                doc_box.bbox_list = [doc_box.bbox_writable]

        # Shrink BBox to fit actual text / children
        if doc_box.category not in self.dont_shrinkwrap_categories:
            shrinkwrap_bbox = BBox.get_maximal_box(doc_box.bbox_list)
            doc_box.update_bbox(shrinkwrap_bbox, format="XYXY")
            doc_box.bbox_writable = shrinkwrap_bbox


    def render_text(self, doc_box, text_generator, **kwargs):
        """ Recursively draw text in DocBox's in document

        Args:
            doc_box (DocBox): DocBox object defining where to draw text
            text_generator (obj): A generator that returns (image, str) tuples
            **kwargs:

        Returns:

        """
        size = doc_box.size
        image = Image.new("L", size, 255)
        self._render_text(image, doc_box, text_generator, **kwargs)
        return image

    def _create_ocr(self, doc_box, ocr_format_master, level, ids):
        """ Recursive function called by create_ocr
            Need to add: section relationships (paragraph -> margins)
                         page level
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
            doc_box:
            id:
            filename:

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
"""
COCO:
{"categories": [{"id": 1, "name": "text_paragraph", "supercategory": "text_paragraph"}, {"id": 2, "name": "text_header", "supercategory": "text_header"}, {"id": 3, "name": "paragraph_notes", "supercategory": "paragraph_notes"}, {"id": 4, "name": "text_margin", "supercategory": "text_margin"}, {"id": 5, "name": "text_table", "supercategory": "text_table"}, {"id": 6, "name": "titles", "supercategory": "titles"}],
"images": [{"id": 0, "license": 1, "file_name": "62058_b961185-00149.jpg", "height": 3264, "width": 4576, "date_captured": null}
...

OCR

TODO:
output format
fill in with text - also redefine the boxes

features:
    header
    paragraph note
    same page style (margin width etc.)
    
"""