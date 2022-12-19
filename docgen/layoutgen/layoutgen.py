from collections import defaultdict
import random
from collections import defaultdict
from math import ceil
from typing import Literal
from docgen.dataset_utils import ocr_dataset_to_coco
from PIL import Image

from docgen.bbox import BBox
from docgen.pdf_edit import fill_area_with_words, composite_images2
from docgen.utils import display
from docgen.pdf_edit import convert_to_ocr_format

def flip(prob=.5):
    return random.random() < prob

class DocBox(BBox):
    """
    Document
        Page -> Header
                Margin
                Paragraph -> Paragraph Note

    """
    def __init__(self,
                 bbox,
                 bbox_writable=None,
                 category="",
                 children=None,
                 parent=None,
                 origin: Literal['ul', 'll'] = "ul",
                 line_number=None,
                 line_word_index=None,
                 text=None,
                 parent_obj_bbox=None,
                 paragraph_index=None,
                 font_size=None,
                 id=None):
        super().__init__(origin,bbox)
        self.category = category
        self.children = children if not children is None else []
        self.parent=parent
        self.font_size = font_size
        self.id = id
        if not parent is None:
            parent.children.append(self)

        if bbox_writable is None:
            self.bbox_writable = bbox
        else:
            self.bbox_writable = bbox_writable

class MarginGenerator:
    def __init__(self,
                 top_margin=(0.,0.1),
                 bottom_margin=(0.,0.1),
                 left_margin=(0.,0.1),
                 right_margin=(0.,0.1),
                 naive=False
                 ):
        if naive:
            self.generate_margin_box = self.naive
        else:
            self.top_margin = top_margin
            self.bottom_margin = bottom_margin
            self.left_margin = left_margin
            self.right_margin = right_margin
    
    def naive(self, bbox, *args, **kwargs):
        return bbox 

    def gen_margin(self, width, height):
        top_margin = random.uniform(*self.top_margin) * height
        bottom_margin = random.uniform(*self.bottom_margin) * height
        left_margin = random.uniform(*self.left_margin) * width
        right_margin = random.uniform(*self.right_margin) * width
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

    def  __init__(self,
                  pages_per_image=(1,2),
                  pages_per_image_prob_weights=None,
                  width=3200,
                  height=2400,
                  random_size_factor=2,
                  max_page_width_factor=1.5,
                  page_margins=None,
                  paragraph_margins=None,
                  margin_margins=None,
                  page_title_margins=None,
                  paragraph_note_margins=None,
                  page_header_margins=None,
                  paragraph_height_min_pixels=40,
                  paragraph_height=(.05,.7),
                  margin_notes_probability=.5,
                  margin_blank_probability=.2,
                  margin_notes_width=(.1,.3),
                  stop_page_early_probability=.1,
                  max_lines_margin_notes=5,
                  font_size_pixels=(20,64),
                  font_variability=(.9,1),
                  page_title_height=(.02,.1),
                  page_title_prob=.5,
                  page_header_height=(.02, .3),
                  page_header_prob=.1,
                  paragraph_note_probability=.5,
                  paragraph_note_blank_probability=.2,
                  paragraph_note_lines=(1,3)
                  ):
        """

        Args:
            pages_per_image:
            pages_per_image_prob_weights:
            width:
            height:
            random_size_factor:
            max_page_width_factor: page cannot be wider than FACTOR*WIDTH, or smaller than WIDTH/FACTOR
            page_margins:
            paragraph_margins:
            paragraph_height_min_pixels:
            paragraph_height:
            margin_notes_probability:
            margin_notes_width:
            paragraph_note_probability:
            stop_page_early_probability:
            max_lines_margin_notes:
        """
        self.width = width
        self.height = height
        self.random_size_factor = random_size_factor
        self.max_page_width_factor = max_page_width_factor

        self.page_margins = page_margins if not page_margins is None else MarginGenerator(naive=True)
        self.paragraph_margins = paragraph_margins if not paragraph_margins is None else MarginGenerator(naive=True)
        self.margin_margins = margin_margins if not margin_margins is None else MarginGenerator(naive=True)
        self.page_title_margins = page_title_margins if not page_title_margins is None else MarginGenerator(naive=True)
        self.paragraph_note_margins = paragraph_note_margins if not paragraph_note_margins is None else MarginGenerator(naive=True)
        self.page_header_margins = page_header_margins if not page_header_margins is None else MarginGenerator(naive=True)

        self.page_header_height = page_header_height
        self.page_header_prob = page_header_prob

        self.paragraph_height_min_pixels = paragraph_height_min_pixels
        self.paragraph_height = paragraph_height
        self.paragraph_note_probability = paragraph_note_probability
        self.paragraph_note_blank_probability = paragraph_note_blank_probability

        self.margin_notes_probability = margin_notes_probability
        self.margin_notes_width = margin_notes_width
        self.max_lines_margin_notes = max_lines_margin_notes
        self.margin_blank_probability = margin_blank_probability

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
        self.font_variability = font_variability

        self.paragraph_note_lines = paragraph_note_lines
        self.page_title_height = page_title_height
        self.page_title_prob = page_title_prob

    def generate_layout(self):
        width = random.randint(self.width_lower, self.width_upper)
        height = random.randint(self.height_lower, self.height_lower)
        min_pages = ceil(width / (self.width * self.max_page_width_factor))
        max_pages = int(width / (self.width / self.max_page_width_factor))
        min_page = max(min_pages, 1, self.pages_per_image[0])
        max_pages = min(max(min_page, max_pages), self.pages_per_image[1])
        pages = random.randint(min_page, max_pages)
        self.font_size = random.randint(*self.font_size_pixels)
        current_x = 0
        page_width = round(width / pages)
        layout = DocBox(bbox=(0,0,width,height), category="document")
        for page in range(0,pages):
            self.generate_page(starting_x=current_x, page_width=page_width, page_height=height, parent=layout)
            current_x += page_width

        return layout

    def generate_page(self, starting_x, page_width, page_height, parent=None):
        page_bbox=(starting_x, 0, starting_x+page_width, page_height)
        page_bbox_with_margins = page_bbox if self.page_margins is None else self.page_margins.generate_margin_box(page_bbox)
        #full_page = DocBox(bbox=bbox, bbox_writable=bbox_with_margins, parent=parent)
        page = DocBox(bbox=page_bbox, bbox_writable=page_bbox_with_margins, parent=parent, font_size=self.font_size)

        current_y = page.bbox_writable[1]

        # Page Header
        if flip(self.page_header_prob):
            title_box = self.page_title_box(page)
            current_y = title_box.bbox[3]

        # Page title
        if flip(self.page_title_prob):
            title_box = self.page_title_box(page)
            current_y = title_box.bbox[3]

        paragraph_note = flip(self.paragraph_note_probability)

        # build margin
        if flip(self.margin_notes_probability):
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

        stop_early_prob = 0.01 # almost never leave the page without ANY paragraphs
        paragraph_id = 0
        while not flip(stop_early_prob):
            new_paragraph = self.paragraph_box(current_y, all_paragraphs_box, id=paragraph_id)
            if not new_paragraph is None:
                current_y += new_paragraph.height

                if not all_margins_box is None and not flip(self.margin_blank_probability):
                    self.margin_note_box(new_paragraph, all_margins_box, id=paragraph_id)

                if paragraph_note and not flip(self.paragraph_note_blank_probability):
                    # Usually make it 1 row
                    # Usually recenter it a bit
                    new_paragraph_note_box = self.paragraph_note_box(new_paragraph, id=paragraph_id)
                    current_y += new_paragraph_note_box.height
                paragraph_id += 1
            else:
                break
            stop_early_prob = self.stop_page_early_probability
        return page

    def paragraph_note_box(self, paragraph, id=None):
        font_size = scale_font(self.font_size, (.9,1.1))
        height = random.uniform(*self.paragraph_note_lines) * font_size
        bbox = paragraph.bbox[0],paragraph.bbox[3],paragraph.bbox[2],paragraph.bbox[3]+height
        paragraph_note = DocBox(bbox,
                                bbox_writable=self.paragraph_note_margins.generate_margin_box(bbox),
                                parent=paragraph,
                                font_size=font_size,
                                category="paragraph_note",
                                id=id
                                )
        return paragraph_note

    def margin_note_box(self, paragraph, all_margin_note, id=None):
        # TODO: random height offset
        bbox = all_margin_note.bbox[0], paragraph.bbox[1], all_margin_note.bbox[2], paragraph.bbox[3]
        font_size = scale_font(self.font_size, (.9,1.1))
        return DocBox(bbox=bbox,
                 bbox_writable=self.margin_margins.generate_margin_box(bbox, font_size=font_size),
                 category="margin_note",
                 parent=paragraph,
                 font_size=font_size,
                 id=id
                 )

    def paragraph_box(self, current_y, all_paragraphs_box, id=None):
        font_size = self.font_size
        available_space = all_paragraphs_box.bbox_writable[3] - current_y

        if available_space < self.paragraph_height_min_pixels or available_space < font_size:
            return None

        random_height = round(random.uniform(*self.paragraph_height) * all_paragraphs_box.height)
        height = max(min(random_height, available_space), self.paragraph_height_min_pixels, font_size)

        bbox = all_paragraphs_box.bbox_writable[0], current_y, all_paragraphs_box.bbox_writable[2], current_y+height
        return DocBox(bbox=bbox,
                 bbox_writable=self.paragraph_margins.generate_margin_box(bbox, font_size=font_size),
                 category="paragraph",
                 parent=all_paragraphs_box,
                 font_size=font_size,
                 id=id
                 )

    def page_title_box(self, pg_box):
        font_size = scale_font(self.font_size, (1,1.3))
        height = max(pg_box.height * random.uniform(*self.page_title_height), font_size)
        bbox = pg_box.bbox_writable[0], pg_box.bbox_writable[1], pg_box.bbox_writable[2], pg_box.bbox_writable[1] + height
        bbox_writable = self.page_title_margins.generate_margin_box(bbox, font_size=font_size)

        return DocBox(bbox=bbox,
               bbox_writable=bbox_writable,
               category="page_title",
               parent=pg_box,
               font_size=font_size
               )

    def page_header(self, pg_box):
        font_size = scale_font(self.font_size, (1, 1.3))
        height = max(pg_box.height * random.uniform(*self.page_title_height), font_size)
        bbox = pg_box.bbox_writable[0], pg_box.bbox_writable[1], pg_box.bbox_writable[2], pg_box.bbox_writable[
            1] + height
        bbox_writable = self.page_title_margins.generate_margin_box(bbox, font_size=font_size)

        return DocBox(bbox=bbox,
                      bbox_writable=bbox_writable,
                      category="page_title",
                      parent=pg_box,
                      font_size=font_size
                      )

    def _draw_doc_box(self, image, doc_box):
        for child in doc_box.children:
            image = self._draw_doc_box(image, child)
        image = BBox._draw_box(doc_box.bbox, image, "black")
        if doc_box.category in ["paragraph","margin_note","page_title","paragraph_note"]:
            image = BBox._draw_center(doc_box.bbox, image, "red")
        if not doc_box.bbox_writable is None:
            image = BBox._draw_box(doc_box.bbox_writable, image, "red")
        return image

    def draw_doc_boxes(self, doc_box, image=None):
        size = doc_box.size
        if image is None:
            image = Image.new("L", size, 255)
        image = self._draw_doc_box(image, doc_box)
        return image

    def _render_text(self, background_image, doc_box, text_generator, **kwargs):
        for child in doc_box.children:
            self._render_text(background_image, child, text_generator, **kwargs)

        if doc_box.category in ["paragraph", "margin_note", "page_title", "paragraph_note"]:
            max_lines = random.randint(0,self.max_lines_margin_notes) if self.max_lines_margin_notes and doc_box.category=="margin_note" else None
            image, localization = fill_area_with_words(text_generator,
                                                       doc_box.bbox_writable,
                                                       text_list=None,
                                                       max_lines=max_lines,
                                                       error_handling="force",
                                                       indent_new_paragraph_prob=.2,
                                                       scale=1 if text_generator.font_size is None else doc_box.font_size / text_generator.font_size,
                                                       slope=random.gauss(0,0.01),
                                                       slope_drift=(0, 0.001),
                                                       **kwargs)
            composite_images2(background_image, image, doc_box.bbox_writable[0:2])
            doc_box.localization = localization

    def render_text(self, doc_box, text_generator, **kwargs):
        size = doc_box.size
        image = Image.new("L", size, 255)
        self._render_text(image, doc_box, text_generator, **kwargs)
        return image

    def _create_ocr(self, doc_box, ocr_format_master, level, ids):
        """ Need to add: section relationships (paragraph -> margins)
                         page level
        """
        doc_box.id = level, ids[level]
        ids[level] += 1

        for child in doc_box.children:
            self._create_ocr(child, ocr_format_master, level+1, ids)

        if doc_box.category in ["paragraph","margin_note","page_title","paragraph_note"]:
            ocr_dict_paragraph = convert_to_ocr_format(doc_box.localization)
            meta_fields = {'level':level,
                           "category": doc_box.category,
                           "id": doc_box.id,
                           "parent_id": doc_box.parent.id,
                           }
            ocr_dict_paragraph.update(meta_fields)
            ocr_format_master["sections"].append(ocr_dict_paragraph)


    def create_ocr(self, doc_box, id, filename=""):
        ocr_out = {"sections": [], "width": doc_box.width, "height": doc_box.height, "id":id, "filename":filename}
        ids = defaultdict(int)
        self._create_ocr(doc_box, ocr_out, level=0, ids=ids)
        return ocr_out

if __name__ == "__main__":
    page_margins = MarginGenerator()

    page_title_margins = MarginGenerator(top_margin=(-.02,.02),
                 bottom_margin=(-.02,.02),
                 left_margin=(-.02,.5),
                 right_margin=(-.02,.5))
    paragraph_margins = MarginGenerator(top_margin=(-.1,.1),
                 bottom_margin=(-.1,.1),
                 left_margin=(-.1,.1),
                 right_margin=(-.1,.1))
    margin_margins = MarginGenerator(top_margin=(-.1,.5),
                 bottom_margin=(-.1,.1),
                 left_margin=(-.1,.1),
                 right_margin=(-.1,.1))
    paragraph_note_margins = MarginGenerator(top_margin=(-.05, .2),
                                     bottom_margin=(-.05, .2),
                                     left_margin=(-.05, .2),
                                     right_margin=(-.05, .2))
    if True:
        lg = LayoutGenerator(paragraph_margins=paragraph_margins,
                             page_margins=page_margins,
                             margin_margins=margin_margins,
                             page_title_margins=page_title_margins,
                             paragraph_note_margins=paragraph_note_margins,
                             margin_notes_probability=1,
                             pages_per_image=(1,3)
                             )
    else:
        lg = LayoutGenerator(margin_notes_probability=1)

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