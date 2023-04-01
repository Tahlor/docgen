import random
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform
from reportlab.lib.colors import magenta, pink, blue, green
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from pathlib import Path
import os
import sys
if sys.version_info >= (3, 8):
    from typing import Literal, Dict
else:
    from typing_extensions import Literal, Dict
from easydict import EasyDict as edict
from reportlab.pdfbase.pdfmetrics import stringWidth
from docgen.bbox import BBox
from textgen.field_generator.format_form_fields import FieldCollection, FieldPair, Field

folder = Path(os.path.dirname(__file__))

"""
p = Paragraph(text, style=preferred_style)
width, height = p.wrapOn(self.canvas, aW, aH)
p.drawOn(self.canvas, x_pos, y_pos)
"""

reportlab_standard_fonts = [
    "Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique",
    "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic",
    "Courier", "Courier-Bold", "Courier-Oblique", "Courier-BoldOblique",
    #"Symbol", "ZapfDingbats"
]

class FormGenerator():

    def __init__(self,
                 field_label_position: Literal['adjacent', 'inside'] = "adjacent",
                 fill_fields_with_text: bool = False,
                 prob_of_drawing_boxes: float = 0.5,
                 prob_of_drawing_underlines: float = 0.5,
                 use_random_font: bool = True,
                 margins: Dict[str, int] = {"L": 40, "T": 40, "R": 40, "B": 40},
                 horizontal_space: int = 10,
                 vertical_space: int = 12,
                 leading: float = 0.9,
                 line_idx: int = 0,
                 within_line_idx: int = 0,
                 document_height: int = 842,
                 document_width: int = 595,
                 prob_of_random_horizontal_space: float = 0.5,
                 prob_of_random_newline: float = 0.5,
                 randomize_field_order: bool = True,
                 ):
        """
        Initialize the class with various formatting settings.

        Args:
            field_label_position (Literal['adjacent', 'inside']): Position of the field label.
            fill_fields_with_text (bool): Whether to fill fields with text.
            prob_of_drawing_boxes (float): Probability of drawing boxes around fields.
            use_random_font (bool): Use a random font for each field.
            margins (Dict[str, int]): Margins for the document layout as a dict with keys 'L', 'T', 'R', 'B'.
            horizontal_space (int): Horizontal space between fields.
            vertical_space (int): Vertical space between fields.
            leading (float): Line leading (spacing factor).
            line_idx (int): Index for line tracking in multi-line fields.
            within_line_idx (int): Sub-index within lines for detailed field positioning.
            document_height (int): Height of the document.
            document_width (int): Width of the document.
        """
        self.field_label_position = field_label_position
        self.fill_fields_with_text = fill_fields_with_text
        self.prob_of_drawing_boxes = prob_of_drawing_boxes
        self.prob_of_drawing_underlines = prob_of_drawing_underlines
        self.use_random_font = use_random_font
        self.margins = edict(margins)
        self.horizontal_space = horizontal_space
        self.vertical_space = vertical_space
        self.leading = leading
        self.line_idx = line_idx
        self.within_line_idx = within_line_idx
        self.document_height = document_height
        self.document_width = document_width
        self.prob_of_random_horizontal_space = prob_of_random_horizontal_space
        self.prob_of_random_newline = prob_of_random_newline
        self.randomize_field_order = randomize_field_order
        self.margin_buffer_space = 50

    def create_new_form(self,
                        fields,
                        form_name='simple_form.pdf',
                        form_title="New Form"):
        self.ocr = {"sections": []}

        self.c = canvas.Canvas(form_name)
        # self.c._pagesize
        self.c.setPageSize((self.document_width, self.document_height))
        self.shape_excl_margins = (self.c._pagesize[0]-self.margins.L-self.margins.R,
                                self.c._pagesize[1] - self.margins.T - self.margins.B)

        self.setFont("Courier", 20)
        self.current_position = [self.margins.L, self.c._pagesize[1]-self.margins.T]

        self.add_title(form_title)
        self.setFont("Courier", 12)
        self.form = self.c.acroForm
        self.add_fields(fields)
        self.c.save()
        return self.ocr

    def setFont(self, font=None, size=(10,14)):
        if font is None:
            font = random.choice(reportlab_standard_fonts)
        if isinstance(size, (list,tuple)):
            size = random.randint(*size)
        self.c.setFont(font, size)
        self.line_height = self.fontHeight() + self.vertical_space

    def add_to_ocr(self, merged_text, merged_bbox, category, parts=None):
        section = {
            "text": merged_text,
            "bbox": merged_bbox,
            "paragraphs": []
        }
        if parts is not None:
            for bbox, text, sub_category in parts:
                section["paragraphs"].append({
                    "bbox": bbox,
                    "text": text,
                    "category": sub_category
                })

        self.ocr["sections"].append(section)

    def pos_to_bbox(self, width, height=None, x1=None, y1=None):
        x1 = self.current_position[0] if x1 is None else x1
        y1 = self.current_position[1] if y1 is None else y1
        height = self.line_height if height is None else height
        x2, y2 = x1 + width, y1 + height
        return BBox("ll", bbox=[x1,y1,x2,y2], height_ll=self.document_height)


    def add_title(self, title):
        string_width=self.stringWidth(title)
        center = int((self.shape_excl_margins[0]-string_width)/2)+self.margins.L
        # self.c.drawCenteredString()
        self.c.drawString(center, self.current_position[1], title)

        bbox = self.pos_to_bbox(width=string_width)
        self.add_to_ocr(title, bbox, "title")

        # x = self.margins.L + self.shape_excl_margins[0]/2
        # self.c.rect(x-self.stringWidth(title)/2, self.current_position[1], self.stringWidth(title), 20)

        self.current_position = [self.margins.L,
                                 self.current_position[1] - self.line_height]

    def add_table(self):
        raise NotImplemented
        t = Table(tableData, style=tStyle)
        t.canv = myCanvas
        w, h = t.wrap(0, 0)

    def add_checkbox(self):
        raise NotImplemented
        self.c.drawString(10, 650, 'Dog:')
        self.form.checkbox(name='cb1', tooltip='Field cb1',
                      x=110, y=645, buttonStyle='check',
                      borderColor=magenta, fillColor=pink,
                      textColor=blue, forceBorder=True)

    def stringWidth(self, text, fontName=None, fontSize=None):
        if fontName is None:
            fontName = self.c._fontname
        if fontSize is None:
            fontSize = self.c._fontsize
        return stringWidth(text, fontName, fontSize)

    def fontHeight(self, fontSize=None):
        if fontSize is None:
            fontSize = self.c._fontsize
        return self.leading * fontSize

    def validate(self, text):
        if isinstance(text, (float, int)):
            return str(text)
        return text

    def truncate_text_to_fit(self, text):
        space_available = self.remaining_horizontal_space()
        if self.stringWidth(text) > space_available:
            _text = text[:int(len(text) * space_available/self.stringWidth(text))]
            text = _text[:_text.rfind(' ')]
        return text

    def _add_field_from_list(self, field, font_size=12):
        self.setFont()
        if self.use_random_font:
            self.setFont()
        if len(field)==3:
            title, value, width = field
        else:
            title, value, width = *field, 0
        title = self.validate(title)
        value = self.validate(value)
        self.add_field(title, width, value)

    def add_fields(self, fields):
        if isinstance(fields, dict):
            fields = list(tuple(x) for x in fields.items())
        if isinstance(fields, (list, tuple)):
            fields = list(fields)
            if self.randomize_field_order:
                random.shuffle(fields)
        elif isinstance(fields, FieldCollection) and self.randomize_field_order:
            fields.randomize()

        for field in fields:
            if isinstance(field, (list, tuple)):
                self._add_field_from_list(field)
            elif isinstance(field, dict):
                self._add_field_from_list(field.items())
            elif isinstance(field, FieldPair):
                self._add_field_from_field_pair(field)
            if random.random() < self.prob_of_random_horizontal_space:
                self.current_position[0] += random.random() * self.remaining_horizontal_space() / 2
            if random.random() < self.prob_of_random_newline:
                self.current_position[1] -= self.line_height

    def _add_fields_from_field_collection(self, field_collection):
        for field_pair in field_collection:
            self.add_field(field_pair.prompt, field_pair.response)

    def _add_field_from_field_pair(self, field_pair):
        self.add_section(field_pair)

    def add_section(self, field_pair: FieldPair):
        full_string = field_pair.prompt.to_string() + field_pair.response.to_string()
        self.new_line_if_needed(text_width=self.stringWidth(full_string), additional_spaces=self.margin_buffer_space + self.horizontal_space)

        self.add_prompt(field_pair.prompt)
        self.add_response(field_pair.response)

    def add_prompt(self, prompt: Field):
        self.populate_paragraph(prompt, "prompt")

    def add_response(self, response: Field):
        self.populate_paragraph(response, "response")

    def populate_paragraph(self, field: Field, field_type: str):
        parts = []

        box_height = self.fontHeight() + self.vertical_space / 2
        position_with_vertical_offset = self.current_position[1] + (self.line_height - self.fontHeight()) / 2

        # THIS SHOULD BE HANDLED BY THE CHILD OBJ
        parsed_values2 = self.parse_nested_field(field.field_data)
        parsed_values = field.field_data
        for key, sub_value in parsed_values2:
            #sub_value = f" {sub_value} " if key != "value" else sub_value
            sub_value_width = self.stringWidth(sub_value)
            bbox = self.pos_to_bbox(sub_value_width,
                                    box_height,
                                    self.current_position[0],
                                    self.current_position[1] + (self.line_height-box_height)/2
            )

            # bbox = self.pos_to_bbox(sub_value_width, box_height,
            #                         self.current_position[0],
            #                         self.current_position[1] + (self.line_height - box_height) / 2)
            if self.fill_fields_with_text:
                self.c.drawString(self.current_position[0], position_with_vertical_offset, sub_value)

            #self.draw_bbox(bbox)

            parts.append((bbox, sub_value.strip(), key if key else field_type))
            self.current_position[0] += sub_value_width
        self.current_position[0] += self.horizontal_space

        if parts:
            #merged_text, merged_bbox = self.merge_text_and_bboxes(parts)
            merged_text = field.to_string()
            merged_bbox = BBox(origin="ll", bbox=BBox.get_maximal_box([bbox for bbox, _, _ in parts]),
                              height_ll=self.document_height)
            self.add_to_ocr(merged_text, merged_bbox, field_type, parts)
            self.draw_form_element(merged_bbox)

    def draw_form_element(self, bbox):
        if random.random() < self.prob_of_drawing_boxes:
            # add random padding
            new_bbox = bbox.copy().expand_random_amount(max_padding_horizontal=5, max_padding_vertical=5)
            self.draw_bbox(new_bbox)
        elif random.random() < self.prob_of_drawing_underlines:
            self.c.line(bbox.x1, bbox.y1, bbox.x2, bbox.y1)

    def draw_bbox(self, bbox):
        self.c.rect(x=bbox.x1, y=bbox.y1, width=bbox.width, height=bbox.height)

    def new_line(self):
        self.current_position[1] -= self.line_height
        self.current_position[0] = self.margins.L
        self.within_line_idx = 0

    def new_line_if_needed(self, text_width, additional_spaces=0):
        if text_width + additional_spaces > self.remaining_horizontal_space():
            self.new_line()

    @staticmethod
    def parse_nested_field(field):
        if isinstance(field, list):
            parsed_fields = []
            for f in field:
                if isinstance(f, dict):
                    key = list(f.keys())[0]
                    value = f[key]
                    parsed_fields.append((key, value))
                else:
                    parsed_fields.append(("", f))
            return parsed_fields
        elif isinstance(field, dict):
            key = list(field.keys())[0]
            value = field[key]
            return [(key, value)]
        return [("", field)]

    def add_field(self, title, user_specified_value_width, value=''):
        self.setFont()
        if hasattr(value, "to_string"):
            full_string = value.to_string()
        else:
            full_string = value

        parsed_values = self.parse_nested_field(value)  # Parse the nested structure
        title_width = self.stringWidth(title)
        total_value_width = sum(self.stringWidth(f" {v} ") for _, v in parsed_values)
        value_width = max(total_value_width, user_specified_value_width)
        self.move_to_next_line_if_needed(text_width=title_width, form_field_width=value_width)

        if self.field_label_position == "adjacent":
            position_with_vertical_offset = self.current_position[1] + (self.line_height - self.fontHeight()) / 2
            self.c.drawString(self.current_position[0], position_with_vertical_offset, title)
            bbox = self.pos_to_bbox(width=self.stringWidth(title))
            self.add_to_ocr(title, bbox, "field_name")
            self.current_position[0] += self.stringWidth(title) + self.horizontal_space

        elif self.field_label_position == "inside":
            raise NotImplementedError

        # Handle nested fields
        for key, sub_value in parsed_values:
            sub_value = f" {sub_value} "
            sub_value_width = self.stringWidth(sub_value)

            if self.within_line_idx == 0:
                sub_value = self.truncate_text_to_fit(sub_value) + " "
                sub_value_width = max(self.stringWidth(sub_value), user_specified_value_width)

            box_height = self.fontHeight() + self.vertical_space / 2

            bbox = self.pos_to_bbox(sub_value_width,
                                    box_height,
                                    self.current_position[0],
                                    self.current_position[1] + (self.line_height - box_height) / 2)
            if random.random() < self.prob_of_drawing_boxes:
                self.c.rect(x=bbox.x1, y=bbox.y1, width=bbox.width, height=bbox.height)

            position_with_vertical_offset = self.current_position[1] + (self.line_height - self.fontHeight()) / 2
            if self.fill_fields_with_text:
                self.c.drawString(self.current_position[0], position_with_vertical_offset, sub_value)

            self.add_to_ocr(sub_value.strip(), bbox, key if key else "field")
            self.within_line_idx += 1
            self.current_position[0] += sub_value_width + self.horizontal_space

        self.within_line_idx = 0
        self.current_position[1] -= self.line_height  # Move to the next line
        self.current_position[0] = self.margins.L  # Reset horizontal position

    def move_to_next_line_if_needed(self, text_width, form_field_width, new_line_height=None):
        if new_line_height is None:
            new_line_height = self.fontHeight() + self.vertical_space

        # New line
        if text_width + form_field_width > self.remaining_horizontal_space():
            self.current_position[1] -= new_line_height
            self.current_position[0] = self.margins.L
            self.within_line_idx = 0
            self.line_idx += 1

    def remaining_horizontal_space(self):
        return self.c._pagesize[0] - self.current_position[0] - self.margins.R

def filter_to_one_row(fields):
    for i, cell in enumerate(fields):
        if isinstance(cell, str) and "\n" in cell:
            fields[i] = cell.split("\n")[0]
        if isinstance(cell, (list,tuple)):
            fields[i] = random.choice(cell)
        if isinstance(cell, dict):
            fields[i] = random.choice(list(cell.items()))
    return fields

if __name__ == '__main__':
    from textgen.table_from_faker import TableDataFromFaker
    functions = ["address",
                 "relationship",
                 "job",
                 "date",
                 "height",
                 "phone_number",
                 "ssn",
                 "aba",
                 "bank_country",
                 "license_plate",
                 "bban",
                 "iban",
                 "swift8",
                 "ean13",
                 "company",
                 "currency",
                 "date_time_this_century",
                 "password",
                 "paragraph"]

    fg = FormGenerator()
    fields = [["Name:","Taylor", 50],
              ["Phone:", "610-573-7638", 50],
              ["Address:", "1973 Montana Ave", 100]
              ]

    generator = TableDataFromFaker(functions=functions,
                                   include_row_number=False,
                                   random_fields=27)
    one_row = filter_to_one_row(list(generator.gen_content(1))[0])
    fg.create_new_form(zip(generator.header_names+generator.extra_headers, one_row))

    # print the location
    print(f"Form saved to {folder/'simple_form.pdf'}")