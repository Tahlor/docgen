import random
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform
from reportlab.lib.colors import magenta, pink, blue, green
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from pathlib import Path
import os
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from easydict import EasyDict as edict
from reportlab.pdfbase.pdfmetrics import stringWidth
from docgen.bbox import BBox
folder = Path(os.path.dirname(__file__))

"""
p = Paragraph(text, style=preferred_style)
width, height = p.wrapOn(self.canvas, aW, aH)
p.drawOn(self.canvas, x_pos, y_pos)
"""

class FormGenerator():

    def __init__(self,
                 field_label_position: Literal['adjacent','inside']="adjacent",
                 fill_fields_with_text: bool=False):
        """

        Args:
            field_label: field label inside field OR to the left of the field
            fill_fields_with_text:
        """
        self.field_label_position = field_label_position
        self.margins = edict({"L":40,"T":40,"R":40,"B":40}) # L T R B
        self.horizontal_space = 10
        self.vertical_space = 12  # intended as whitespace between lines
        self.leading = 0.9
        self.line_idx = 0
        self.within_line_idx = 0
        self.fill_fields_with_text = fill_fields_with_text

    def create_new_form(self, fields, form_name='simple_form.pdf'):
        self.ocr = {"sections": []}

        self.c = canvas.Canvas(form_name)
        # self.c._pagesize
        self.c.setPageSize((595, 842))
        self.shape_excl_margins = (self.c._pagesize[0]-self.margins.L-self.margins.R,
                                self.c._pagesize[1] - self.margins.T - self.margins.B)

        self.setFont("Courier", 20)
        self.current_position = [self.margins.L, self.c._pagesize[1]-self.margins.T]

        self.add_title("New Form")
        self.setFont("Courier", 12)
        self.form = self.c.acroForm
        self.add_fields(fields)
        self.c.save()
        return self.ocr

    def setFont(self, font, size):
        self.c.setFont(font, size)
        self.line_height = self.fontHeight() + self.vertical_space

    def add_to_ocr(self,
                   text,
                   bbox,
                   category=""):
        self.ocr["sections"].append({"paragraphs": [{"bbox":bbox, "text":text, "category":category}]})

    def pos_to_bbox(self, width, height=None, x1=None, y1=None):
        x1 = self.current_position[0] if x1 is None else x1
        y1 = self.current_position[1] if y1 is None else y1
        height = self.line_height if height is None else height
        x2, y2 = x1 + width, y1 + height
        return BBox("ul", bbox=[x1,y1,x2,y2])


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

    def add_fields(self, field_list, font_size=12):
        for field in field_list:
            if len(field)==3:
                title, value, width = field
            else:
                title, value, width = *field, 0
            title = self.validate(title)
            value = self.validate(value)
            self.add_field(title, width, value)

    def add_field(self, title, user_specified_value_width, value=''):
        value = f" {value} "
        value_width = max(self.stringWidth(value), user_specified_value_width)
        title_width = self.stringWidth(title)
        self.move_to_next_line_if_needed(text_width=title_width, form_field_width=value_width)
        if self.field_label_position== "adjacent":
            position_with_vertical_offset = self.current_position[1] + (self.line_height-self.fontHeight())/2
            self.c.drawString(self.current_position[0], position_with_vertical_offset, title)
            bbox = self.pos_to_bbox(width=self.stringWidth(title))
            self.add_to_ocr(title, bbox, "field_name")

            self.current_position[0] += self.stringWidth(title) + self.horizontal_space

        elif self.field_label_position== "inside":
            raise NotImplementedError
            self.c.drawString(*self.current_position, title)

        if self.within_line_idx==0:
            value = self.truncate_text_to_fit(value) + " "
            value_width = max(self.stringWidth(value), user_specified_value_width)

        print(self.current_position)
        box_height = self.fontHeight() + self.vertical_space / 2

        if False:
            self.form.textfield(name=title, tooltip=title,
                                x=self.current_position[0],
                                y=self.current_position[1] + (self.line_height-box_height)/2,
                                width=value_width,
                                height=box_height,
                                borderStyle='inset',
                                forceBorder=True,
                                value=value)
        else:
            bbox = self.pos_to_bbox(value_width,
                                    box_height,
                                    self.current_position[0],
                                    self.current_position[1] + (self.line_height-box_height)/2
            )
            self.c.rect(x=bbox.x1,y=bbox.y1,width=bbox.width,height=bbox.height)
            position_with_vertical_offset = self.current_position[1] + (self.line_height - self.fontHeight()) / 2

            if self.fill_fields_with_text:
                self.c.drawString(self.current_position[0], position_with_vertical_offset,value)
            self.add_to_ocr(value, bbox, "field")

        self.within_line_idx += 1
        self.current_position[0] += value_width + self.horizontal_space

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
    from docgen.content.table_from_faker import TableDataFromFaker
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
