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
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from typing import List
from docgen.bbox import BBox
from docgen.utils import utils
from docgen.render_doc import *
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