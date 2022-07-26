from pdfminer.layout import LAParams, LTTextBox, LTChar, LTRect
from pdfminer.converter import PDFPageAggregator
from collections import defaultdict
from collections.abc import Iterable
from pdfminer.converter import PDFLayoutAnalyzer
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
import numpy as np
import re

space = re.compile ( r"\s" )

#import pdftotext

""" Useful metadata:
    # Word bounding boxes
    # Word-field association
    # Font / Handwriting style

### NEED TO ADD:
    # Resizing
    # line parsing/wrapping
    
## Notes
    # PDF localization assumes a LOWER LEFT ORIGIN, whereas most image manipulation assumes upper left
        => y-coodinates need to be inverted
        => bboxes are similarly defined with lower left origin, make sure you paste to the upper left of the bbox
"""
def combine_boxes(list_of_coords):
    xys = np.array(list_of_coords).flatten()
    x_min = np.min(xys[::2])
    y_min = np.min(xys[1::2])
    x_max = np.max(xys[::2])
    y_max = np.max(xys[1::2])
    return [x_min, y_min, x_max, y_max]

def normalize_bbox(bbox, size_x, size_y):
    return (bbox[0] / size_x,
                       bbox[1] / size_y,
                       bbox[2] / size_x,
                       bbox[3] / size_y)

def clnstr(txt, remove_chars):
    return txt.translate(None, ''.join(remove_chars))

def generate_localization_from_bytes(pdf_file,
                                     level="word",
                                     first_page_only=False,
                                     ignore_chars=[],
                                     mark_for_replacement=False
                                     ):
    """

    Args:
        pdf_file:
        level: str - character, word, None

    Returns:

    """
    def character():
        while len(stack) > 0:
            obj = stack.pop(0)

            if isinstance(obj, LTChar):
                localization[pg_num]["localization_character"].append(
                    {"text": obj.get_text(), "bbox": obj.bbox, "font": obj.fontname})
            else:
                if isinstance(obj, Iterable):
                    stack.extend(list(iter(obj)))

    def reset(word, boxes, last_font=None):
        """ Create new item in localization list

        Args:
            word:
            boxes:
            obj:

        Returns:
            localization[pg_num]
                ["localization_textboxes"]
                    "norm_bbox"
                    "bbox"
                    "text"
                ["localization_word"]

        """
        if word.strip() and boxes:
            if space.match(word[-1]): # exclude ending space
                boxes = boxes[:-1]
            bbox = combine_boxes(boxes[:])
            normalized_bbox = normalize_bbox(bbox,
                                             localization[pg_num]["size"][0],
                                             localization[pg_num]["size"][1])

            # Manage Text
            replace = True if mark_for_replacement in word or not mark_for_replacement else False
            word = clnstr(word).replace("\u00A0", " ").replace("\u2011", "-")

            item = {"text": word.strip(),
                    "bbox": bbox,
                    "norm_bbox": normalized_bbox,
                    "font": last_font,
                    "replace":replace}

            localization[pg_num]["localization_word"].append(item)
        word = ''; boxes = [];
        return word, boxes

    def word():
        word = ""
        boxes = []
        last_font = None

        while len(stack) > 0:
            obj = stack.pop(0)

            if isinstance(obj, LTChar):
                text = obj.get_text()
                last_font = obj.fontname
                if text:
                    boxes.append(obj.bbox)
                    word += text
                    if space.match(word):
                        word, boxes = reset(word, boxes, last_font)
            else:
                word, boxes = reset(word, boxes, last_font)
                if isinstance(obj, Iterable):
                    stack.extend(list(iter(obj)))
        word, boxes = reset(word, boxes, last_font)
    process_level = {"character":character, "word":word}[level]

    localization = {}
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    pages = PDFPage.get_pages(pdf_file)
    for pg_num, page in enumerate(pages):
        print('Processing next page...')
        interpreter.process_page(page)
        layout = device.get_result()
        localization[pg_num] = defaultdict(list)
        #localization[pg_num]["textboxes"] = []
        localization[pg_num]["level"] = level
        localization[pg_num]["mediabox"] = page.mediabox
        localization[pg_num]["size"] = (page.mediabox[2]-page.mediabox[0],
                                        page.mediabox[3]-page.mediabox[1])

        for lobj in layout:
            if isinstance(lobj, LTTextBox):
                normalized_bbox = normalize_bbox(lobj.bbox, localization[pg_num]["size"][0], localization[pg_num]["size"][1])
                localization[pg_num]["localization_textboxes"].append({"text":lobj.get_text(),
                                                          "bbox":lobj.bbox,
                                                          "norm_bbox":normalized_bbox})
                #print(lobj.get_text())
                stack = [lobj._objs]
                if not level is None:
                    process_level()
        if first_page_only:
            break

    return localization

def generate_localization_from_file(pdf_file_path, **kwargs):
    with open(pdf_file_path, 'rb') as fp:
        return generate_localization_from_bytes(fp, **kwargs)

class CustomConverter(PDFLayoutAnalyzer):
    def receive_layout(self, ltpage):
        localization = []
        stack = [ltpage]
        while len(stack) > 0:
            lobj = stack.pop()

            if isinstance(lobj, LTChar):
                localization.append({"text":lobj.get_text(),
                                     "bbox":lobj.bbox,
                                     "font":lobj.fontname})

            if isinstance(lobj, Iterable):
                stack.extend(list(iter(lobj)))

        return localization[::-1]

def character_level_bounding_boxes(pdf_file):
    localization = defaultdict(list)
    rsrcmgr = PDFResourceManager()
    device = CustomConverter(rsrcmgr)

    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(pdf_file, 'rb') as fin:
        for page_num, page in enumerate(PDFPage.get_pages(fin)):
            localization[page_num] = interpreter.process_page(page)

    device.close()
    return localization
