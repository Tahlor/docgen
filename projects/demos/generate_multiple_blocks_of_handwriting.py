import torch
import json
import random
from docgen.pdf_edit import *
from PIL import Image
from textgen.unigram_dataset import Unigrams
from docgen.rendertext.render_word import RenderWordFont
from handwriting.data.saved_handwriting_dataset import SavedHandwriting
import numpy as np
from docgen import utils
from docgen.utils import display
from docgen.image_composition.utils import new_textbox_given_background
from textgen.wikipedia_dataset import Wikipedia
from handwriting.data.hw_generator import HWGenerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from docgen.utils import file_incrementer
from docgen.dataset_utils import load_json, draw_boxes_sections
from docgen.pdf_edit import convert_to_ocr_format

""" TODO
* truncate words
* bad lining up
* last line cutoff

"""

PATH= r"C:\Users\tarchibald\github\handwriting\handwriting\data\datasets\synth_hw\style_298_samples_0.npy"
PDF_FILE = r"C:\Users\tarchibald\github\docx_localization\temp\TEMPLATE.pdf"
UNIGRAMS = r"C:\Users\tarchibald\github\textgen\textgen\datasets\unigram_freq.csv"
TESTING=False
BATCH_SIZE = 12
RESUME = False
FREQ = 5000

def main():
    global IDX
    from handwriting.data.basic_text_dataset import VOCABULARY
    print(f"Vocab: {VOCABULARY}")
    basic_text_dataset = Wikipedia(
        dataset=load_dataset("wikipedia", "20220301.en")["train"],
        vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
        exclude_chars="0123456789()+*;#:!/",
        min_sentence_length=60,
        max_sentence_length=64
    )
    renderer = HWGenerator(next_text_dataset=basic_text_dataset,
                           batch_size=BATCH_SIZE,
                           model="CVL")

    dataloader = DataLoader(basic_text_dataset,
                            batch_size=BATCH_SIZE,
                            collate_fn=basic_text_dataset.collate_fn)
    remainder = 1000
    for i, d in enumerate(dataloader):
        process_batch(d, renderer)
        ii = i * BATCH_SIZE
        if remainder > ii % FREQ:
            with OUTPUT_OCR_JSON.open("w") as ff:
                json.dump(OUTPUT_DICT, ff)
        remainder = ii % FREQ

def try_try_again(func):
    def try_again(*args,**kwargs):
        i = 0
        while True:
            try:
                i+=1
                result = func(*args, **kwargs)
                return result

            except KeyboardInterrupt as e:
                with OUTPUT_OCR_JSON.open("w") as ff:
                    json.dump(OUTPUT_DICT, ff)
            except Exception as e:
                print(f"ERROR {i} {e}")
                torch.cuda.empty_cache()
                if i >= 10:
                    return
    return try_again

@try_try_again
def process_batch(d, renderer):
    global IDX
    global OUTPUT_DICT

    def new_paragraph(shp, offset=None, random_sample=False):
        if not random_sample:
            nonlocal sample
        else:
            sample = random.choice(new_sample_batch)

        size, origin = new_textbox_given_background(shp)
        if not offset is None:
            origin = origin[0]+offset[0], origin[1]+offset[1]

        scale = random.uniform(.7,1.8)
        box1, localization = fill_area_with_words(word_imgs=sample["words"],
                                        bbox=[0,0,*size],
                                        text_list=sample["raw_text"].split(" "),
                                        max_intraline_vertical_space_offset=5,
                                        error_handling="expand",
                                        scale=scale
                                        )
        background_img.paste(Image.fromarray(box1), origin)
        ocr_format = convert_to_ocr_format(localization, origin_offset=origin, section=section)
        return size,origin,box1, localization, ocr_format

    """
    font_resize_factor = random.uniform(.8,2.5)
    default_font_size = 32
    word_imgs = [renderer.render_word(t, size=int(default_font_size*font_resize_factor)) for t in text_list]

    """

    new_sample_batch = list(renderer.process_batch(d))
    for sample in new_sample_batch:
        section = 0
        canvas_size = 768,1152
        ocr_out = {"sections": [],"width":canvas_size[0], "height":canvas_size[1]}

        background_img = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas_size = utils.shape(background_img)

        text_box_shp, origin, box1, bboxs1, ocr_format = new_paragraph(canvas_size)
        ocr_out["sections"].append(ocr_format)
        section +=1

        # Make another paragraph if there's space
        remaining_vertical_space = canvas_size[1] - origin[1] - text_box_shp[1]
        remaining_horizontal_space = canvas_size[0] - origin[0] - text_box_shp[0]

        if remaining_vertical_space > .25 * canvas_size[1]:
            offset = 0, origin[1] + text_box_shp[1]
            canvas_size = [canvas_size[0],remaining_vertical_space]
            text_box_shp, origin, box1, bboxs1, ocr_format = new_paragraph(canvas_size, offset, random_sample=True)
            ocr_out["sections"].append(ocr_format)

        elif remaining_horizontal_space > .25 * canvas_size[0]:
            canvas_size = [remaining_horizontal_space, canvas_size[1]]
            offset = origin[0] + text_box_shp[0], 0
            text_box_shp, origin, box1, bboxs1, ocr_format = new_paragraph(canvas_size, offset, random_sample=True)
            ocr_out["sections"].append(ocr_format)

        # for bbox in bboxs1:
        #     bbox.draw_box(background_img)

        # display(background_img)

        file_name = f"{IDX:07.0f}"
        #draw_boxes_sections(ocr_out, background_img)

        utils.save_image(background_img,OUTPUT_PATH / (file_name + ".jpg"))
        # print(shape(background_img))
        OUTPUT_DICT[file_name] = ocr_out
        IDX += 1


if __name__ == "__main__":
    root = Path("./temp/IAM")
    root = Path("/home/taylor/anaconda3/DATASET/")
    RESUME = False
    if RESUME:
        OUTPUT_PATH = root = Path("/home/taylor/anaconda3/DATASET/")
        OUTPUT_OCR_JSON = OUTPUT_PATH / "OCR.json"
        OUTPUT_DICT = load_json(OUTPUT_OCR_JSON)
        IDX = max(int(x) for x in OUTPUT_DICT)+1
        print(f"STARTING AT {IDX}")

    else:
        OUTPUT_PATH = file_incrementer(root, create_dir=True)
        OUTPUT_OCR_JSON = OUTPUT_PATH / "OCR.json"
        OUTPUT_DICT = {}
        IDX = 0

    for i in range(0,1):
        background_img, ocr_format = main()


# truncate words
# bad lining up
# last line cutoff