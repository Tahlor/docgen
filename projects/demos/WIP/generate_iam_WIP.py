import random

from docgen.pdf_edit import *
from PIL import Image
from textgen.unigram_dataset import Unigrams
from docgen.rendertext.render_word import RenderWordFont
from hwgen.data.saved_handwriting_dataset import SavedHandwriting
import numpy as np
from docgen import utils
from docgen.utils import display
from docgen.image_composition.utils import new_textbox_given_background
from textgen.wikipedia_dataset import Wikipedia
from datasets import load_dataset

PATH= r"C:\Users\tarchibald\github\handwriting\handwriting\data\datasets\synth_hw\style_298_samples_0.npy"
PDF_FILE = r"C:\Users\tarchibald\github\docx_localization\temp\TEMPLATE.pdf"
UNIGRAMS = r"C:\Users\tarchibald\github\textgen\textgen\datasets\unigram_freq.csv"
TESTING=False
def main():
    if TESTING:
        words = Unigrams(csv_file=UNIGRAMS)
        renderer = SavedHandwriting(format="numpy",
                                    dataset_path=PATH,
                                    random_ok=True,
                                    conversion=lambda image: np.uint8(image * 255)
                                    )
    else:
        from hwgen.data.basic_text_dataset import VOCABULARY
        basic_text_dataset = Wikipedia(
            dataset=load_dataset("wikipedia", "20220301.en")["train"],
            vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
            encode_function=model.netconverter.encode,
            min_sentence_length=60,
            max_sentence_length=64
        )

    #renderer = RenderWordFont(format="numpy")

    text_list = words.sample(n=500)

    font_resize_factor = random.uniform(.8,2.5)
    default_font_size = 32
    word_imgs = [renderer.render_word(t, size=int(default_font_size*font_resize_factor)) for t in text_list]

    shp = 768,1152
    background_img = Image.new("RGB", shp, (255, 255, 255))
    shp = utils.shape(background_img)
    size, origin = new_textbox_given_background(shp)
    box1, bboxs1 = fill_area_with_words(word_imgs=[w["image"] for w in word_imgs],
                                    bbox=[0,0,*size],
                                    text_list=[w["raw_text"] for w in word_imgs],
                                    max_intraline_vertical_space_offset=5,
                                    error_handling="force",
                                    )

    # Test draw boxes on numpy
    for bbox in bboxs1:
        bbox.draw_box(box1)

    background_img.paste(Image.fromarray(box1), origin)
    ocr_format = convert_to_ocr_format(bboxs1, origin_offset=origin)
    display(background_img)
    return background_img, ocr_format


def draw_boxes(ocr_format, background_img, origin):
    for paragraph in ocr_format["paragraphs"]:
        for line in paragraph["lines"]:
            BBox._draw_box(BBox._offset_origin(line["box"], *origin), background_img)
        BBox._draw_box(BBox._offset_origin(paragraph["box"], *origin), background_img)



if __name__ == "__main__":
    for i in range(0,1):
        background_img, ocr_format = main()



# truncate words
# bad lining up
# last line cutoff