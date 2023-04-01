from docgen.pdf_edit import *
from PIL import Image
from docgen.pdf_edit import PDF
from textgen.unigram_dataset import Unigrams
from textgen.rendertext.render_word import RenderWordFont
from hwgen.data.saved_handwriting_dataset import SavedHandwriting
import numpy as np
from docgen.utils import utils
from hwgen.data.utils import display

PATH= r"C:\Users\tarchibald\github\handwriting\handwriting\data\datasets\synth_hw\style_298_samples_0.npy"
PDF_FILE = r"C:\Users\tarchibald\github\docx_localization\temp\TEMPLATE.pdf"
UNIGRAMS = r"C:\Users\tarchibald\github\textgen\textgen\datasets\unigram_freq.csv"

def create_renderer():
    words = Unigrams(csv_file=UNIGRAMS)
    renderer = SavedHandwriting(format="PIL",
                                dataset_path=PATH,
                                random_ok=True,
                                conversion=lambda image: np.uint8(image*255)
                                )
    return words, renderer

def _test_fill_area_with_words():
    words, renderer = create_renderer()
    img = Image.new("RGB", (800, 1280), (255, 255, 255))
    text_list = [words.sample() for w in range(0,20)]
    word_imgs = [renderer.render_word(word) for word in text_list]
    fill_area_with_words(img,
                         bbox=None,
    )

def numpy_localization_to_pil(img, xdim,ydim):
    return img.swapaxes(xdim,ydim)

def pil_localization_to_numpy(img, xdim, ydim):
    return img.swapaxes(xdim, ydim)

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    words = Unigrams(csv_file=UNIGRAMS)
    renderer = SavedHandwriting(format="numpy",
                                dataset_path=PATH,
                                random_ok=True,
                                conversion=lambda image: np.uint8(image*255)
                                )
    renderer = RenderWordFont(format="numpy")

    text_list = words.sample(n=40)
    word_imgs = [renderer.render_word(t, size=32) for t in text_list]

    background_img = Image.new("RGB", (768, 1152), (255, 255, 255))
    shp = utils.shape(background_img)
    box1, bboxs1 = fill_area_with_words(word_imgs=[w["image"] for w in word_imgs],
                         bbox=[0, 0, 600, 1200],
                         text_list=[w["text_raw"] for w in word_imgs]
                         )
    box2, bboxs2 = fill_area_with_words(word_imgs=[w["image"] for w in word_imgs],
                                        bbox=[0, 0, 300, 1200],
                                        text_list=[w["text_raw"] for w in word_imgs],
                                        max_vertical_offset_between_words= 5
                                        )
    box3, bboxs3 = fill_area_with_words(word_imgs=[w["image"] for w in word_imgs],
                                        bbox=[0, 0, 300, 1200],
                                        text_list=[w["text_raw"] for w in word_imgs],
                                        max_vertical_offset_between_words=15
                                        )

    result = convert_to_ocr_format(bboxs1)

    bbox1_origin = (100, 50)
    bbox2_origin = (50, 450)
    bbox3_origin = (400, 450)
    # bboxs3=offset_bboxes(bboxs3, bbox3_origin)
    # bboxs2=offset_bboxes(bboxs2, bbox2_origin)
    # bboxs1=offset_bboxes(bboxs1, bbox1_origin)
    # specify font by resizing

    # Test draw boxes on numpy
    for bbox in bboxs1:
        bbox.draw_form_element(box1)

    background_img.paste(Image.fromarray(box1), bbox1_origin)
    background_img.paste(Image.fromarray(box2), bbox2_origin)
    background_img.paste(Image.fromarray(box3), bbox3_origin)

    for bbox in bboxs2:
        bbox.offset_origin(*bbox2_origin)
        bbox.draw_form_element(background_img)

    for bbox in bboxs3:
        bbox.offset_origin(*bbox3_origin)
        bbox.draw_form_element(background_img)

    display(background_img)
    pass

