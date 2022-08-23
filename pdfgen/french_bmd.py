from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import traceback
from layoutgen.layoutgen import *
from handwriting.data.saved_handwriting_dataset import SavedHandwriting
from textgen.unigram_dataset import Unigrams
from rendertext.render_word import RenderImageTextPair
from pathlib import Path
from pdfgen.dataset_utils import load_and_draw_and_display, save_json
from pdfgen.degradation.degrade import degradation_function_composition
from pdfgen.utils import file_incrementer, handler
import multiprocessing
from pdfgen.layoutgen.layout_dataset import LayoutDataset
from torch.utils.data import DataLoader

TESTING = True
MULTITHREAD = False
PATH = r"C:\Users\tarchibald\github\handwriting\handwriting\data\datasets\synth_hw\style_298_samples_0.npy"
UNIGRAMS = r"C:\Users\tarchibald\github\textgen\textgen\datasets\unigram_freq.csv"


def multiprocess(func, output, iterations):
    if MULTITHREAD:
        temp_results = process_map(func, range(0, iterations),
                                   max_workers=multiprocessing.cpu_count())  # iterates through everything all at once
        for name, result in temp_results:
            if not name is None:
                output[name] = result
    else:
        for i in range(0, iterations):
            name, result = func(i)
            output[name] = result


@handler(testing=TESTING, return_on_fail=(None, None))
def make_one_image(i):
    global lg, OUTPUT
    name = f"{i:07.0f}"
    layout = lg.generate_layout()
    image = lg.render_text(layout, render_text_pair)
    save_path = OUTPUT / f"{name}.jpg"
    image = degradation_function_composition(image)
    image.save(save_path)
    return name, lg.create_ocr(layout, id=i, filename=name)


def draw_layout(layout, image):
    image = lg.draw_doc_boxes(layout, image)
    image.show()


def main():
    global OUTPUT, render_text_pair, lg
    page_margins = MarginGenerator()
    page_header_margins = MarginGenerator(top_margin=(-.02, .02),
                                          bottom_margin=(-.02, .02),
                                          left_margin=(-.02, .5),
                                          right_margin=(-.02, .5))
    paragraph_margins = MarginGenerator(top_margin=(-.05, .05),
                                        bottom_margin=(-.05, .05),
                                        left_margin=(-.02, .02),
                                        right_margin=(-.02, .02))
    margin_margins = MarginGenerator(top_margin=(-.1, .2),
                                     bottom_margin=(-.05, .3),
                                     left_margin=(-.05, .1),
                                     right_margin=(-.05, .1))

    paragraph_note_margins = MarginGenerator(top_margin=(-.05, .2),
                                             bottom_margin=(-.05, .2),
                                             left_margin=(-.05, .2),
                                             right_margin=(-.05, .2))

    lg = LayoutGenerator(paragraph_margins=paragraph_margins,
                         page_margins=page_margins,
                         margin_margins=margin_margins,
                         page_header_margins=page_header_margins,
                         paragraph_note_margins=paragraph_note_margins,
                         margin_notes_probability=.5,
                         page_header_prob=.5,
                         paragraph_note_probability=.5,
                         pages_per_image=(1, 3)
                         )

    renderer = SavedHandwriting(
        format="PIL",
        dataset_path=PATH,
        random_ok=True,
        conversion=None,  # lambda image: np.uint8(image*255)
        font_size=32
    )
    words = Unigrams(csv_file=UNIGRAMS, newline_freq=0)
    render_text_pair = RenderImageTextPair(renderer, words)

    OUTPUT = Path("./temp/FRENCH_BMD_LAYOUTv2")
    OUTPUT = file_incrementer(OUTPUT, create_dir=True)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    OCR_PATH = OUTPUT / "OCR.json"
    COCO_PATH = OUTPUT / "COCO.json"

    layout_dataset = LayoutDataset(layout_generator=lg,
                            render_text_pairs=render_text_pair,
                            output_path=OUTPUT,
                            lenth=1000)
    layout_loader = DataLoader(layout_dataset,
                               batch_size=4,
                               collate_fn=layout_dataset.collate_fn,
                               num_workers=5) # multiprocessing.cpu_count()

    ocr_dataset = {}

    # Multiprocessing
    #multiprocess(make_one_image, ocr_dataset, 5)
    for batch in tqdm(layout_loader):
        for name,data in batch:
            ocr_dataset[name] = data
        pass

    save_json(OCR_PATH, ocr_dataset)
    coco = ocr_dataset_to_coco(ocr_dataset, "French BMD Layout - v0.0.0.1 pre-Alpha")
    save_json(COCO_PATH, coco)

    ## TEST LAST IMAGE - OCR AND COCO DATASET + BBOXS
    name, d = next(iter(ocr_dataset.items()))
    save_path = OUTPUT / f"{name}.jpg"
    load_and_draw_and_display(save_path, OCR_PATH)
    load_and_draw_and_display(save_path, COCO_PATH, format="COCO")


if __name__ == "__main__":
    main()
