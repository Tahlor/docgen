import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import traceback
from docgen.layoutgen.layoutgen import LayoutGenerator, MarginGenerator
#from docgen.layoutgen.layoutgen import *
from handwriting.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
from textgen.unigram_dataset import Unigrams
from docgen.rendertext.render_word import RenderImageTextPair
from pathlib import Path
from docgen.dataset_utils import load_and_draw_and_display, save_json, ocr_dataset_to_coco
from docgen.degradation.degrade import degradation_function_composition
from docgen.utils import file_incrementer, handler
import multiprocessing
from docgen.layoutgen.layout_dataset import LayoutDataset
from torch.utils.data import DataLoader

TESTING = True # TESTING = disables error handling

#PATH = r"C:\Users\tarchibald\github\handwriting\handwriting\data\datasets\synth_hw\style_298_samples_0.npy"

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
                                        left_margin=(-.05, .02),
                                        right_margin=(-.05, .02))
    margin_margins = MarginGenerator(top_margin=(-.1, .2),
                                     bottom_margin=(-.05, .3),
                                     left_margin=(-.05, .1),
                                     right_margin=(-.08, .1))

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

    DATASETS = Path("/home/taylor/anaconda3/datasets/")
    OUTPUT = DATASETS / "FRENCH_BMD_LAYOUTv2"
    OUTPUT = file_incrementer(OUTPUT, create_dir=True)
    OUTPUT.mkdir(exist_ok=True, parents=True)
    OCR_PATH = OUTPUT / "OCR.json"
    COCO_PATH = OUTPUT / "COCO.json"
    HWR_FILES = Path("/home/taylor/anaconda3/datasets/HANDWRITING_WORD_DATA/")
    HWR_FILE = list(HWR_FILES.rglob("*.npy"))[0]
    #UNIGRAMS = r"C:\Users\tarchibald\github\textgen\textgen\datasets\unigram_freq.csv"
    UNIGRAMS = r"../../textgen/textgen/datasets/unigram_freq.csv"
    WORKERS = max(multiprocessing.cpu_count() - 8,2)
    if TESTING:
        WORKERS = 0
    NUMBER_OF_DOCUMENTS = 100
    BATCH_SIZE = 4

    if False:
        words = Unigrams(csv_file=UNIGRAMS, newline_freq=0)
    else:
        from datasets import load_dataset
        from textgen.wikipedia_dataset import Wikipedia, WikipediaWord
        from handwriting.data.basic_text_dataset import VOCABULARY, ALPHA_VOCABULARY

        words = WikipediaWord(
                Wikipedia(
                dataset=load_dataset("wikipedia", "20220301.en")["train"],
                vocabulary=set(ALPHA_VOCABULARY),  # set(self.model.netconverter.dict.keys())
                exclude_chars="0123456789()+*;#:!/.,",
                min_sentence_length=None,
                max_sentence_length=None,
                shuffle=True,
            ),
            process_fn=lambda x:x.lower()
        )

    def create_dataset():
        renderer = SavedHandwritingRandomAuthor(
            format="PIL",
            dataset_root=HWR_FILES,
            #dataset_path=HWR_FILE,
            random_ok=True,
            conversion=None,  # lambda image: np.uint8(image*255)
            font_size=32
        )

        render_text_pair = RenderImageTextPair(renderer, words)
        layout_dataset = LayoutDataset(layout_generator=lg,
                                render_text_pairs=render_text_pair,
                                output_path=OUTPUT,
                                lenth=NUMBER_OF_DOCUMENTS)
        layout_loader = DataLoader(layout_dataset,
                                   batch_size=BATCH_SIZE,
                                   collate_fn=layout_dataset.collate_fn,
                                   num_workers=WORKERS)
        return layout_loader

    layout_loader = create_dataset()
    ocr_dataset = {}

    for batch in tqdm(layout_loader):
        for name,data in batch:
            ocr_dataset[name] = data

    save_json(OCR_PATH, ocr_dataset)
    coco = ocr_dataset_to_coco(ocr_dataset, "French BMD Layout - v0.0.0.3 pre-Alpha")
    save_json(COCO_PATH, coco)

    ## TEST LAST IMAGE - OCR AND COCO DATASET + BBOXS
    name, d = next(iter(ocr_dataset.items()))
    save_path = OUTPUT / f"{name}.jpg"
    load_and_draw_and_display(save_path, OCR_PATH)
    load_and_draw_and_display(save_path, COCO_PATH, format="COCO")


if __name__ == "__main__":
    main()
