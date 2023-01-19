import os
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import traceback
from docgen.layoutgen.layoutgen import LayoutGenerator, MarginGenerator
#from docgen.layoutgen.layoutgen import *
from hwgen.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
from textgen.unigram_dataset import Unigrams
from docgen.rendertext.render_word import RenderImageTextPair
from pathlib import Path
from docgen.dataset_utils import load_and_draw_and_display, save_json, ocr_dataset_to_coco
from docgen.degradation.degrade import degradation_function_composition
from docgen.utils import file_incrementer, handler
import multiprocessing
from docgen.layoutgen.layout_dataset import LayoutDataset
from torch.utils.data import DataLoader
import site

TESTING = True # TESTING = disables error handling

RESOURCES = Path(site.getsitepackages()[0]) / "docgen/resources"
ROOT = Path(__file__).parent.absolute()

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--ocr_path", type=str, default=None)
    parser.add_argument("--coco_path", type=str, default=None)
    parser.add_argument("--hwr_files", type=str, default="sample", help="sample, eng_latest, or path to folder with npy\
                                                                    files of pregenerated handwriting, 1 per author style")
    parser.add_argument("--unigrams", type=str, default=None)
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--wikipedia", action="store_true")

    args = parser.parse_args()

    if args.output is None:
        args.output = ROOT / "french_bmd_output"
    if not args.overwrite and args.output.exists():
        args.output = file_incrementer(args.output, create_dir=True)
    elif not args.output.exists():
        args.output.mkdir(parents=True, exist_ok=True)
    if not args.ocr_path:
        args.ocr_path = args.output / "OCR.json"
    if not args.coco_path:
        args.coco_path = args.output / "COCO.json"
    return args

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


def main(opts):
    global render_text_pair, lg
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

    WORKERS = max(multiprocessing.cpu_count() - 8,2)
    if TESTING:
        WORKERS = 0

    if opts.wikipedia:
        from datasets import load_dataset
        from textgen.wikipedia_dataset import Wikipedia, WikipediaWord
        from hwgen.data.basic_text_dataset import VOCABULARY, ALPHA_VOCABULARY

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
    else:
        words = Unigrams(csv_file=opts.unigrams, newline_freq=0)

    def create_dataset():
        renderer = SavedHandwritingRandomAuthor(
            format="PIL",
            dataset_root=opts.hwr_files,
            #dataset_path=HWR_FILE,
            random_ok=True,
            conversion=None,  # lambda image: np.uint8(image*255)
            font_size=32
        )

        render_text_pair = RenderImageTextPair(renderer, words)
        layout_dataset = LayoutDataset(layout_generator=lg,
                                render_text_pairs=render_text_pair,
                                output_path=opts.output,
                                lenth=opts.count)
        layout_loader = DataLoader(layout_dataset,
                                   batch_size=opts.batch_size,
                                   collate_fn=layout_dataset.collate_fn,
                                   num_workers=WORKERS)
        return layout_loader

    layout_loader = create_dataset()
    ocr_dataset = {}

    for batch in tqdm(layout_loader):
        for name,data in batch:
            ocr_dataset[name] = data

    save_json(opts.ocr_path, ocr_dataset)
    coco = ocr_dataset_to_coco(ocr_dataset, "French BMD Layout - v0.0.0.3 pre-Alpha")
    save_json(opts.coco_path, coco)

    ## TEST LAST IMAGE - OCR AND COCO DATASET + BBOXS
    name, d = next(iter(ocr_dataset.items()))
    save_path = OUTPUT / f"{name}.jpg"
    load_and_draw_and_display(save_path, opts.ocr_path)
    load_and_draw_and_display(save_path, opts.coco_path, format="COCO")


if __name__ == "__main__":
    opts = parser()
    main(opts)
