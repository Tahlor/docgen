TESTING = False # TESTING = disables error handling

if TESTING:
    # set seeds
    seed = 3
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from config.parse_config import parse_config, DEFAULT_CONFIG
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import traceback
from docgen.layoutgen.layoutgen import LayoutGenerator, SectionTemplate
#from docgen.layoutgen.layoutgen import *
from hwgen.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
from textgen.unigram_dataset import Unigrams
from docgen.rendertext.render_word import RenderImageTextPair
from pathlib import Path
from docgen.dataset_utils import load_and_draw_and_display, save_json, ocr_dataset_to_coco
from docgen.degradation.degrade import degradation_function_composition2
from docgen.utils import file_incrementer, handler
import multiprocessing
from docgen.layoutgen.layout_dataset import LayoutDataset
from torch.utils.data import DataLoader
import site


RESOURCES = Path(site.getsitepackages()[0]) / "docgen/resources"
ROOT = Path(__file__).parent.absolute()

def parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to layout config file")
    parser.add_argument("--output", type=str, default=None, help="Path to save images and json files")
    parser.add_argument("--ocr_path", type=str, default=None, help="Path to save OCR json file")
    parser.add_argument("--coco_path", type=str, default=None, help="Path to save COCO json file")
    parser.add_argument("--hwr_files", type=str, default="sample", help="sample, eng_latest, or path to folder with npy\
                                                                    files of pregenerated handwriting, 1 per author style")
    parser.add_argument("--download_all_hw_styles", action="store_true", help="Will download all handwriting styles from S3 if they don't exist")
    parser.add_argument("--unigrams", type=str, default=None, help="Path to unigram file (list of words for text generation)")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite output directory if it exists")
    parser.add_argument("--batch_size", type=int, default=4, help="Number of images to generate at once, will use parallel processing")
    parser.add_argument("--count", type=int, default=100, help="Number of images to generate")
    parser.add_argument("--wikipedia", action="store_true", help="Use wikipedia data for text generation")
    parser.add_argument("--degradation", action="store_true", help="Apply degradation function to images")
    parser.add_argument("--workers", type=int, default=None, help="How many parallel processes to spin up? (default is number of cores - 2)")

    args = parser.parse_args()

    if args.output is None:
        args.output = ROOT / "output" / "french_bmd_output"
    if not args.overwrite and args.output.exists():
        args.output = file_incrementer(args.output, create_dir=True)
    elif not args.output.exists():
        args.output.mkdir(parents=True, exist_ok=True)
    if not args.ocr_path:
        args.ocr_path = args.output / "OCR.json"
    if not args.coco_path:
        args.coco_path = args.output / "COCO.json"
    if args.degradation:
        args.degradation_function = degradation_function_composition2
    else:
        args.degradation_function = None
    if TESTING:
        args.batch_size = 2
        args.count = 2

    if args.workers is None:
        args.workers = max(multiprocessing.cpu_count() - 8,2)
        if TESTING:
            args.workers = 0
    else:
        args.workers = args.workers

    return args


def draw_layout(layout, image):
    image = lg.draw_doc_boxes(layout, image)
    image.show()


def main(opts):
    global render_text_pair, lg

    # Read the YAML file and parse the parameters
    config_dict = parse_config(opts.config)

    # Create a MarginGenerator object for each set of margins
    page_template = SectionTemplate(**config_dict["page_template"])
    page_title_template = SectionTemplate(**config_dict["page_title_template"])
    page_header_template = SectionTemplate(**config_dict["page_header_template"])
    paragraph_template = SectionTemplate(**config_dict["paragraph_template"])
    margin_notes_template = SectionTemplate(**config_dict["margin_notes_template"])
    paragraph_note_template = SectionTemplate(**config_dict["paragraph_note_template"])

    if opts.wikipedia:
        from datasets import load_dataset
        from textgen.wikipedia_dataset import Wikipedia, WikipediaWord
        from textgen.basic_text_encoded_dataset import VOCABULARY, ALPHA_VOCABULARY

        words_dataset = WikipediaWord(
                Wikipedia(
                dataset=load_dataset("wikipedia", "20220301.fr")["train"],
                vocabulary=set(ALPHA_VOCABULARY),  # set(self.model.netconverter.dict.keys())
                exclude_chars="0123456789()+*;#:!/.,",
                min_sentence_length=None,
                max_sentence_length=None,
                shuffle=True,
            ),
            process_fn=["lower"],

        )
    else:
        words_dataset = Unigrams(csv_file=opts.unigrams, newline_freq=0)

    def create_dataset():
        renderer = SavedHandwritingRandomAuthor(
            format="PIL",
            dataset_root=opts.hwr_files,
            #dataset_path=HWR_FILE,
            random_ok=True,
            conversion=None,  # lambda image: np.uint8(image*255)
            font_size=32
        )

        render_text_pair = RenderImageTextPair(renderer, words_dataset)

        # Create a LayoutGenerator object with the parsed parameters
        lg = LayoutGenerator(paragraph_template=paragraph_template,
                             page_template=page_template,
                             page_title_template=page_title_template,
                             margin_notes_template=margin_notes_template,
                             page_header_template=page_header_template,
                             paragraph_note_template=paragraph_note_template,
                             pages_per_image=config_dict["pages_per_image"],
                             img_text_pair_gen=render_text_pair,
                             )

        layout_dataset = LayoutDataset(layout_generator=lg,
                                       render_text_pairs=render_text_pair,
                                       length=opts.count,
                                       degradation_function=opts.degradation_function,
                                       output_path=opts.output,
                                       )

        layout_loader = DataLoader(layout_dataset,
                                   batch_size=opts.batch_size,
                                   collate_fn=layout_dataset.collate_fn,
                                   num_workers=opts.workers)
        return layout_loader, layout_dataset

    layout_loader, layout_dataset = create_dataset()
    ocr_dataset = {}

    # Generate documents
    import time
    start = time.time()

    if True: # use the dataloader
        for batch in tqdm(layout_loader):
            for name,data,img in batch:
                ocr_dataset[name] = data
    else:
        for i in tqdm(range(0,len(layout_loader)*opts.batch_size)):
            name,data,img = layout_dataset[i]
            ocr_dataset[name] = data
    stop = time.time()
    print("TIME: ", stop-start)


    save_json(opts.ocr_path, ocr_dataset)
    coco = ocr_dataset_to_coco(ocr_dataset, "French BMD Layout - v0.0.0.3 pre-Alpha", exclude_cats="word")
    save_json(opts.coco_path, coco)

    ## TEST LAST IMAGE - OCR AND COCO DATASET + BBOXS
    name, d = next(iter(ocr_dataset.items()))
    image_path = opts.output / f"{name}.jpg"

    coco_seg = (image_path.parent / (image_path.stem+"_with_seg")).with_suffix(image_path.suffix)
    coco_box = (image_path.parent / (image_path.stem+"_with_boxes")).with_suffix(image_path.suffix)

    if opts.display_output:
        load_and_draw_and_display(image_path, opts.ocr_path)
        load_and_draw_and_display(image_path, opts.coco_path, format="COCO", draw_boxes=True, draw_segmentations=False
                                   , save_path=coco_box)
        load_and_draw_and_display(image_path, opts.coco_path, format="COCO", draw_boxes=False, draw_segmentations=True, save_path=coco_seg)

if __name__ == "__main__":
    opts = parser()
    main(opts)
