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
from hwgen.data.hw_generator import HWGenerator
import torch
import site

import logging
logger = logging.getLogger()


RESOURCES = Path(site.getsitepackages()[0]) / "docgen/resources"
ROOT = Path(__file__).parent.absolute()


def parser():
    global TESTING
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to layout config file")
    parser.add_argument("--output", type=str, default=None, help="Path to save images and json files")
    parser.add_argument("--ocr_path", type=str, default=None, help="Path to save OCR json file")
    parser.add_argument("--coco_path", type=str, default=None, help="Path to save COCO json file")
    parser.add_argument("--renderer", default="saved", help="'saved' for saved handwriting, or 'novel' to generate novel handwriting")
    parser.add_argument("--saved_hw_files", type=str, default="sample", help="sample, eng_latest, or path to folder with npy\
                                                                    files of pregenerated handwriting, 1 per author style")
    parser.add_argument("--saved_hw_model", type=str, default="IAM", help="CVL or IAM, will download from S3")
    parser.add_argument("--unigrams", type=str, default=None, help="Path to unigram file (list of words for text generation)")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite output directory if it exists")
    parser.add_argument("--hw_batch_size", type=int, default=8, help="Number of HW images to generate at once, depends on GPU memory")
    parser.add_argument("--count", type=int, default=None, help="Number of images to generate")
    parser.add_argument("--wikipedia", default=None, help="Use wikipedia data for text generation, e.g., 20220301.fr, 20220301.en")
    parser.add_argument("--degradation", action="store_true", help="Apply degradation function to images")
    parser.add_argument("--workers", type=int, default=None, help="How many parallel processes to spin up for LAYOUT only? (default is number of cores - 2)")
    parser.add_argument("--display_output", action="store_true", help="Display output images")
    parser.add_argument("--verbose", action="store_true", help="Display warnings etc.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc.")
    parser.add_argument("--TESTING", action="store_true", help="Enable testing mode")

    args = parser.parse_args()

    # disable warnings
    if not args.verbose:
        import warnings
        warnings.filterwarnings("ignore")
        hw_logger = logging.getLogger("hwgen.data.saved_handwriting_dataset")
        hw_logger.setLevel(logging.CRITICAL)

    if args.output is None:
        args.output = ROOT / "output" / "french_bmd_output"
    else:
        args.output = Path(args.output)
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
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.TESTING:
        TESTING = True
    if TESTING:
        args.batch_size = 2
        if args.count is None:
            args.count = 2
    if args.count is None:
        args.count = 100

    if args.workers is None:
        args.workers = max(multiprocessing.cpu_count() - 1,1)
        if TESTING:
            args.workers = 0
    else:
        args.workers = args.workers

    logger.info(args)
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

    if opts.wikipedia is not None:
        from datasets import load_dataset
        from textgen.wikipedia_dataset import WikipediaEncodedTextDataset, WikipediaWord
        from textgen.basic_text_dataset import VOCABULARY, ALPHA_VOCABULARY

        words_dataset = WikipediaEncodedTextDataset(
            use_unidecode=True,
            shuffle_articles=True,
            random_starting_word=True,
            dataset=load_dataset("wikipedia", opts.wikipedia)["train"],
            vocabulary=set(ALPHA_VOCABULARY),  # set(self.model.netconverter.dict.keys())
            exclude_chars="",
            symbol_replacement_dict = {
                "}": ")",
                 "{": "(",
                 "]": ")",
                 "[": "(",
                 "–": "-",
                 " ()": "",
                 "\n":" "
            }
        )
    else:
        words_dataset = Unigrams(csv_file=opts.unigrams, newline_freq=0)

    def create_dataset():
        nonlocal words_dataset
        if opts.renderer_daemon == "saved":
            renderer = SavedHandwritingRandomAuthor(
                format="PIL",
                dataset_root=opts.saved_hw_files,
                #dataset_path=HWR_FILE,
                random_ok=True,
                conversion=None,  # lambda image: np.uint8(image*255)
                font_size=32
            )
            if opts.wikipedia is not None:
                words_dataset = WikipediaWord(words_dataset,
                process_fn=["lower"],
                random_next_article=True,)

            render_text_pair = RenderImageTextPair(renderer, words_dataset)
        elif opts.renderer_daemon == "novel":
            render_text_pair = HWGenerator(next_text_dataset=words_dataset,
                                   batch_size=opts.hw_batch_size,
                                   model=opts.saved_hw_model,
                                   device=opts.device,
                                   style=opts.saved_hw_model,
                                   )



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
                                       #output_path=opts.output,
                                       )

        layout_loader = DataLoader(layout_dataset,
                                   batch_size=1, # no GPU usage, so no need to batch; "workers" is your parallelism!
                                   collate_fn=layout_dataset.collate_fn,
                                   num_workers=opts.workers)
        return layout_loader, layout_dataset

    layout_loader, layout_dataset = create_dataset()
    ocr_dataset = {}

    # Generate documents
    import time
    start = time.time()

    for batch in tqdm(layout_loader):
        # batch size should just be 1
        for name,data,img in batch:
            ocr_dataset[name] = data
            img.save(opts.output / f"{name}.jpg")

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
