import socket

TESTING = False # TESTING = disables error handling

#from textgen.basic_text_dataset import BasicTextDataset

if TESTING:
    # set seeds
    seed = 3
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
from hwgen.daemon import Daemon

import os
from time import sleep
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from projects.french_bmd.config.parse_config import parse_config, DEFAULT_CONFIG
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
from docgen.utils.utils import file_incrementer, handler
import multiprocessing
from docgen.layoutgen.layout_dataset import LayoutDataset
from torch.utils.data import DataLoader
from hwgen.data.hw_generator import HWGenerator
from textgen.basic_text_dataset import VOCABULARY, ALPHA_VOCABULARY
import torch
import site

import logging
logger = logging.getLogger()


RESOURCES = Path(site.getsitepackages()[0]) / "docgen/resources"
ROOT = Path(__file__).parent.absolute()


def parser(args=None):
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
    parser.add_argument("--saved_hw_model_folder", type=str, default=None, help="Use random author for each word")
    parser.add_argument("--saved_hw_model", type=str, default="IAM", help="CVL or IAM, will download from S3")
    parser.add_argument("--unigrams", type=str, default=None, help="Path to unigram file (list of words for text generation)")
    parser.add_argument("--overwrite", type=bool, default=False, help="Overwrite output directory if it exists")
    parser.add_argument("--hw_batch_size", type=int, default=8, help="Number of HW images to generate at once, depends on GPU memory")
    parser.add_argument("--count", type=int, default=None, help="Number of images to generate")
    parser.add_argument("--start_iteration", type=int, default=0, help="Number to start generating from")
    parser.add_argument("--wikipedia", default=None, help="Use wikipedia data for text generation, e.g., 20220301.fr, 20220301.en")
    parser.add_argument("--degradation", action="store_true", help="Apply degradation function to images")
    parser.add_argument("--workers", type=int, default=None, help="How many parallel processes to spin up for LAYOUT only? (default is number of cores - 2)")
    parser.add_argument("--display_output", action="store_true", help="Display output images")
    parser.add_argument("--verbose", action="store_true", help="Display warnings etc.")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, etc.")
    parser.add_argument("--TESTING", action="store_true", help="Enable testing mode")
    parser.add_argument("--vocab", default=VOCABULARY, type=str, help="The list of vocab tokens to use")
    parser.add_argument("--exclude_chars", default="0123456789()+*;#:!/,.", type=str, help="Exclude these chars from the vocab")

    if args is not None:
        import shlex
        args = parser.parse_args(shlex.split(args))
    else:
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


class WordIterator:
    def __init__(self, input_data_iterator, buffer_size=4000):
        self.input_data_iterator = input_data_iterator
        self.buffer_size = buffer_size
        self.renderer_daemon = Daemon(self.input_data_iterator, buffer_size=self.buffer_size)
        self.timeout = .2
        self.renderer_daemon.start()

    def restart_daemon_if_needed(self):
        if not self.renderer_daemon.is_alive():
            print("Daemon thread has died, restarting...")
            old_daemon = self.renderer_daemon
            self.renderer_daemon = Daemon(self.input_data_iterator, buffer_size=self.buffer_size)
            self.renderer_daemon.start()
            old_daemon.join()

    def get_next_word_iterator(self):
        failures = 0
        while True:
            item = None
            try:
                item = self.renderer_daemon.queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                #logger.exception("Timeout waiting for next item")
                failures += 1
                if failures and failures % 10 == 0:
                    logger.info(f"Timeout waiting for next item: {failures}")
                    self.restart_daemon_if_needed()
                continue

            #print(item)
            if item is not None:
                failures = 0
                for i in range(len(item["text_list"])):
                    if 0 in item["word_imgs"][i].shape:
                        continue  # kind of a bug, it's an empty image e.g. \n or something

                    try:
                        yield {"img": item["word_imgs"][i],
                               "text": item["text_list"][i],
                               "style": item["author_id"],
                               "text_decode_vocab": item["text_list_decode_vocab"][i]
                               }
                    except Exception as e:
                        #logger.exception(e)
                        continue

    def get_next_word_queue(self):
        while True:
            item = None
            try:
                item = self.renderer_daemon.queue.get(block=True, timeout=self.timeout)
            except Exception as e:
                #logger.exception("Timeout waiting for next item")
                continue
            #print(item)
            if item is not None:
                for i in range(len(item["text_list"])):
                    if 0 in item["word_imgs"][i].shape:
                        continue  # kind of a bug, it's an empty image e.g. \n or something

                    try:
                        yield {"img": item["word_imgs"][i],
                               "text": item["text_list"][i],
                               "style": item["author_id"],
                               "text_decode_vocab": item["text_list_decode_vocab"][i]
                               }
                    except Exception as e:
                        #logger.exception(e)
                        continue
    def stop(self):
        self.renderer_daemon.stop()
        self.renderer_daemon.join()



def main(opts):
    global lg

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

        words_dataset = WikipediaEncodedTextDataset(
            use_unidecode=True,
            shuffle_articles=True,
            random_starting_word=True,
            dataset=load_dataset("wikipedia", opts.wikipedia)["train"],
            vocabulary=set(opts.vocab),  # set(self.model.netconverter.dict.keys())
            exclude_chars=opts.exclude_chars,
            # symbol_replacement_dict = {
            #     "}": ")",
            #      "{": "(",
            #      "]": ")",
            #      "[": "(",
            #      "–": "-",
            #      " ()": "",
            #      "\n":" "
            # },
            decode_vocabulary="default_expanded",
        )
    else:
        words_dataset = Unigrams(csv_file=opts.unigrams, newline_freq=0)

    render_text_pair = daemon_iterator = None
    def create_dataset():
        nonlocal words_dataset, render_text_pair, daemon_iterator
        if opts.renderer == "saved":
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
        elif opts.renderer == "novel":
            render_text_pair_gen = HWGenerator(next_text_dataset=words_dataset,
                                   batch_size=opts.hw_batch_size,
                                   model=opts.saved_hw_model,
                                   resource_folder=opts.saved_hw_model_folder,
                                   device=opts.device,
                                   style=opts.saved_hw_model,
                                   )

            daemon_iterator = WordIterator(render_text_pair_gen)
            render_text_pair = daemon_iterator.get_next_word_iterator()
            print("LOOPING")
            x = next(render_text_pair)
            print(x)

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
                                       start_idx_offset=opts.start_iteration,
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
    iteration = start_iteration = opts.start_iteration

    try:
        for i, batch in tqdm(enumerate(layout_loader)):
            iteration = i + start_iteration

            if iteration >= opts.count:
                break
            elif iteration > start_iteration and iteration % 1000 == 0:
                ocr_dataset = save_out(ocr_dataset, iteration, opts)
                ocr_dataset = {}

            # batch size should just be 1 since there is no GPU acceleration on LayoutGeneration
            # using multiple workers allows for parallelization, but it doesn't work with the Daemon
            for name,data,img in batch:
                ocr_dataset[name] = data
                img.save(opts.output / f"{name}.jpg")


    except Exception as e:
        print(e)

    # Stop Daemon if needed
    try:
        daemon_iterator.stop()
    except Exception as e:
        print(f"Couldn't stop daemon, {e}")

    save_out(ocr_dataset, iteration + 1, opts)


    stop = time.time()
    print("TIME: ", stop-start)

    if opts.display_output:
        display_output(ocr_dataset)

def save_out(ocr_dataset, i, opts):
    logger.info(f"Saving out at {i}")
    coco_path, ocr_path = add_suffix_to_path(opts.coco_path, i), add_suffix_to_path(opts.ocr_path, i)
    save_json(ocr_path, ocr_dataset)
    coco = ocr_dataset_to_coco(ocr_dataset, "French BMD Layout - v0.0.0.3 pre-Alpha", exclude_cats="word")
    save_json(coco_path, coco)


def add_suffix_to_path(path, index):
    # use pathlib
    path = Path(path)
    return path.parent / (path.stem + f"_{index}" + path.suffix)


def display_output(ocr_dataset):
        ## TEST LAST IMAGE - OCR AND COCO DATASET + BBOXS
        name, d = next(iter(ocr_dataset.items()))
        image_path = opts.output / f"{name}.jpg"

        coco_seg = (image_path.parent / (image_path.stem + "_with_seg")).with_suffix(image_path.suffix)
        coco_box = (image_path.parent / (image_path.stem + "_with_boxes")).with_suffix(image_path.suffix)

        load_and_draw_and_display(image_path, opts.ocr_path)
        load_and_draw_and_display(image_path, opts.coco_path, format="COCO", draw_boxes=True, draw_segmentations=False
                                   , save_path=coco_box)
        load_and_draw_and_display(image_path, opts.coco_path, format="COCO", draw_boxes=False, draw_segmentations=True, save_path=coco_seg)

if __name__ == "__main__":
    if socket.gethostname() == "PW01AYJG":
        args = """
          --config ./config/default.yaml 
          --count 1
          --renderer novel
          --output  /mnt/g/s3/synthetic_data/FRENCH_BMD
          --wikipedia 20220301.fr
          --saved_hw_model IAM
          --hw_batch_size 8
          --workers 0
        """

    elif socket.gethostname() == "Galois":
        args = """
          --config ./config/default.yaml 
          --start 53000
          --count 100000
          --renderer novel
          --output /media/EVO970/data/synthetic/french_bmd/ 
          --saved_hw_model_folder /media/data/1TB/datasets/s3/HWR/synthetic-data/python-package-resources/handwriting-models 
          --wikipedia 20220301.fr
          --saved_hw_model IAM
          --hw_batch_size 80    
          --workers 0
        """
    else:
        args = None

    opts = parser(args)
    main(opts)
