from time import sleep
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from docgen.dataset_utils import ocr_dataset_to_coco
import traceback
from tqdm import tqdm
import torch
import json
import random
from docgen.pdf_edit import *
from PIL import Image
from textgen.unigram_dataset import Unigrams
from docgen.rendertext.render_word import RenderWordFont
from hwgen.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
import numpy as np
from docgen import utils
from hwgen.data.utils import display
from docgen.image_composition.utils import new_textbox_given_background, new_textbox_given_background_line
from textgen.wikipedia_dataset import WikipediaEncodedTextDataset
from hwgen.data.hw_generator import HWGenerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from docgen.utils import file_incrementer
from docgen.dataset_utils import load_json, draw_boxes_sections
from docgen.pdf_edit import convert_to_ocr_format
import argparse
from pathlib import Path
import shlex
from docgen.rendertext.render_word import RenderImageTextPair
from ast import literal_eval as make_tuple
from docgen.render_doc import BoxFiller
import logging
from hwgen.daemon import Daemon
from textgen.basic_text_dataset import VOCABULARY
from docgen.cuda_utils import try_try_again_factory

try_try_again = try_try_again_factory(debug=True)

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent.absolute()
DEBUG = True
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"

class LineGenerator:
    def __init__(self, args=None):
        parser = create_parser()
        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))
        self.args = process_args(args)

    def main(self):
        print(f"Vocab: {VOCABULARY}")
        if self.args.wikipedia is not None:
            self.basic_text_dataset = WikipediaEncodedTextDataset(
                dataset=load_dataset("wikipedia", self.args.wikipedia)["train"],
                vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
                exclude_chars="0123456789()+*;#:!/,.",
                use_unidecode=True,
                min_chars=self.args.min_chars,
                max_chars=self.args.max_chars,
            )

        elif self.args.unigrams is not None:
            basic_text_dataset = Unigrams(
                csv_file=self.args.unigrams,
            )

        self.renderer_daemon = Daemon(
                HWGenerator(
                           next_text_dataset=self.basic_text_dataset,
                           batch_size=self.args.batch_size,
                           model=self.args.saved_handwriting_model,
                           device=self.args.device
                            )
                , buffer_size=100,
                )
        self.renderer_daemon.start()
        self.next_word_iterator = self.get_next_word_iterator()
        self.BOX_FILLER = BoxFiller(default_max_lines=self.args.max_lines,
                                   default_error_mode="expand",
                                   img_text_pair_gen=self.next_word_iterator)

        if True:
            for i in tqdm(range(self.args.count)):
                self.create_line(i)
        #except Exception as e:
        else:
            logger.exception(e)
            warnings.warn(f"Only generated {i} out of {self.args.count} images")

        self.renderer_daemon.stop()
        self.renderer_daemon.join()

        with self.args.output_ocr_json.open("w") as f:
            json.dump(ocr_dict, f)
        with self.args.output_text_json.open("w") as f:
            out = {k:d['sections'][0]['paragraphs'][0]["lines"][0]["text"] for k,d in ocr_dict.items()}
            json.dump(out, f)
        coco = ocr_dataset_to_coco(ocr_dict, "French Lines - v0.0.1.0 Alpha", exclude_cats="word")
        with self.args.output_coco_json.open("w") as f:
            json.dump(coco, f)
        return ocr_dict

    def get_next_word_iterator(self):
        item = self.renderer_daemon.queue.get()
        while True:
            for i in range(len(item["text_list"])):
                if 0 in item["word_imgs"][i].shape:
                    continue # kind of a bug, it's an empty image e.g. \n or something

                yield item["word_imgs"][i], item["text_list"][i]
            while True:
                try:
                    item = self.renderer_daemon.queue.get()
                    break
                except:
                    #sleep(.5)
                    #logger.exception("Timeout waiting for next item")
                    #logger.warning("Timeout waiting for next item")
                    continue

    @try_try_again
    def create_line(self, idx):
        """
        """
        def new_paragraph(shp, offset=None):
            scale = random.uniform(.7, 1.8)
            font_size = scale * 32
            if self.args.max_lines > 1:
                size, origin = new_textbox_given_background(shp, font_size=font_size)
            else:
                size, origin, font_size = new_textbox_given_background_line(shp,
                                                                            font_size=font_size,
                                                                            minimum_width_percent=self.args.min_width)

            if not offset is None:
                origin = origin[0] + offset[0], origin[1] + offset[1]

            box1, localization = self.BOX_FILLER.fill_box(bbox=[0, 0, *size],
                                                     img=background_img,
                                                     )

            ocr_format = convert_to_ocr_format(localization, origin_offset=origin, section=section)
            return size, origin, box1, localization, ocr_format

        """
        font_resize_factor = random.uniform(.8,2.5)
        default_font_size = 32
        word_imgs = [renderer.render_word(t, size=int(default_font_size*font_resize_factor)) for t in text_list]
        """
        section = 0
        canvas_size = self.args.canvas_size
        ocr_out = {"sections": [], "width": canvas_size[0], "height": canvas_size[1]}

        background_img = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas_size = utils.shape(background_img)

        text_box_shp, origin, box1, bboxs1, ocr_format = new_paragraph(canvas_size)
        ocr_out["sections"].append(ocr_format)
        section += 1

        file_name = f"{idx:07.0f}"
        # draw_boxes_sections(ocr_out, background_img)

        utils.save_image(background_img, OUTPUT_PATH / (file_name + ".jpg"))
        # print(shape(background_img))
        ocr_dict[file_name] = ocr_out

def create_parser():
    global ocr_dict, OUTPUT_OCR_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_handwriting_data",
                        action="store", const="sample", nargs="?",
                        help="Path to saved handwriting, 'sample' or 'eng_latest' to pull from S3")
    parser.add_argument("--saved_handwriting_model",
                        action="store", const="IAM", nargs="?",
                        help="Path to HWR model, OR 'CVL' or 'IAM'",
                        )
    parser.add_argument("--unigrams", action="store_const", const=True,
                        help="Path to unigram frequency file, if 'true' it will be downloaded from S3")
    parser.add_argument("--wikipedia", action="store", const="20220301.en", nargs="?",
                        help="20220301.en, 20220301.fr, etc.")
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for processing")
    parser.add_argument("--count", default=100, type=int, help="Batch size for processing")
    parser.add_argument("--resume", action="store_const", const=-1, help="Resuming from previous process")
    parser.add_argument("--freq", default=5000, type=int, help="How often to update JSON GT file, in case generation is interrupted")
    parser.add_argument("--output_folder", default=ROOT / "output", help="Path to output directory")
    parser.add_argument("--output_ocr_json", default=None, help="Path to output json (OCR format)")
    parser.add_argument("--output_text_json", default=None, help="Path to output json (just text transcriptions)")
    parser.add_argument("--output_coco_json", default=None, help="Path to output json (COCO format)")
    parser.add_argument("--incrementer", default=True, help="Increment output folder")
    parser.add_argument("--debug", action="store_true", help="Debugging mode")
    parser.add_argument("--display_output", action="store_true", help="Display sample output segmentation")
    parser.add_argument("--canvas_size", default=(1152, 48), type=str, help="Canvas size")
    parser.add_argument("--min_chars", default=8, type=int, help="Min chars to be generated by textgen")
    parser.add_argument("--max_chars", default=20*8, type=int, help="Max chars to be generated by textgen")
    parser.add_argument("--max_lines", default=1, type=int, help="Max lines in a paragraph")
    parser.add_argument("--max_paragraphs", default=1, type=int, help="Max paragraphs in a document")
    parser.add_argument("--min_width", default=.75, type=int, help="Minimum width of a textbox as a percent of document")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu, cuda, cuda:0, etc.")
    return parser

def process_args(args):
    global ocr_dict, OUTPUT_PATH
    print(args)
    if args.incrementer:
        args.output_folder = Path(file_incrementer(args.output_folder))
    else:
        args.output_folder = Path(args.output_folder)
    args.output_folder.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH = args.output_folder
    print(f"OUTPUT: {Path(OUTPUT_PATH).resolve()}")
    args.last_idx = 0
    if args.saved_handwriting_model is None and args.saved_handwriting_data is None:
        raise ValueError("Must specify either saved handwriting model or saved handwriting data")
    if args.unigrams is None and args.wikipedia is None:
        warnings.warn("No text dataset specified, will try to use unigrams CSV resource (pulled from S3)")

    # Set up paths
    out = {"OCR": args.output_ocr_json, "COCO": args.output_coco_json, "TEXT": args.output_text_json}
    for k, v in out.items():
        if v is not None:
            setattr(args, f"output_{k.lower()}_json", Path(v))
            out[k].parent.mkdir(parents=True, exist_ok=True)
        else:
            setattr(args, f"output_{k.lower()}_json", args.output_folder / f"{k}.json")

    if isinstance(args.canvas_size, str):
        args.canvas_size = make_tuple(args.canvas_size)

    if args.resume:
        ocr_dict = load_json(args.output_ocr_json)

        if args.resume == -1:
            args.last_idx = max(int(x) for x in ocr_dict) + 1
        if args.incrementer:
            warnings.warn("Incrementer is on, but resuming from previous process")
            args.output_folder = file_incrementer(args.output_folder)
    else:
        if args.incrementer:
            args.output_folder = file_incrementer(args.output_folder)

        ocr_dict = {}

    if args.debug:
        DEBUG = True

    return args


def testing():
    output = ROOT / "output"

    # USE SAVED HANDWRITING - NOT WORKING RIGHT NOW
    command = fr"""
    --output_folder {output}
    --batch_size 16 
    --freq 5000 
    --unigrams
    --saved_handwriting_data sample""".replace("\n"," ")

    # GENERATE NEW HANDWRITING ON THE FLY -- NEEDED FOR FRENCH
    # --output_folder {output}
    command2 = rf"""
    --batch_size 16 
    --freq 5000 
    --saved_handwriting_model IAM
    --wikipedia 20220301.fr
    --max_lines 1
    --max_chars 10
    --min_chars 3
    --canvas_size 1152,48
    """.replace("\n"," ")

    # GENERATE FRENCH PARAGRAPHS
    command3 = rf"""
    --output_folder ./output --batch_size 16  
    --freq 1 
    --saved_handwriting_model IAM
     --wikipedia 20220301.fr 
    --canvas_size 768,1152 
    --min_chars 50 
    --max_chars 64 
    --max_lines 100 
    --max_paragraphs 2
    """

    background_img, ocr_format = main(command3)
    return background_img, ocr_format

if __name__ == "__main__":
    # ' --output_folder C:\\Users\\tarchibald\\github\\docgen\\projects\\demos\\output --batch_size 16  --freq 1  --saved_handwriting_model IAM --wikipedia 20220301.fr '

    ocr_format = main()

    #testing()

# truncate words
# bad lining up
# last line cutoff