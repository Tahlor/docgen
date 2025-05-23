import sys
from time import sleep
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import datetime
from docgen.dataset_utils import ocr_dataset_to_coco
import math
import traceback
from tqdm import tqdm
import torch
import json
import random
from docgen.pdf_edit import *
from PIL import Image
from textgen.unigram_dataset import Unigrams, UnigramsData
from textgen.rendertext.render_word import RenderWordFont
from hwgen.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
import numpy as np
from docgen.utils import utils
from hwgen.data.utils import display
from docgen.image_composition.utils import new_textbox_given_background, new_textbox_given_background_line
from textgen.wikipedia_dataset import WikipediaEncodedTextDataset
from hwgen.data.hw_generator import HWGenerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from docgen.utils.utils import file_incrementer
from docgen.dataset_utils import load_json, draw_boxes_sections
from docgen.pdf_edit import convert_to_ocr_format
import argparse
from pathlib import Path
import shlex
from textgen.rendertext.render_word import RenderImageTextPair
from ast import literal_eval as make_tuple
from docgen.render_doc import BoxFiller
import logging
from hwgen.daemon import Daemon
from textgen.basic_text_dataset import VOCABULARY
from docgen.cuda_utils import try_try_again_factory
from torch.nn import functional
from docgen.utils.file_utils import get_last_file_in_collection_matching_base_path
from docgen.image_composition.autocrop import AutoCropper


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ROOT = Path(__file__).parent.absolute()
DEBUG = False
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = "cpu"
TESTING = False

if TESTING:
    # set seeds
    seed = 7
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    DEBUG=True

try_try_again = try_try_again_factory(debug=DEBUG)

class LineGenerator:
    def __init__(self, args=None):
        parser = create_parser()
        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))
        self.args = self.process_args(args)
        logger.info(f"Args: {self.args}")

    def process_args(self, args):
        if not args.no_incrementer and not args.resume:
            args.output_folder = file_incrementer(args.output_folder)
        args.output_folder = Path(args.output_folder)
        args.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"OUTPUT: {Path(args.output_folder).resolve()}")
        args.last_idx = 0
        if args.saved_handwriting_model is None and args.saved_handwriting_data is None:
            raise ValueError("Must specify either saved handwriting model or saved handwriting data")
        if args.unigram_file is None and args.wikipedia is None and args.unigram_list is None:
            warnings.warn("No text dataset specified, will try to use unigrams CSV resource (pulled from S3)")

        # Set up paths
        out = {"OCR": args.output_ocr_json, "COCO": args.output_coco_json, "TEXT": args.output_text_json}
        for k, v in out.items():
            if v is not None:
                setattr(args, f"output_{k.lower()}_json", Path(v))
                out[k].parent.mkdir(parents=True, exist_ok=True)
            else:
                setattr(args, f"output_{k.lower()}_json", args.output_folder / f"{k}.json")

        if args.resume:
            if args.resume > 0:
                args.last_idx = args.resume
            else:
                last_file = get_last_file_in_collection_matching_base_path(args.output_ocr_json)
                logger.info(f"Last file found: {last_file}")
                if last_file:
                    if args.resume == -1:
                        _ocr_dict = load_json(last_file)
                        if _ocr_dict:
                            args.last_idx = max(int(x) for x in _ocr_dict) + 1
                            logger.info(f"Resuming from index {args.last_idx}")
                else:
                    logger.warning("No OCR JSON files found in output folder, starting from scratch")

        if isinstance(args.canvas_size, str):
            args.canvas_size = make_tuple(args.canvas_size)

        if args.autocrop:
            self.autocropper = AutoCropper(crop_mode="horizontal", crop_color=(255,255,255))


        return args

    def main(self):

        if self.args.wikipedia is not None:
            preprocessed = ["en", "fr", "it", "de"]
            if not Path(self.args.wikipedia).suffix[-2:] in preprocessed:
                date, language = self.args.wikipedia.split(".")
                current_year = str(datetime.datetime.now().year)
                date = date.replace("2022", current_year)
                dataset = load_dataset("wikipedia", date=date, language=language, beam_runner="DirectRunner",
                                       cache_dir=self.args.cache_dir)["train"]
            else:
                dataset = load_dataset("wikipedia", self.args.wikipedia, cache_dir=self.args.cache_dir)["train"]

            if self.args.prep_wikipedia_only:
                return

            self.basic_text_dataset = WikipediaEncodedTextDataset(
                dataset=dataset,
                vocabulary=set(self.args.vocab),  # set(self.model.netconverter.dict.keys())
                exclude_chars=self.args.exclude_chars,
                use_unidecode=True,
                min_chars=self.args.min_chars,
                max_chars=self.args.max_chars,
                decode_vocabulary="default_expanded" if not self.args.no_text_decode_vocab else None,
            )
        elif self.args.unigram_file is not None:
            self.basic_text_dataset = Unigrams(
                csv_file=self.args.unigram_file,
            )
        elif self.args.unigram_list is not None:
            self.basic_text_dataset = UnigramsData(
                words=self.args.unigram_list,
                counts=None,
                top_k=10000,
                sample="unweighted",
                newline_freq=0,
                size_override=sys.maxsize,
            )


        self.renderer_daemon = Daemon(
                HWGenerator(
                           next_text_dataset=self.basic_text_dataset,
                           batch_size=self.args.batch_size,
                           model=self.args.saved_handwriting_model,
                           model_path=self.args.saved_hw_model_folder,
                           device=self.args.device,
                           data_split=self.args.style_data_split,
                           iterations_before_new_style=self.args.iterations_before_new_style,
                           )
                , buffer_size=1000,
                )
        self.renderer_daemon.start()
        self.next_word_iterator = self.get_next_word_iterator()
        self.BOX_FILLER = BoxFiller(default_max_lines=self.args.max_lines,
                                    default_error_mode="expand",
                                    img_text_word_dict=self.next_word_iterator,
                                    default_max_words=self.args.max_words,)

        def create_dataset_piece(start_idx, batch_size):
            nonlocal next_img_idx
            try:
                for i in tqdm(range(start_idx, start_idx+batch_size)):
                    self.create_line(next_img_idx)
                    next_img_idx +=1
            except Exception as e:
                logger.exception(e)
                warnings.warn(f"Only generated {i} out of {self.args.count} images")

        self.reset_output_data_dict()
        next_img_idx = self.args.last_idx
        logger.info(f"Starting from index {next_img_idx}")
        while next_img_idx < self.args.count:
            create_dataset_piece(next_img_idx, min(self.args.save_frequency, self.args.count-next_img_idx))
            number_of_zeros_fmt = f"0{int(math.log(self.args.count) // math.log(10)) + 1}d"
            self.save_out_data_dict(f"_{next_img_idx:{number_of_zeros_fmt}}")
            self.reset_output_data_dict()

        self.renderer_daemon.stop()
        self.renderer_daemon.join()

    def save_out_data_dict(self, suffix):
        # add suffix to paths
        OCR = self.args.output_ocr_json.with_name(self.args.output_ocr_json.stem + suffix + self.args.output_ocr_json.suffix)
        TEXT = self.args.output_text_json.with_name(self.args.output_text_json.stem + suffix + self.args.output_text_json.suffix)
        COCO = self.args.output_coco_json.with_name(self.args.output_coco_json.stem + suffix + self.args.output_coco_json.suffix)
        logger.info(f"Saving out to {suffix}")
        with OCR.open("w") as f:
            json.dump(self.ocr_dict, f)
        self.dump_text_json(TEXT)
        coco = ocr_dataset_to_coco(self.ocr_dict, f"{self.args.wikipedia} Lines - v0.1.0.0 - piece {suffix}", exclude_cats="word")
        with COCO.open("w") as f:
            json.dump(coco, f)

    def dump_text_json(self, TEXT):
        with TEXT.open("w") as f:
            if self.args.no_text_decode_vocab:
                out = {k: {"text": d['sections'][0]['paragraphs'][0]["lines"][0]["text"],
                           "style": d['sections'][0]["style"],
                           }
                       for k, d in self.ocr_dict.items()}
            else:
                out = {k:
                           {"text": d['sections'][0]['paragraphs'][0]["lines"][0]["text"],
                            "text_decode_vocab": d['sections'][0]['paragraphs'][0]["lines"][0]["text_decode_vocab"],
                            "style": d['sections'][0]["style"],
                            }
                       for k, d in self.ocr_dict.items()}
            json.dump(out, f)

    def reset_output_data_dict(self):
        self.ocr_dict = {}

    def get_next_word_iterator(self):
        item = self.renderer_daemon.queue.get()
        while True:
            for i in range(len(item["text_list"])):
                if 0 in item["word_imgs"][i].shape:
                    continue # kind of a bug, it's an empty image e.g. \n or something
                # yield item["word_imgs"][i], item["text_list"][i], item["author_id"][i]
                yield {"img": item["word_imgs"][i],
                       "text": item["text_list"][i],
                       "style": item["author_id"],
                       "text_decode_vocab": item["text_list_decode_vocab"][i]
                       }
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
        # if random.random() < .5:
        #     raise Exception("Randomly failing to test try_try_again")

        def new_paragraph(shp, offset=None):
            scale = random.uniform(.7, 1.8)
            font_size = scale * 32
            size, origin, font_size, bbox = new_textbox_given_background_line(shp,
                                                                            font_size=font_size,
                                                                            minimum_width_percent=self.args.min_width)
            if not offset is None:
                bbox.offset_origin(*offset)

            box_dict = self.BOX_FILLER.fill_box(bbox=bbox,
                                                     img=background_img,
                                                     )
            image = box_dict["img"]
            localization = box_dict["bbox_list"]
            styles = box_dict["styles"]

            ocr_format = convert_to_ocr_format(localization, section=section, text_decode_vocab=not self.args.no_text_decode_vocab)
            ocr_format["style"] = tuple(styles)

            return size, origin, image, localization, ocr_format

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

        text_box_shp, origin, img, bboxs1, ocr_format = new_paragraph(canvas_size)
        ocr_out["sections"].append(ocr_format)
        section += 1

        file_name = f"{idx:09.0f}"
        # draw_boxes_sections(ocr_out, background_img)

        if self.autocropper:
            if random.random() < self.args.random_rotation_probability:
                background_img = background_img.transpose(Image.ROTATE_90)

            background_img = self.autocropper.crop(background_img)
            # randomly flip 180 or mirror
            if random.random() < self.args.random_mirror_probability:
                background_img = background_img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < self.args.random_vertical_flip_probability:
                background_img = background_img.transpose(Image.ROTATE_180)



        utils.save_image(background_img, self.args.output_folder / (file_name + ".jpg"))
        # print(shape(background_img))
        self.ocr_dict[file_name] = ocr_out

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_handwriting_data",
                        action="store", const="sample", nargs="?",
                        help="Path to saved handwriting, 'sample' or 'eng_latest' to pull from S3")
    parser.add_argument("--saved_handwriting_model",
                        action="store", const="IAM", nargs="?",
                        help="Path to HWR model, OR 'CVL' or 'IAM'",
                        )
    parser.add_argument("--saved_hw_model_folder", type=str, default=None, help="Use random author for each word")
    parser.add_argument("--style_data_split", default="all", type=str, help="train, test, or all")
    parser.add_argument("--unigram_file", action="store_const", const=True,
                        help="Path to unigram frequency file, if 'true' it will be downloaded from S3")
    parser.add_argument("--unigram_list", help="List of unigrams to use", nargs="+", default=None)
    parser.add_argument("--wikipedia", action="store", const="20220301.en", nargs="?",
                        help="20220301.en, 20220301.fr, etc.")
    parser.add_argument("--cache_dir", action="store", const=None, nargs="?",
                        help="where to store the downloaded files")
    parser.add_argument("--vocab", default=VOCABULARY, type=str, help="The list of vocab tokens to use")
    parser.add_argument("--no_text_decode_vocab", action="store_true", help="Don't save out text with alternate vocabulary")
    parser.add_argument("--exclude_chars", default="0123456789()+*;#:!/,.", type=str, help="Exclude these chars from the vocab")
    parser.add_argument("--batch_size", default=12, type=int, help="Batch size for processing")
    parser.add_argument("--count", default=100, type=int, help="Total number of images to generate")
    parser.add_argument('--resume', type=int, default=None, nargs='?', const=-1,
                            help='What index to start geneating from; -1 will attempt to infer last index from last JSON')

    parser.add_argument("--save_frequency", default=50, type=int, help="How often to update JSON GT file, in case generation is interrupted")
    parser.add_argument("--output_folder", default=ROOT / "output", help="Path to output directory")
    parser.add_argument("--output_ocr_json", default=None, help="Path to output json (OCR format)")
    parser.add_argument("--output_text_json", default=None, help="Path to output json (just text transcriptions)")
    parser.add_argument("--output_coco_json", default=None, help="Path to output json (COCO format)")
    parser.add_argument("--no_incrementer", action="store_true", help="DON'T increment output folder")
    parser.add_argument("--debug", action="store_true", help="Debugging mode")
    parser.add_argument("--display_output", action="store_true", help="Display sample output segmentation")
    parser.add_argument("--canvas_size", default=(1152, 48), type=str, help="Canvas size W x H")
    parser.add_argument("--autocrop", action="store_true", help="Crop whitespace from output images")
    parser.add_argument("--min_chars", default=8, type=int, help="Min chars to be generated by textgen")
    parser.add_argument("--max_chars", default=20*8, type=int, help="Max chars to be generated by textgen")
    parser.add_argument("--max_words", default=sys.maxsize, type=int, help="Max words in an image filled by BoxFiller")
    parser.add_argument("--max_lines", default=1, type=int, help="Max lines in a paragraph")
    parser.add_argument("--max_paragraphs", default=1, type=int, help="Max paragraphs in a document")
    parser.add_argument("--min_width", default=.75, type=int, help="Minimum width of a textbox as a percent of document")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu, cuda, cuda:0, etc.")
    parser.add_argument("--prep_wikipedia_only", action="store_true", help="download or prepare wikipedia data only without processing")
    parser.add_argument("--iterations_before_new_style", default=100, type=int, help="How many iterations before changing style")
    parser.add_argument("--random_vertical_flip_probability", default=0, type=float)
    parser.add_argument("--random_mirror_probability", default=0, type=float)
    parser.add_argument("--random_rotation_probability", default=0, type=float)

    return parser

def _testing():
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

    # --saved_hw_model_folder /media/data/1TB/datasets/s3/HWR/synthetic-data/python-package-resources/handwriting-models

    configurations = [
        {
            "letter": "'l' 'L'",
            "params": {
                "random_vertical_flip_probability": .5,
                "random_mirror_probability": .5,
                "random_rotation_probability": 1,
                "max_words": 1,
                "count": 1000,
            }
        },
        {
            "letter": "'x'",
            "params": {
                "random_vertical_flip_probability": .5,
                "random_mirror_probability": .5,
                "random_rotation_probability": .5,
                "max_words": 1,
                "count": 1000,
            }
        },
        {
            "letter": "'-' '-' '-' ",
            "params": {
                "random_vertical_flip_probability": .5,
                "random_mirror_probability": .5,
                "random_rotation_probability": 0,
                "max_words": 1,
                "count": 1000,
            }
        },
        {
            "letter": "'Le' 'le'",
            "params": {
                "random_vertical_flip_probability": 0,
                "random_mirror_probability": 0,
                "random_rotation_probability": 0,
                "max_words": 1,
                "count": 1000,
            }
        },
        {
            "letter": "'Lan' 'lan'",
            "params": {
                "random_vertical_flip_probability": 0,
                "random_mirror_probability": 0,
                "random_rotation_probability": 0,
                "max_words": 1,
                "count": 500,
            }
        }
    ]

    for config in configurations[2:]:
        letter = config["letter"]
        params = config["params"]
        letter_folder = letter.split(" ")[0].replace("'", "")

        args = f"""--batch_size 16 
        --saved_handwriting_model IAM
        --unigram_list {letter}
        --output_folder '{ROOT / "output" / letter_folder}'
        --max_lines 1
        --max_chars 1
        --min_chars 1
        --max_words {params["max_words"]}
        --canvas_size 96,48
        --saved_handwriting_model IAM
        --autocrop
        --count {params["count"]}
        --save_frequency 1000
        --iterations_before_new_style 1
        --random_vertical_flip_probability {params["random_vertical_flip_probability"]}
        --random_mirror_probability {params["random_mirror_probability"]}
        --random_rotation_probability {params["random_rotation_probability"]}
        """.replace("\n", " ")
        print(f"output folder: {ROOT / 'output' / letter_folder}")
        try:
            ocr_format = LineGenerator(args=args).main()
        except:
            pass
        print(f"Done with {letter}")

