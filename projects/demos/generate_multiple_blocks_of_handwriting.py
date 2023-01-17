import traceback

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
from docgen.image_composition.utils import new_textbox_given_background
from textgen.wikipedia_dataset import Wikipedia
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

""" TODO
* truncate words
* bad lining up
* last line cutoff
"""

ROOT = Path(__file__).parent.absolute()
DEBUG = True

def create_parser():
    global OUTPUT_DICT, OUTPUT_OCR_JSON
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
    parser.add_argument("--resume", action="store_const", const=-1, help="Resuming from previous process")
    parser.add_argument("--freq", default=5000, type=int, help="Frequency of processing")
    parser.add_argument("--output_folder", default=ROOT / "output", help="Path to output directory")
    parser.add_argument("--output_json", default=None, help="Path to output directory")
    parser.add_argument("--incrementer", default=True, help="Increment output folder")
    parser.add_argument("--debug", action="store_true", help="Debugging mode")

    return parser

def process_args(args):
    global OUTPUT_DICT
    print(args)
    args.output_folder = Path(args.output_folder)
    args.last_idx = 0
    if args.saved_handwriting_model is None and args.saved_handwriting_data is None:
        raise ValueError("Must specify either saved handwriting model or saved handwriting data")
    if args.unigrams is None and args.wikipedia is None:
        warnings.warn("No text dataset specified, will try to use unigrams CSV resource (pulled from S3)")
    if args.output_json is None:
        args.output_json = args.output_folder / "output.json"

    if args.resume:
        OUTPUT_DICT = load_json(args.output_json)

        if args.resume == -1:
            args.last_idx = max(int(x) for x in OUTPUT_DICT) + 1
        if args.incrementer:
            warnings.warn("Incrementer is on, but resuming from previous process")
            args.output_folder = file_incrementer(args.output_folder)
    else:
        if args.incrementer:
            args.output_folder = file_incrementer(args.output_folder)

        OUTPUT_DICT = {}

    if args.debug:
        DEBUG = True

    return args


def main(args=None):
    global IDX
    from textgen.basic_text_encoded_dataset import VOCABULARY
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(args))
    args = process_args(args)
    IDX = args.last_idx
    print(f"Vocab: {VOCABULARY}")

    if args.wikipedia is not None:
        basic_text_encoded_dataset = Wikipedia(
            dataset=load_dataset("wikipedia", args.wikipedia)["train"],
            vocabulary=set(VOCABULARY),  # set(self.model.netconverter.dict.keys())
            exclude_chars="0123456789()+*;#:!/",
            min_sentence_length=60,
            max_sentence_length=64
        )
    elif args.unigrams is not None:
        basic_text_encoded_dataset = Unigrams(
            csv_file=args.unigrams,
        )

    if args.saved_handwriting_model is not None:
        renderer = HWGenerator(next_text_dataset=basic_text_encoded_dataset,
                           batch_size=args.batch_size,
                           model="IAM")

    elif args.saved_handwriting_data is not None:
        saved_hw_dataset = SavedHandwritingRandomAuthor(
            format="PIL",
            dataset_root=args.saved_handwriting_data,
            random_ok=True,
            conversion=None,  # lambda image: np.uint8(image*255)
            font_size=32
        )
        # Right now, RenderImageTextPair takes in both the saved dataset and the basic text dataset
        # And produces a batch of images and text
        # But the HWR model takes in a batch of text and produces a batch of images
        # The problem with LIVE generation is, it generates like BATCH SIZE * N words,
        # because half of the effort is getting it to mimic a specific style
        # what you want is the generator to have a text generator that it can be off generating
        # it's own text as much as it wants, and you just pull in whatever it generates,
        # i.e., YOU NEVER NEED THE GENERATOR TO GENERATE SPECIFIC TEXT ON THE FLY
        # THEN THE renderer is the way to go
        renderer = DataLoader(RenderImageTextPair(saved_hw_dataset, basic_text_encoded_dataset),
                              collate_fn=RenderImageTextPair.no_collate_dict,
                              batch_size=args.batch_size)

    text_dataloader = DataLoader(basic_text_encoded_dataset,
                            batch_size=args.batch_size,
                            collate_fn=basic_text_encoded_dataset.collate_fn)
    remainder = 1000
    for i, d in enumerate(text_dataloader):
        process_batch(d, renderer)
        ii = i * args.batch_size
        if remainder > ii % args.freq:
            with args.output_json.open("w") as ff:
                json.dump(OUTPUT_DICT, ff)
        remainder = ii % args.freq

def try_try_again(func):
    def try_again(*args,**kwargs):
        i = 0
        if DEBUG:
            return func(*args,**kwargs)

        while True:
            try:
                i+=1
                result = func(*args, **kwargs)
                return result

            except KeyboardInterrupt as e:
                with OUTPUT_OCR_JSON.open("w") as ff:
                    json.dump(OUTPUT_DICT, ff)
            except Exception as e:
                print(f"ERROR {i} {e}")
                torch.cuda.empty_cache()
                if i >= 10:
                    return
    return try_again

@try_try_again
def process_batch(d, renderer):
    """

    Args:
        d: dict with [{"text", "text_length", "text_idx"}]
        renderer:

    Returns:

    """
    global OUTPUT_DICT, IDX, OUTPUT_PATH

    def new_paragraph(shp, offset=None, random_sample=False):
        if not random_sample:
            nonlocal sample
        else:
            sample = random.choice(new_sample_batch)

        size, origin = new_textbox_given_background(shp)
        if not offset is None:
            origin = origin[0]+offset[0], origin[1]+offset[1]

        scale = random.uniform(.7,1.8)
        box1, localization = fill_area_with_words(word_imgs=sample["word_imgs"],
                                                  bbox=[0,0,*size],
                                                  text_list=sample["raw_text"].split(" "),
                                                  max_vertical_offset_between_words=5,
                                                  error_handling="expand",
                                                  scale=scale
                                                  )
        background_img.paste(Image.fromarray(box1), origin)
        ocr_format = convert_to_ocr_format(localization, origin_offset=origin, section=section)
        return size,origin,box1, localization, ocr_format

    """
    font_resize_factor = random.uniform(.8,2.5)
    default_font_size = 32
    word_imgs = [renderer.render_word(t, size=int(default_font_size*font_resize_factor)) for t in text_list]

    """
    if isinstance(renderer, HWGenerator):
        new_sample_batch = list(renderer.process_batch(d)) # process batch returns [{"words": [PIL], "raw_text": str}]
    # elif isinstance(renderer, SavedHandwriting):
    #     new_sample_batch = renderer.batch_get(d["text"])
    elif isinstance(renderer, RenderImageTextPair):
        new_sample_batch = next(iter(renderer))
    else:
        raise Exception("Unknown renderer type")

    for sample in new_sample_batch:
        section = 0
        canvas_size = 768,1152
        ocr_out = {"sections": [],"width":canvas_size[0], "height":canvas_size[1]}

        background_img = Image.new("RGB", canvas_size, (255, 255, 255))
        canvas_size = utils.shape(background_img)

        text_box_shp, origin, box1, bboxs1, ocr_format = new_paragraph(canvas_size)
        ocr_out["sections"].append(ocr_format)
        section +=1

        # Make another paragraph if there's space
        remaining_vertical_space = canvas_size[1] - origin[1] - text_box_shp[1]
        remaining_horizontal_space = canvas_size[0] - origin[0] - text_box_shp[0]

        if remaining_vertical_space > .25 * canvas_size[1]:
            offset = 0, origin[1] + text_box_shp[1]
            canvas_size = [canvas_size[0],remaining_vertical_space]
            text_box_shp, origin, box1, bboxs1, ocr_format = new_paragraph(canvas_size, offset, random_sample=True)
            ocr_out["sections"].append(ocr_format)

        elif remaining_horizontal_space > .25 * canvas_size[0]:
            canvas_size = [remaining_horizontal_space, canvas_size[1]]
            offset = origin[0] + text_box_shp[0], 0
            text_box_shp, origin, box1, bboxs1, ocr_format = new_paragraph(canvas_size, offset, random_sample=True)
            ocr_out["sections"].append(ocr_format)

        # for bbox in bboxs1:
        #     bbox.draw_box(background_img)

        # display(background_img)

        file_name = f"{IDX:07.0f}"
        #draw_boxes_sections(ocr_out, background_img)

        utils.save_image(background_img,OUTPUT_PATH / (file_name + ".jpg"))
        # print(shape(background_img))
        OUTPUT_DICT[file_name] = ocr_out
        IDX += 1


if __name__ == "__main__":
    output = ROOT / "output"
    command = f"""
    --output_folder {output}
    --batch_size 16 
    --freq 5000 
    --unigrams
    --saved_handwriting_data sample"""
    command2 = f"""
    --output_folder {output}
    --batch_size 16 
    --freq 5000 
    --saved_handwriting_model
    --wikipedia
    """

    for i in range(0,1):
        background_img, ocr_format = main(command)


# truncate words
# bad lining up
# last line cutoff