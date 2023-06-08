from tqdm import tqdm
import argparse
from docgen.dataset_utils import draw_boxes_sections_COCO
import sys
import socket
import logging
import json
from docgen.dataset_utils import ocr_dataset_to_coco, load_json, save_json
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def parser_args(args=None):
    parser = argparse.ArgumentParser(description='Process COCO format data')

    parser.add_argument('--ocr_json_path', type=str, required=True,
                        help='The COCO format file')

    if args is not None:
        import shlex
        args = parser.parse_args(shlex.split(args))
    else:
        args = parser.parse_args()

    return args

def read_json_or_npy(json_path):
    logger.info(f"Reading {json_path}")
    if json_path.endswith(".json"):
        with open(json_path, "r") as f:
            data = json.load(f)
    elif json_path.endswith(".npy"):
        import numpy as np
        data = np.load(json_path, allow_pickle=True).item()
    else:
        raise ValueError("Unknown file type: {}".format(json_path))
    logger.info("Done loading")
    return data

def main(args=None):
    args = parser_args(args)

    # read in the coco format file /
    logger.info(f"Loading {args.ocr_json_path}")
    ocr = load_json(args.ocr_json_path)

    # re-encode OCR
    new_path = args.ocr_json_path.replace("OCR", "OCR_NEW")
    logger.info(f"Saving to {new_path}")
    save_json(new_path, ocr)

    # save
    coco = ocr_dataset_to_coco(ocr, exclude_cats="word")
    output_path = args.ocr_json_path.replace("OCR", "COCO_NEW")
    logger.info(f"Saving to {output_path}")
    save_json(output_path, coco)

    # save only word cats
    coco = ocr_dataset_to_coco(ocr, exclude_cats=["section", "paragraph", "line", "margin_note", "paragraph_note", "page_title", "page_header"])
    output_path = args.ocr_json_path.replace("OCR", "COCO_WORD_ONLY")
    logger.info(f"Saving to {output_path}")
    save_json(output_path, coco)


if __name__ == "__main__":
    args = r"""--ocr_json_path 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.1.0\singles\OCR_1000.json' """
    args = r"""--ocr_json_path 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.1.0\OCR.json' """

    args = args.replace("\n","")
    if not args or len(sys.argv) > 1:
        args = sys.argv[1:]
    main(args)
