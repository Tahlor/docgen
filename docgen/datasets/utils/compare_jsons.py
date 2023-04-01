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
from docgen.dataset_utils import ocr_dataset_to_coco, load_json, save_json
from docgen.dataset_utils import CustomDecoder
import json
import os

import json
import os

def compare_nested_jsons(filepath1, filepath2):
    # Load JSONs
    max_level = 0
    logger.info(f"Loading {filepath1} using CustomDecoder")
    data1 = load_json(filepath1)

    logger.info(f"Loading {filepath2} using json.load")
    with open(filepath2, 'r') as f:
        data2 = json.load(f, cls=CustomDecoder)

    # Add some dummy data to data2
    data2["1000000"] = []
    data2["0000000"] = {
        "sections": [
            {
                "paragraphs": [
                    {
                        "lines": [
                            {
                                "bbox": [10, 241, 184, 268],
                            }
    ]}]}]}

    def recursive_compare(data1, data2, path="", level=0):
        nonlocal max_level
        if level > max_level:
            max_level = level
            logger.info(f"Max level: {max_level}")

        if type(data1) != type(data2):
            print(f"Different type at {path}: {type(data1)} vs {type(data2)}")
        elif isinstance(data1, dict):
            for key in data1.keys():
                if key not in data2:
                    print(f"Missing key in second json at {path}: {key}")
                else:
                    recursive_compare(data1[key], data2[key], path + f".{key}", level+1)
            for key in data2.keys():
                if key not in data1:
                    print(f"Missing key in first json at {path}: {key}")
        elif isinstance(data1, list):
            for i in range(min(len(data1), len(data2))):
                recursive_compare(data1[i], data2[i], path + f"[{i}]", level+1)
            if len(data1) > len(data2):
                print(f"First list longer at {path}")
            elif len(data1) < len(data2):
                print(f"Second list longer at {path}")
        else:  # for simple data types, e.g. int, string, etc.
            if data1 != data2:
                print(f"Different values at {path}: {data1} vs {data2}")

    recursive_compare(data1, data2, level=0)



# Example usage
filepath1 = "G:/s3/synthetic_data/FRENCH_BMD/FRENCH_BMD_LAYOUTv2.1.0/singles/OCR_1000.json"
filepath2 = "G:/s3/synthetic_data/FRENCH_BMD/FRENCH_BMD_LAYOUTv2.1.0/singles/OCR_NEW_1000.json"

compare_nested_jsons(filepath1, filepath2)

## TO DO:
# re-encode OCR?
# make updated COCO so that you can filter by root type?
# fix the text generation images so they don't generate off of the image
# COCO is too big because it includes all of the word level stuff