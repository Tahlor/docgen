import sys
import shlex
import argparse
import json
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
import os
import logging
from docgen.hdf5.add_images_to_hdf5 import str_to_int

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# log to stdout
logger.addHandler(logging.StreamHandler())


class FindJONS:
    def __init__(self, args=None):
        self.args = self.parse_args(args)
        self.compression = "lzf"
        self.write_mode = "w" if self.args.overwrite else "a"

    def parse_args(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument("npy_folder", help="Path to the folder containing the JSON files")
        parser.add_argument("output_hdf5", help="Path to the output HDF5 file")
        parser.add_argument("overwrite", action="store_true", help="Overwrite the output HDF5 file if it already exists")

        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))
        return args

    def load_npy(self):
        folder = Path(self.args.npy_folder)
        self.text_dict = np.load(folder / "text_labels.npy", allow_pickle=True).item()
        # self.coco_dict = np.load(folder / "coco_labels.npy", allow_pickle=True).item()
        # self.ocr_dict = np.load(folder / "ocr_labels.npy", allow_pickle=True).item()
        self.img_count = len(self.text_dict)

    def main(self):
        logger.info("Loading npy files")
        self.load_npy()
        
        # Convert dictionary into lists
        style_list = []
        text_list = []
        text_decode_vocab_list = []
        max_text_len = 0
        logger.info("Converting dictionary into lists")
        for idx in tqdm(sorted(self.text_dict.keys())):
            value = self.text_dict[idx]
            style = value['style']
            if isinstance(style, list):
                style = ','.join(style)  # Convert list to string separated by commas
            style_list.append(style)
            text_list.append(value['text'])
            text_decode_vocab_list.append(value['text_decode_vocab'].encode('utf-8'))
            max_text_len = max(max_text_len, len(value['text']), len(value['text_decode_vocab']))

        # Create an HDF5 file
        logger.info(f"Creating an HDF5 file @ {self.args.output_hdf5}, with {self.img_count} images, max_text_len={max_text_len}")
        with h5py.File(self.args.output_hdf5, self.write_mode) as hf:
            # Create datasets for style, text, and text_decode_vocab lists
            hf.create_dataset('style', data=style_list, dtype='S24', compression=self.compression)
            hf.create_dataset('text', data=text_list, dtype=f'S{max_text_len}', compression=self.compression)
            hf.create_dataset('text_decode_vocab', data=text_decode_vocab_list, dtype=f'S{max_text_len}', compression=self.compression)

if __name__ == "__main__":
    if True:
        args = "/media/data/1TB/datasets/synthetic/NEW_VERSION/latin_labels"
        args += " /media/data/1TB/datasets/synthetic/NEW_VERSION/latin2.h5"

    if sys.argv[1:]:
        args = None

    hf = FindJONS(args)
    hf.main()

