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

"""
SHOULD BE REFACTORED TO USE "docgen\\datasets\\utils\\combine_jsons.py"
"""


class FindJONS:
    def __init__(self, args=None):
        self.args = self.parse_args(args)
        self.img_count = self.args.img_count
        self.text_labels = {}
        self.coco_labels = {}
        self.ocr_labels = {}
        self.max_json_count = self.args.json_count
        self.write_mode = "w" if self.args.overwrite else "a"
        self.compression = "lzf"
        self.skip_coco = self.args.skip_coco
        self.skip_text = self.args.skip_text
        self.skip_ocr = self.args.skip_ocr

        self.npy_folder = self.args.npy_folder
        
    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("input_folder", help="Path to the folder containing the JSON files")
        parser.add_argument("output_hdf5", help="Path to the output HDF5 file")
        parser.add_argument("--img_count", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--json_count", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite the output HDF5 file if it already exists")
        parser.add_argument("--skip_coco", action="store_true", help="Skip COCO JSON files")
        parser.add_argument("--skip_text", action="store_true", help="Skip TEXT JSON files")
        parser.add_argument("--skip_ocr", action="store_true", help="Skip OCR JSON files")

        parser.add_argument("--npy_folder", type=str, default=None, help="Path to the folder containing the JSON files")

        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))
        return args

    def append_dicts(self, memory_labels, json_data, coco=False):
        if not coco:
            memory_labels.update(json_data)
            return
        for key, value in json_data.items():
            if key in memory_labels:
                if isinstance(value, list):
                    memory_labels[key].extend(json_data[key])
                elif isinstance(value, dict):
                    memory_labels[key].update(json_data[key])

                else:
                    print(
                        f"Warning: Duplicate key '{key}' with non-list and non-dict value found in COCO JSON. Skipping this entry.")
            else:
                memory_labels[key] = value

    @staticmethod
    def dump(hf, name, data_dict):
        labels = hf.create_group(name)
        for key, value in data_dict.items():
            labels.create_dataset(key, data=json.dumps(value))

    @staticmethod
    def coco_dump(hf, name, data_dict):
        labels = hf.create_group(name)
        for key, value in data_dict.items():
            for subkey, subvalue in value.items():
                labels.create_dataset(f"{key}/{subkey}", data=json.dumps(subvalue))

    def parse_files(self):
        # use pathlib
        file_count = 0
        for file in tqdm(Path(self.args.input_folder).iterdir()):

            if self.max_json_count and file_count >= self.max_json_count:
                break

            if not file.name.endswith(".json"):
                continue

            file_path = Path(os.path.join(self.args.input_folder, file))
            with open(file_path, "r") as f:
                data = json.load(f)

            if file_path.stem.startswith("TEXT_") and not self.skip_text:
                file_count += 1
                self.append_dicts(self.text_labels, data)
            elif file_path.stem.startswith("OCR_") and not self.skip_ocr:
                self.append_dicts(self.ocr_labels, data)
            elif file_path.stem.startswith("COCO_") and not self.skip_coco:
                self.append_dicts(self.coco_labels, data, coco=True)

    def delete_existing_datasets(self, hf):
        logger.info("Deleting existing datasets (if any)...")
        for dataset in "text", "ocr", "coco":
            if dataset in hf.keys():
                del hf["labels"][dataset]


    def save_as_npy(self, json_dict, name):
        logger.info(f"Saving {name} as npy")
        np.save(name, np.array(json_dict))

    def save_text_labels_to_hdf5(self):
        # default dict of lists
        from collections import defaultdict
        master_dict = defaultdict(list)
        length_dict = defaultdict(int)

        logger.info("Converting dictionary into lists")
        for idx in tqdm(sorted(self.text_labels.keys())):
            value = self.text_labels[idx]

            for key in value:
                if key == 'text_decode_vocab':
                    master_dict[key].append(value[key].encode('utf-8'))
                elif key == 'style':
                    if isinstance(value[key], list):
                        master_dict[key].append(','.join(value[key]))
                    else:
                        master_dict[key].append(value[key])
                else:
                    master_dict[key].append(value[key])

                length_dict[key] = max(length_dict[key], len(str(value[key])))


        # Create an HDF5 file
        if not self.skip_text:
            logger.info(f"Creating an HDF5 file @ {self.args.output_hdf5}, with {self.img_count} images, max_text_len={length_dict}")
            with h5py.File(self.args.output_hdf5, self.write_mode) as hf:
                #self.delete_existing_datasets(hf)

                # Create datasets for style, text, and text_decode_vocab lists
                for key in master_dict.keys():
                    hf.create_dataset(key, data=master_dict[key], dtype=f'S{length_dict[key]}', compression=self.compression)
        else:
            logger.info(f"Not creating an HDF5 file @ {self.args.output_hdf5} because --skip_text_ocr is set")

    def load_npy(self):
        logger.info("Loading npy files...")
        folder = Path(self.args.npy_folder)
        self.text_labels = np.load(folder / "text_labels.npy", allow_pickle=True).item()
        # self.coco_dict = np.load(folder / "coco_labels.npy", allow_pickle=True).item()
        # self.ocr_dict = np.load(folder / "ocr_labels.npy", allow_pickle=True).item()
        self.img_count = len(self.text_labels)

    def create_npy_files(self):
        root = Path(self.args.output_hdf5).parent
        language = Path(self.args.output_hdf5).stem + "_labels"
        output_folder = (root / language)
        output_folder.mkdir(parents=True, exist_ok=True)

        if not self.skip_text:
            self.save_as_npy(self.text_labels, output_folder / f"text_labels.npy")

        if not self.skip_ocr:
            self.save_as_npy(self.ocr_labels, output_folder / f"ocr_labels.npy")

        if not self.skip_coco:
            self.save_as_npy(self.coco_labels, output_folder / f"coco_labels.npy")

    def main(self):

        if self.args.npy_folder and Path(self.args.npy_folder).exists() and (Path(self.args.npy_folder) / "text_labels.npy").exists():
            self.load_npy()
        else:
            logger.info(f"Npy {self.args.npy_folder} not found, parsing files...")
            self.parse_files()
            self.create_npy_files()

        self.save_text_labels_to_hdf5()


if __name__ == "__main__":
    args = []
    import socket
    if socket.gethostname().lower() == "galois":
        if False:
            "/media/data/1TB/datasets/synthetic/NEW_VERSION/latin /media/data/1TB/datasets/synthetic/NEW_VERSION/latin.h5  --img_count 5000000"
            args = " /media/data/1TB/datasets/synthetic/NEW_VERSION/latin"
            args += " /media/data/1TB/datasets/synthetic/NEW_VERSION/latin.h5"
            args += " --img_count 5000000"
            args += " --json_count 2"
            args += " --overwrite"
        elif False:
            args = fr"'G:\synthetic_data\one_line\french' french.h5"
            args += " --img_count 1000"
        else:
            language = "latin"
            count = 5000000 if language != "english" else 10000000
            args.append(f"/media/data/1TB/datasets/synthetic/NEW_VERSION/{language}")
            args.append(f" /media/data/1TB/datasets/synthetic/NEW_VERSION/{language}.h5")
            args.append(f" --npy_folder '/media/data/1TB/datasets/synthetic/NEW_VERSION/{language}_labels'")
            args.append(f" --img_count {count}")
    elif socket.gethostname().lower() == "pw01ayjg":
        args.append(rf"G:/synthetic_data/one_line/english")
        args.append(rf"G:/synthetic_data/one_line/english.h5")

    args = " ".join(args)

    if sys.argv[1:]:
        args = None

    hf = FindJONS(args)
    hf.main()

