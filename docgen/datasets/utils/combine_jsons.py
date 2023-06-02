import sys
import shlex
import argparse
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# log to stdout
logger.addHandler(logging.StreamHandler())


class FindJONS:
    def __init__(self, args=None):
        self.args = self.parse_args(args)
        self.img_count = self.args.img_count
        self.text_dict = {}
        self.coco_dict = {}
        self.ocr_dict = {}
        self.max_json_count = self.args.json_count
        self.write_mode = "w" if self.args.overwrite else "a"
        self.skip_coco = self.args.skip_coco
        self.skip_text = self.args.skip_text
        self.skip_ocr = self.args.skip_ocr
        self.do_save_npy = self.args.save_npy

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("input_folder", help="Path to the folder containing the JSON files")
        parser.add_argument("--npy_folder", help="Path to the folder containing the NPY files", default=None)
        parser.add_argument("--output_folder", default=None, help="Folder to save the JSON files")
        parser.add_argument("--img_count", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--json_count", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--overwrite", action="store_true",
                            help="")
        parser.add_argument("--skip_coco", action="store_true", help="Skip COCO JSON files")
        parser.add_argument("--skip_text", action="store_true", help="Skip TEXT JSON files")
        parser.add_argument("--skip_ocr", action="store_true", help="Skip OCR JSON files")
        parser.add_argument("--save_npy", action="store_true", help="Save the numpy arrays")

        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))

        if args.output_folder is None:
            args.output_folder = args.input_folder

        if args.output_folder is None:
            args.output_folder = args.input_folder
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
                self.append_dicts(self.text_dict, data)
            elif file_path.stem.startswith("OCR_") and not self.skip_ocr:
                self.append_dicts(self.ocr_dict, data)
            elif file_path.stem.startswith("COCO_") and not self.skip_coco:
                self.append_dicts(self.coco_dict, data, coco=True)

    def save_as_npy(self, json_dict, name):
        logger.info(f"Saving {name} as npy")
        np.save(name, np.array(json_dict))

    def load_npy(self, folder=None):
        if folder is None:
            folder = self.args.npy_folder
        logger.info("Loading npy files...")
        folder = Path(folder)
        if (folder / "text_labels.npy").exists():
            self.text_dict = np.load(folder / "text_labels.npy", allow_pickle=True).item()
        if (folder / "coco_labels.npy").exists():
            self.coco_dict = np.load(folder / "coco_labels.npy", allow_pickle=True).item()
        if (folder / "ocr_labels.npy").exists():
            self.ocr_dict = np.load(folder / "ocr_labels.npy", allow_pickle=True).item()

    def create_aggregate_npy_files(self):

        if not self.skip_text:
            self.save_as_npy(self.text_dict, self.args.output_folder / f"text_labels.npy")

        if not self.skip_ocr:
            self.save_as_npy(self.ocr_dict, self.args.output_folder / f"ocr_labels.npy")

        if not self.skip_coco:
            self.save_as_npy(self.coco_dict, self.args.output_folder / f"coco_labels.npy")

    def save_json(self, json_dict, path):
        logger.info(f"Saving {path}")
        with open(path, self.write_mode) as f:
            json.dump(json_dict, f, indent=4)

    def create_aggregate_jsons(self):
        if not self.skip_coco and self.coco_dict:
            self.save_json(self.coco_dict, self.args.output_folder / f"COCO.json")
        if not self.skip_ocr and self.ocr_dict:
            self.save_json(self.ocr_dict, self.args.output_folder / f"OCR.json")
        if not self.skip_text and self.text_dict:
            self.save_json(self.text_dict, self.args.output_folder / f"TEXT.json")

    def main(self):
        Path(self.args.output_folder).mkdir(exist_ok=True, parents=True)

        self.parse_files()
        if self.do_save_npy:
            self.create_aggregate_npy_files()
        self.create_aggregate_jsons()


if __name__ == "__main__":
    args = []

    if sys.argv[1:]:
        args = None

    hf = FindJONS(args)
    hf.main()

