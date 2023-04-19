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
        self.img_count = self.args.img_count
        self.text_labels = {}
        self.coco_labels = {}
        self.ocr_labels = {}
        self.max_json_count = self.args.json_count
        self.write_mode = "w" if self.args.overwrite else "a"

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("input_folder", help="Path to the folder containing the JSON files")
        parser.add_argument("output_hdf5", help="Path to the output HDF5 file")
        parser.add_argument("--img_count", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--json_count", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite the output HDF5 file if it already exists")

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

            if file_path.stem.startswith("TEXT_"):
                file_count += 1
                self.append_dicts(self.text_labels, data)
            elif file_path.stem.startswith("OCR_"):
                self.append_dicts(self.ocr_labels, data)
            elif file_path.stem.startswith("COCO_"):
                self.append_dicts(self.coco_labels, data, coco=True)

    def delete_existing_datasets(self, hf):
        logger.info("Deleting existing datasets (if any)...")
        for dataset in "text", "ocr", "coco":
            if dataset in hf.keys():
                del hf["metadata"][dataset]

    def add_dict_to_hdf5(self, dataset, data_dict):
        for key, value in data_dict.items():
            key = str_to_int(key)
            dataset[key] = json.dumps(value)

    def create_dataset_for_each_key_in_label_dict(self, hf, data_dict):
        datasets = next(iter(data_dict.values())).keys()
        dataset_ref_dict = {}
        for dataset_name in datasets: # text, style, text_decode_vocab
            if dataset_name in hf.keys():
                dataset = hf.get(dataset_name)
            else:
                # Create dataset if it doesn't exist
                size = max(len(data_dict), self.img_count)
                dataset = hf.create_dataset(dataset_name, shape=(size,),
                                           dtype=h5py.special_dtype(vlen=str), compression='lzf')
            dataset_ref_dict[dataset_name] = dataset

        for key, value in data_dict.items():
            key = str_to_int(key)
            for dataset_name in datasets:
                dataset = dataset_ref_dict[dataset_name]
                try:
                    if isinstance(value[dataset_name], (list,tuple)):
                        dataset[key] = json.dumps(value[dataset_name])
                    else:
                        dataset[key] = value[dataset_name]
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"key: {key}, value: {value}, dataset_name: {dataset_name}")

    def save_as_npy(self, json_dict, name):
        np.save(name, np.array(json_dict))

    def main(self):
        self.parse_files()

        root = Path(self.args.output_hdf5).parent
        language = Path(self.args.output_hdf5).stem + "_labels"
        output_folder = (root / language)
        output_folder.mkdir(parents=True, exist_ok=True)

        logger.info("Saving labels as npy files...")
        self.save_as_npy(self.text_labels, output_folder / f"text_labels.npy")
        self.save_as_npy(self.ocr_labels, output_folder / f"ocr_labels.npy")
        self.save_as_npy(self.coco_labels, output_folder / f"coco_labels.npy")

        with h5py.File(self.args.output_hdf5, self.write_mode) as hf:
            if not "labels" in hf:
                label_group = hf.create_group("labels")
            else:
                label_group = hf["labels"]

            self.delete_existing_datasets(hf)

            logger.info("Saving TEXT labels as HDF5 datasets...")
            self.create_dataset_for_each_key_in_label_dict(label_group, self.text_labels)
            #self.create_dataset_for_each_key_in_label_dict(label_group, self.ocr_labels)
            #self.coco_dump(label_group, "coco", self.coco_labels)



if __name__ == "__main__":
    if True:
        "/media/data/1TB/datasets/synthetic/NEW_VERSION/latin /media/data/1TB/datasets/synthetic/NEW_VERSION/latin.h5  --img_count 5000000"
        args = "/media/data/1TB/datasets/synthetic/NEW_VERSION/latin"
        args += "/media/data/1TB/datasets/synthetic/NEW_VERSION/latin.h5"
        args += " --img_count 5000000"
        args += " --json_count 2"
        args += " --overwrite"
    else:
        args = fr"'G:\synthetic_data\one_line\french' french.hdf5"
        args += " --img_count 1000"
    if sys.argv[1:]:
        args = None

    hf = FindJONS(args)
    hf.main()

