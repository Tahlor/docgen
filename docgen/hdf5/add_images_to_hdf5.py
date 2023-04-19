import shlex
import argparse
import json
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np
from pathlib import Path
import re
import os
re_digit = re.compile(r'\D', re.IGNORECASE)
def str_to_int(s):
    return int(re_digit.sub('', s))

def to_gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

class HDF5Maker:
    def __init__(self, args=None):
        self.args = self.parse_args(args)
        self.file_count = 0
        self.memory_labels = {}
        self.coco_labels = {}
        self.write_mode = "w" if self.args.overwrite else "a"
        self.grayscale = self.args.grayscale
        self.img_count = self.args.count
        self.chunk_size = self.args.chunk_size

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("input_folder", help="Path to the folder containing the JSON files")
        parser.add_argument("output_hdf5", help="Path to the output HDF5 file")
        parser.add_argument("--max_images", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite the output HDF5 file if it already exists")
        parser.add_argument("--count", type=int, default=None, help="Total number of files to allocate space for in the HDF5 file.")
        parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for the HDF5 file.")

        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))
        print(f"Args: {args}")
        return args


    def main(self):

        with h5py.File(self.args.output_hdf5, "w") as hf:
            images = hf.create_group("images")

            img_file_count = 0
            for file in tqdm(os.listdir(self.args.input_folder)):
                if file.endswith(".jpg"):
                    img_path = os.path.join(self.args.input_folder, file)
                    img = Image.open(img_path)
                    img_data = np.array(img)
                    img_key = os.path.splitext(file)[0]
                    images.create_dataset(img_key, data=img_data, compression="gzip")
                    img_file_count += 1
                if self.args.max_files and img_file_count >= self.args.max_files:
                    break

    @staticmethod
    def get_next_image(path):
        for img in path.glob('*.jpg'):
            yield img

    def process_img(self, img_path):
        img = Image.open(img_path)
        idx = str_to_int(img_path.stem)
        img_data = np.array(img)
        if self.grayscale:
            img_data = to_gray(img_data)
        return idx, img_data

    def compute_img_size(self):
        _, sample_image = self.process_img(next(iter(self.get_next_image(Path(self.args.input_folder)))))
        return sample_image.shape

    def main2(self):
        img_shape = self.compute_img_size()
        with h5py.File(self.args.output_hdf5, 'w') as f:
            # Determine the shape of the images

            if self.img_count is None:
                all_images = list(self.get_next_image(Path(self.args.input_folder)))
                self.img_count = len(all_images)

            self.chunk_size = min(self.args.chunk_size, self.img_count)

            # Create datasets for images and metadata
            images = f.create_dataset('images', shape=(self.img_count, *img_shape),
                                      dtype=np.uint8, chunks=(self.chunk_size, *img_shape), compression='lzf')
            metadata = f.create_group('labels')

            for i, img_path in enumerate(self.get_next_image(Path(self.args.input_folder))):
                idx, img = self.process_img(img_path)
                images[idx] = img


if __name__ == "__main__":
    args = fr"'G:\synthetic_data\one_line\french' french.hdf5 --grayscale --overwrite"
    args += " --count 5000000"
    hf = HDF5Maker(args)
    hf.main2()
