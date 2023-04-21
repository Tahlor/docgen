import sys
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
import warnings
import socket

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
        self.enable_process_img = self.args.process_img
        if self.grayscale and not self.enable_process_img:
            warnings.warn("Cannot convert to grayscale without processing images")
        self.img_count = self.args.img_count
        self.chunk_size = self.args.chunk_size
        self.compression = self.args.compression
        self.enable_one_dataset = self.args.one_dataset
        if self.enable_process_img and not self.enable_one_dataset:
            warnings.warn("Storing processed images in ONE dataset")

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("input_folder", help="Path to the folder containing the JSON files")
        parser.add_argument("output_hdf5", help="Path to the output HDF5 file")
        parser.add_argument("--max_images", type=int, default=None, help="Maximum number of files to process")
        parser.add_argument("--process_img", action="store_true", help="Process to numpy")
        parser.add_argument("--grayscale", action="store_true", help="Convert images to grayscale")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite the output HDF5 file if it already exists")
        parser.add_argument("--img_count", type=int, default=None, help="Total number of files to allocate space for in the HDF5 file.")
        parser.add_argument("--chunk_size", type=int, default=1024, help="Chunk size for the HDF5 file.")
        parser.add_argument("--compression", type=str, default=None, help="Compression for the HDF5 file.")
        parser.add_argument("--one_dataset", action="store_true", help="Store all images in one dataset.")

        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))
        print(f"Args: {args}")
        return args

    @staticmethod
    def get_next_image(path):
        for img in path.glob('*.jpg'):
            yield img

    def process_img_to_numpy(self, img_path):
        img = Image.open(img_path)
        return self.transform(img, grayscale=self.grayscale)

    def read_img_as_binary(self, img_path):
        with open(img_path, 'rb') as img_f:
            binary_data = img_f.read()  # read the image as python binary
            #binary_data_np = np.asarray(binary_data)
        return binary_data

    def load_img(self, img_path):
        idx = str_to_int(img_path.stem)
        if self.enable_process_img:
            img_data = self.process_img_to_numpy(img_path)
        else:
            img_data = self.read_img_as_binary(img_path)
        return idx, img_data

    @staticmethod
    def transform(img, grayscale=True):
        img_data = np.array(img)
        if grayscale:
            img_data = to_gray(img_data)
        return img_data

    def compute_img_size(self):
        _, sample_image = self.load_img(next(iter(self.get_next_image(Path(self.args.input_folder)))))
        return sample_image.shape

    def pytorch_dataset(self, folder):
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
        from torchvision.transforms import ToTensor, Resize, Normalize, Compose
        transforms = Compose(HDF5Maker.transform)
        dataset = ImageFolder(folder, transform=transforms)
        for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16):
            yield batch


    def add_processed_images_to_hdf5(self, f):
        img_shape = self.compute_img_size()

        if self.img_count is None:
            all_images = list(self.get_next_image(Path(self.args.input_folder)))
            self.img_count = len(all_images)

        self.chunk_size = min(self.args.chunk_size, self.img_count)

        # Create datasets for images and metadata
        if not 'images' in f and self.enable_process_img:
            images = f.create_dataset('images', shape=(self.img_count, *img_shape),
                                      dtype=np.uint8, chunks=(self.chunk_size, *img_shape), compression=self.compression)
        else:
            images = f['images']

        for i, img_path in tqdm(enumerate(self.get_next_image(Path(self.args.input_folder)))):
            idx, img = self.load_img(img_path)
            images[idx] = img

    def add_raw_images_to_hdf5_each_as_dataset(self, f):
        chunk_size=2500000
        #chunk_size=100
        if "images" in f:
            if isinstance(f["images"], h5py.Dataset):
                print("Deleting old images dataset")
                del f['images']
            else:
                print("Deleting old images group")
                del f['images']

        if not 'images' in f:
            images = f.create_group('images')
        else:
            images = f['images']

        other_groups = []
        for i, img_path in tqdm(enumerate(self.get_next_image(Path(self.args.input_folder)))):
            idx, img = self.load_img(img_path)
            if str(idx) in images:
                del images[str(idx)]
            images.create_dataset(str(idx), data=np.array(img), compression=self.compression)

            if i % chunk_size == 0 and i > 0:
                group_idx = i // chunk_size + 1
                key_name = f'images{group_idx}'
                if key_name in f:
                    del f[key_name]

                images = f.create_group(key_name)
                other_groups.append((images, key_name))

        for other_group, other_group_name in other_groups:
            print(f"Updating {other_group_name}")
            f['images'].update(other_group)
            print(f"Deleting {other_group_name}")
            del f[other_group_name]


    def add_raw_images_to_hdf5(self, f):
        """ The problem right now is the NULL CHAR in the byte code, VLEN doesn't work with it
            You might try replacing it and unreplacing it or something

        Args:
            f:

        Returns:

        """
        if "images" in f and not isinstance(f["images"], h5py.Dataset):
            print("Deleting old images dataset")
            del f['images']

        if not 'images' in f:
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            images = f.create_dataset("images", shape=(self.img_count,), dtype=dt)
        else:
            images = f['images']

        for i, img_path in tqdm(enumerate(self.get_next_image(Path(self.args.input_folder)))):
            idx, img = self.load_img(img_path)
            images[idx] = np.fromstring(img, dtype='uint8')


    def main(self):
        with h5py.File(self.args.output_hdf5, self.write_mode) as f:
            if self.enable_process_img:
                self.add_processed_images_to_hdf5(f)
            else:
                if self.enable_one_dataset:
                    self.add_raw_images_to_hdf5(f)
                else:
                    self.add_raw_images_to_hdf5_each_as_dataset(f)

            print("Total images:")
            print(len(f["images"]))
            print(f.keys())


def run():
    if socket.gethostname().lower() != "galois":
        args = fr"'G:\synthetic_data\one_line\french' french.hdf5"
        args += " --img_count 1000"
    else:
        args = fr"'/media/data/1TB/datasets/synthetic/NEW_VERSION/latin' '/media/data/1TB/datasets/synthetic/NEW_VERSION/latin.hdf5' "
        args += " --img_count 5000000"

    if sys.argv[1:]:
        args = None

    hf = HDF5Maker(args)
    hf.main()


def use_dataloader_for_multiprocessing(folder):
    """ Attempt to use the DatasetFolder class to load the images"""
    """ JUST read the images in as binary"""
    from torchvision.datasets import ImageFolder, DatasetFolder, VisionDataset
    from torch.utils.data import DataLoader
    from torchvision.transforms import ToTensor, Resize, Normalize, Compose
    from classless_dataset import UnlabeledImageDataset

    transforms = Compose([HDF5Maker.transform])
    dataset = UnlabeledImageDataset(folder, transform=transforms)
    for batch in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6):
        for i in range(len(batch[0])):
            yield batch["image"][i], batch["idx"][i]

def test_multiprocessing():
    folder = rf"G:\synthetic_data\one_line\french"
    x = list(use_dataloader_for_multiprocessing(folder))

if __name__ == "__main__":
    run()

# PRIMARY:
# Create npy file Class
    # Generate CSV from NPY
# DatasetFolder: does this take too long to parse??? maybe just tell it how many images there are
# RSYNC the other files

# TODO: add images to HDF5 in multithreaded fashion

