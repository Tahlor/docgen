from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import h5py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse


class ToCSV:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.args = self.parse_args(args)
        self.compression = "lzf"
        self.write_mode = "w" if self.args.overwrite else "a"

    def parse_args(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument("npy_folder", help="Path to the folder containing the JSON files")
        parser.add_argument("output_hdf5", help="Path to the output HDF5 file")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite the output HDF5 file if it already exists")

        if args is None:
            args = parser.parse_args()
        else:
            args = parser.parse_args(shlex.split(args))
        return args

