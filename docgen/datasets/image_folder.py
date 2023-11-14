import numpy as np
import sys
from typing import Dict, Tuple
import math
from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationCompositionDataset
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import Compose
import logging
from docgen.transforms.transforms_torch import ResizeAndPad, ToTensorIfNeeded
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
import random
from docgen.datasets.generic_dataset import GenericDataset

class NaiveImageFolder(GenericDataset):
    def __init__(self, img_dir,
                 transform_list=None,
                 max_length=None,
                 color_scheme="RGB",
                 recursive=True,
                 extensions=(".jpg", ".png", ".jfif"),
                 return_format="just_image",
                 shuffle=True,
                 require_non_empty_result=False,
                 filters=None,
                 max_uniques=None,
                 max_length_override=None,
                 collate_fn=None,
                 **kwargs):
        """
        Common transforms:
        - By default it returns a PIL image
        - ToTensorIfNeeded() # converts to tensor if not already a tensor

        Args:
            img_dir:
            transforms:
            max_length:
            color_scheme:
            recursive:
            extensions:
            return_format: "dict" or "just_image"
            **kwargs:
        """
        if collate_fn is None:
            if return_format == "just_image":
                collate_fn = GenericDataset.tensor_collate_fn
            elif return_format == "dict":
                collate_fn = GenericDataset.dict_collate_fn

        super().__init__(
            max_uniques=max_uniques,
            max_length_override=max_length_override,
            transform_list=transform_list,
            collate_fn=collate_fn,
        )

        self.shuffle = shuffle
        self.require_non_empty_result = require_non_empty_result
        self.img_dirs = img_dir
        if not isinstance(img_dir, (tuple, list)):
            self.img_dirs = [img_dir]

        self.imgs = []
        extensions = set([ext.lower() for ext in extensions])
        for img_dir in self.img_dirs:
            img_dir = Path(img_dir)
            if recursive:
                self.imgs += list(img_dir.rglob("*.*"))
            else:
                self.imgs += list(img_dir.glob("*.*"))
        self.imgs = [img for img in self.imgs if img.suffix.lower() in extensions]
        self.filters = list(filters) if filters is not None else []
        self.failed_filters_count = 0

        if self.shuffle:
            random.shuffle(self.imgs)
        else:
            self.imgs = sorted(self.imgs)

        self.color_scheme = color_scheme

        if len(self.imgs) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.max_length = max_length if max_length is not None else len(self.imgs)
        self.current_img_idx = 0
        self.return_format = return_format

    def __len__(self):
        return min(len(self.imgs), self.max_length)

    def _get(self, idx):
        empty_results = 0
        while True:
            try:
                idx = self._validate_idx(idx)
                img_path = self.imgs[idx]

                # load from png and convert to tensor
                img = Image.open(str(img_path)).convert(self.color_scheme)
                if self.transform_composition.transforms is not None:
                    img = self.transform_composition(img)

                if self.filters and self.reject_because_of_filter(img):
                    idx += 1
                    continue

                if self.return_format == "just_image":
                    return img
                elif self.return_format == "dict":
                    return {'image': img,  "name": img_path.stem}
                else:
                    raise NotImplementedError(f"return_format {self.return_format} not implemented")

            except:
                logger.exception(f"Error loading image {img_path}")
                idx += 1

    def reject_because_of_filter(self, img):
        for filter in self.filters:
            if filter(img):
                self.failed_filters_count += 1
                if self.failed_filters_count and self.failed_filters_count % 5 == 0:
                    logger.warning(f"Filters failed {self.failed_filters_count} times (current filter: {filter})")
                return True
        self.failed_filters_count = 0
        return False

    def get(self, idx=None):
        if idx is None:
            idx = self.current_img_idx
            self.current_img_idx = (1 + self.current_img_idx ) % len(self.imgs)
        return self._get(idx)

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:

        """
        return self.get(idx)


    @staticmethod
    def collate_fn(batch):
        return SemanticSegmentationCompositionDataset.collate_fn(batch)


class NaiveImageFolderPatch(NaiveImageFolder):
    def __init__(self, img_dir,
                 patch_size: Tuple[int, int],
                 transform_list=None,
                 max_length=None,
                 color_scheme="RGB",
                 longest_side=None,
                 pad_to_be_divisible_by=32,
                 max_uniques=None,
                 max_length_override=None,
                 **kwargs):
        """ Safer as an iterator, might work as a indexable dataset, but length changes dynamically

        Args:
            img_dir:
            patch_size:
            transform_list:
            max_length:
            color_scheme:
            longest_side:
            pad_to_be_divisible_by:
            **kwargs:
        """
        super().__init__(
            img_dir=img_dir,
            transform_list=[ResizeAndPad( longest_side=448, pad_to_be_divisible_by=pad_to_be_divisible_by)],
            color_scheme=color_scheme,
            max_uniques=None,
            max_length_override=None,
            **kwargs)
        self.patch_size = patch_size
        self.patch_idx = 0
        self.current_img_patches = None

        img_count = len(self.imgs)
        self.imgs_idx_list = list(range(0, img_count)) # shuffle this if you want to shuffle the images
        self.current_img_idx = 0
        self.img_and_patch_idx = 0
        self.current_size = max_length if max_length else img_count * 50

    def __len__(self):
        return self.current_size

    def _get_patches(self, img):
        # Calculate the number of patches needed
        if isinstance(img, torch.Tensor):
            img.width = img.shape[-1]
            img.height = img.shape[-2]
        patch_column_count = math.ceil(img.width / self.patch_size[0])
        patch_row_count = math.ceil(img.height / self.patch_size[1])

        # Calculate the overlap needed
        overlap_w = max(0, (patch_column_count * self.patch_size[0] - img.width) // (patch_column_count - 1) if patch_column_count > 1 else 0)
        overlap_h = max(0, (patch_row_count * self.patch_size[1] - img.height) // (patch_row_count - 1) if patch_row_count > 1 else 0)

        patches = []
        coords = []

        for i in range(patch_row_count):
            for j in range(patch_column_count):

                left = max(0, j * self.patch_size[0] - j * overlap_w)
                upper = max(0, i * self.patch_size[1] - i * overlap_h)
                right = min(img.width+1, left + self.patch_size[0])
                lower = min(img.height+1, upper + self.patch_size[1])

                # If this is the last patch, adjust the coordinates to be relative to the bottom right corner of the image
                if j == patch_column_count - 1 and patch_column_count > 1:
                    left = img.width - self.patch_size[0]
                    right = img.width
                if i == patch_row_count - 1 and patch_row_count > 1:
                    upper = img.height - self.patch_size[1]
                    lower = img.height

                if isinstance(img, Image.Image):
                    patch = img.crop((left, upper, right, lower))
                elif torch.is_tensor(img):
                    # Convert coordinates to PyTorch style (C, H, W)
                    left, upper, right, lower = map(int, [left, upper, right, lower])
                    patch = img[:, upper:lower, left:right]
                else:
                    raise TypeError("img should be a PIL Image or a PyTorch tensor.")

                # Save the patch and its top left corner coordinate
                patches.append(patch)
                coords.append((left, upper))

        return patches, coords, patch_column_count, patch_row_count

    def _get(self, idx):
        """

        Args:
            idx: Just ignore this value, since we are iterating over the images and don't know how many patches there will be

        Returns:

        """
        while True:
            try:

                # If we have patches from the previous image, return the next one
                if self.current_img_patches is not None and self.patch_idx < len(self.current_img_patches):
                    patch = self.current_img_patches[self.patch_idx]
                    coord = self.current_img_coords[self.patch_idx]

                    # if self.transform_composition.transforms is not None:
                    #     patch = self.transform_composition(patch)

                    patch_row, patch_column = self.patch_idx // self.patch_column_count, self.patch_idx % self.patch_column_count
                    name = f"{self.current_img_path.stem}_{patch_row}-{patch_column}_{'-'.join([str(c) for c in coord])}"

                    out =  {'image': patch,
                            "name": name,
                            "patch_coordinate": (patch_row, patch_column),
                            "abs_coordinate": coord}

                    last_patch = self.patch_idx + 1 == len(self.current_img_patches)
                    last_img = self.current_img_idx + 1 == len(self.imgs)
                    if last_img and last_patch:
                        self.current_size = self.img_and_patch_idx + 1

                    self.patch_idx += 1
                    self.img_and_patch_idx += 1

                    return out

                # Load from png and convert to tensor
                idx = self.imgs_idx_list[self.current_img_idx % len(self.imgs)]
                self.current_img_path = self.imgs[idx]
                self.current_img = Image.open(self.current_img_path).convert(self.color_scheme)


                if self.transform_composition is not None:
                    self.current_img = self.transform_composition(self.current_img)

                self.current_img_patches, self.current_img_coords, self.patch_column_count, self.patch_row_count = self._get_patches(self.current_img)
                self.patch_idx = 0
                self.current_img_idx += 1

            except:
                logger.exception(f"Error loading image {self.current_img_path}")
                self.current_img_idx += 1
                self.current_img_patches = None
                self.patch_idx = 0

    @staticmethod
    def collate_fn(batch):
        return SemanticSegmentationCompositionDataset.collate_fn(batch)

    def __iter__(self):
        self.current_img_idx = 0
        self.patch_idx = 0
        self.img_and_patch_idx = 0
        self.current_img_patches = None
        return self

    def __next__(self):
        # If there is no more data to return, raise StopIteration
        if self.current_img_idx >= len(self.imgs_idx_list):
            raise StopIteration

        # Use your existing _get method to get the next value
        value = self._get(self.current_img_idx)

        # If _get didn't find a valid patch, raise StopIteration
        if value is None:
            raise StopIteration

        return value


