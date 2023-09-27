import warnings

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision import transforms
import torch
from torchvision.transforms import ToTensor

# target size = 448
# for most things:
    # randomly resize 1/2 to 2x
    # crop, pad if necessary to 448
# for backgrounds: randomly resize from between 448 to 2x
    # crop, pad if necessary to 448

from torchvision.transforms import functional as F
from torchvision.transforms import Resize
import random
from albumentations import ImageOnlyTransform
import cv2
import random

class RandomResize:
    def __init__(self, min_scale: float, max_scale: float,
                 min_pixels: int = None, max_upscale: float = 2.0):
        """
        Randomly resizes an image.

        Args:
            min_scale (float): Minimum scale factor.
            max_scale (float): Maximum scale factor.
            min_pixels (int): If specified, ensures the minimum side length.
            max_upscale (float): Maximum upscale factor, to avoid excessive enlargement
                                 Only used IF minimum side length is specified and a huge upscale is required to get there
        """
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_pixels = min_pixels
        self.max_upscale = max_upscale

    def __call__(self, img):
        # Calculate random scale factor
        scale_factor = random.uniform(self.min_scale, self.max_scale)

        original_width, original_height = img.shape[-2:]

        # Compute new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)

        # Ensure minimum side length, if specified
        if self.min_pixels:
            if new_width < self.min_pixels:
                new_width = self.min_pixels
            if new_height < self.min_pixels:
                new_height = self.min_pixels

        # Ensure we don't upscale too much
        upscale_factor_w = new_width / original_width
        upscale_factor_h = new_height / original_height

        if upscale_factor_w > self.max_upscale:
            new_width = int(original_width * self.max_upscale)
        if upscale_factor_h > self.max_upscale:
            new_height = int(original_height * self.max_upscale)

        # Apply the resize
        return F.resize(img, (new_height, new_width))

class RandomCropIfTooBig:
    def __init__(self, size: tuple):
        self.size = size
        if isinstance(size, int):
            self.size = (size, size)
        self.random_crop = transforms.RandomCrop(size)

    def __call__(self, img):
        if not isinstance(img, torch.Tensor):
            raise Exception("Must be a tensor")

        # if it's big enough, crop
        if img.shape[-2] >= self.size[0] and img.shape[-1] >= self.size[1]:
            return self.random_crop(img)
        # if only one side is big enough, just do one
        elif img.shape[-2] >= self.size[0]:
            return F.crop(img, 0, 0, self.size[0], img.shape[-1])
        elif img.shape[-1] >= self.size[1]:
            return F.crop(img, 0, 0, img.shape[-2], self.size[1])
        # if neither side is big enough, just return it
        else:
            return img


class ResizeAndPad:
    def __init__(self, longest_side, div):
        self.resize = ResizeLongestSide(longest_side) if longest_side else None
        self.div = div

    def __call__(self, img):
        if self.resize:
            img = self.resize(img)
        img = pad_divisible_by(img, self.div)
        return img

class PadToBeDvisibleBy:
    def __init__(self, div):
        """

        Args:
            resize: The side of the longest side will be resized to this
            div: The other side will be padded to be divisible by this
        """
        self.div = div

    def __call__(self, img):
        img = pad_divisible_by(img, self.div)
        return img

def pad_divisible_by(image, pad_divisible_by=32):
    if isinstance(image, Image.Image):
        warnings.warn("Cannot pad PIL")

    # Calculate padding
    h, w = image.shape[-2:]
    h_pad = (pad_divisible_by - h % pad_divisible_by) % pad_divisible_by
    w_pad = (pad_divisible_by - w % pad_divisible_by) % pad_divisible_by

    if h_pad == 0 and w_pad == 0:
        return image
    else:
        padding = transforms.Pad((w_pad // 2, h_pad // 2, w_pad - w_pad // 2, h_pad - h_pad // 2), fill=1)
        image = padding(image)

    return image

class ResizeLongestSide:

    def __init__(self, longest_side=448):
        """ Must have 3 channels, MUST BE A TENSOR

        Args:
            longest_side:
        """
        self.size = longest_side

    def get_size(self):
        if isinstance(self.size, int):
            return self.size
        else:
            return random.randint(self.size[0], self.size[1])

    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            #image = self.to_tensor(image)
            raise Exception("Must be a tensor")

        h, w = image.shape[-2:]
        new_longest_side = self.get_size()
        if h > w:
            new_h = new_longest_side
            new_w = int(new_longest_side * w / h)
        else:
            new_w = new_longest_side
            new_h = int(new_longest_side * h / w)

        return transforms.Resize((new_h, new_w))(image)

class ResizeLongestSideIfTooBig:

    def __init__(self, size=448):
        """ Must have 3 channels

        Args:
            size:
        """
        self.size = size
        self.resize = ResizeLongestSide(size)

    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            #image = self.to_tensor(image)
            raise Exception("Must be a tensor")


        h, w = image.shape[-2:]
        if h > self.size or w > self.size:
            return self.resize(image)
        return image


class IdentityTransform:
    """A transform that does nothing"""
    def __call__(self, x):
        return x

class ToTensorIfNeeded:
    def __init__(self):
        self.totensor = ToTensor()
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            return self.totensor(x)
        else:
            return x

class RandomEdgeCrop:
    def __init__(self, left_crop=37, bottom_crop=52):
        self.left_crop = left_crop
        self.bottom_crop = bottom_crop

    def __call__(self, image, **params):
        if random.choice(['left', 'bottom']) == 'left':
            return image[:, :, self.left_crop:]
        else:
            return image[:, :-self.bottom_crop, :]

class RandomScaleResize(transforms.Resize):
    def __init__(self, min_scale=0.5, max_scale=3, interpolation=Image.BILINEAR):
        super(RandomScaleResize, self).__init__(size=0, interpolation=interpolation)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = [int(dim * scale) for dim in img.size]
        return super(RandomScaleResize, self).resize(img, new_size)
