import warnings

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision import transforms
import torch
from torchvision.transforms import ToTensor


class RandomCropIfTooBig:
    def __init__(self, size):
        self.size = size
        self.random_crop = transforms.RandomCrop(size)
        self.totensor = ToTensor()


    def __call__(self, img):
        img = self.totensor(img)
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
        self.totensor = ToTensor()

    def __call__(self, img):
        img = self.totensor(img)
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

    def __init__(self, size=448):
        """ Must have 3 channels

        Args:
            size:
        """
        self.size = size
        self.to_tensor = ToTensor()

    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.to_tensor(image)

        h, w = image.shape[-2:]
        if h > w:
            new_h = self.size
            new_w = int(self.size * w / h)
        else:
            new_w = self.size
            new_h = int(self.size * h / w)

        return transforms.Resize((new_h, new_w))(image)

class ResizeLongestSideIfTooBig:

    def __init__(self, size=448):
        """ Must have 3 channels

        Args:
            size:
        """
        self.size = size
        self.to_tensor = ToTensor()
        self.resize = ResizeLongestSide(size)

    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            image = self.to_tensor(image)

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