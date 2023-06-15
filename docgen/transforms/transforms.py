import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision import transforms


class ResizeAndPad:
    def __init__(self, resize, div):
        """

        Args:
            resize: The side of the longest side will be resized to this
            div: The other side will be padded to be divisible by this
        """
        self.resize = ResizeLongestSide(resize)
        self.div = div

    def __call__(self, img):
        img = self.resize(img)
        img = pad_divisible_by(img, self.div)
        return img

def pad_divisible_by(image, pad_divisible_by=32):
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
        self.size = size

    def __call__(self, image):

        h, w = image.shape[-2:]
        if h > w:
            new_h = self.size
            new_w = int(self.size * w / h)
        else:
            new_w = self.size
            new_h = int(self.size * h / w)

        return transforms.Resize((new_h, new_w))(image)

class IdentityTransform:
    """A transform that does nothing"""
    def __call__(self, x):
        return x
