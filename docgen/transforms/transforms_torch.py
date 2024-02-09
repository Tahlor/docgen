import torchvision.transforms as T
import warnings

import torch
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
from torchvision import transforms
import torch
from torchvision.transforms import ToTensor
import albumentations as A

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
import numpy as np
from PIL import Image, ImageOps, PpmImagePlugin

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

        original_height, original_width = img.shape[-2:]

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
        self.crop_height = self.size[0]
        self.crop_width = self.size[1]
        self.random_crop = transforms.RandomCrop(size)
        self.params = None

    def __call__(self, image):
        if not isinstance(image, torch.Tensor):
            raise Exception("Must be a tensor")

    def crop_img(self, image, reuse_params=False):
        height = image.shape[-2]
        width = image.shape[-1]

        # if it's big enough, crop
        if height >= self.crop_height and width >= self.crop_width:
            if self.params is None or not reuse_params:
                self.params = self.random_crop.get_params(image, (self.crop_height, self.crop_width))
            return F.crop(image, *self.params)
        # if only one side is big enough, just do one
        elif height >= self.crop_height:
            return F.crop(image, 0, 0, self.crop_height, width)
        elif width >= self.crop_width:
            return F.crop(image, 0, 0, height, self.crop_width)
        # if neither side is big enough, just return it
        else:
            return image

class ToTensorWithLabel(transforms.ToTensor):
    def __call__(self, image, label):
        return super().__call__(image), super().__call__(label)


class RandomCropIfTooBigWithLabel(RandomCropIfTooBig):
    def __init__(self, size: tuple):
        super().__init__(size)

    def __call__(self, image, label):
        if not isinstance(image, torch.Tensor):
            raise Exception("Must be a tensor")
        if not isinstance(label, torch.Tensor):
            raise Exception("Must be a tensor")
        image = self.crop_img(image, reuse_params=False)
        label = self.crop_img(label, reuse_params=True)
        return image, label



class ResizeAndPad:
    def __init__(self, longest_side, div):
        """

        Args:
            longest_side:
            div:
        """
        self.resize = ResizeLongestSide(longest_side) if longest_side else None
        self.div = div

    def __call__(self, img):
        if self.resize:
            img = self.resize(img)
        img = pad_divisible_by(img, self.div)
        return img

class PadToBeDivisibleBy:
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

class PadToBeDivisibleByWithLabel:
    def __init__(self, div):
        """

        Args:
            resize: The side of the longest side will be resized to this
            div: The other side will be padded to be divisible by this
        """
        self.div = div

    def __call__(self, image, label):
        image = pad_divisible_by(image, self.div)
        label = pad_divisible_by(label, self.div)
        return image, label

def pad_divisible_by(image, pad_divisible_by=32):
    if image is None:
        return None
    if isinstance(image, Image.Image):
        warnings.warn("Cannot pad PIL")

    # Calculate padding
    h, w = image.shape[-2:]
    h_pad = (pad_divisible_by - h % pad_divisible_by) % pad_divisible_by
    w_pad = (pad_divisible_by - w % pad_divisible_by) % pad_divisible_by

    if h_pad == 0 and w_pad == 0:
        return image
    else:
        x1_pad = round(w_pad / 2)
        y1_pad = round(h_pad / 2)
        x2_pad = w_pad - x1_pad
        y2_pad = h_pad - y1_pad
        if isinstance(image, np.ndarray):
            t = A.PadIfNeeded(min_height=y1_pad+y2_pad+h, min_width=x1_pad+x2_pad+w, border_mode=0, value=1)
            image = t(image=image)
        else:
            padding = transforms.Pad((x1_pad, y1_pad, x2_pad, y2_pad), fill=1)
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

    def __init__(self, longest_side=448):
        """ Must have 3 channels

        Args:
            size:
        """
        self.size = longest_side
        self.resize = ResizeLongestSide(self.size)

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

class RandomBottomLeftEdgeCrop:
    def __init__(self, left_crop=37, bottom_crop=52):
        self.left_crop = left_crop
        self.bottom_crop = bottom_crop

    def __call__(self, image, **params):
        if random.choice(['left', 'bottom']) == 'left':
            return image[:, :, self.left_crop:]
        else:
            return image[:, :-self.bottom_crop, :]

class RandomBottomLeftEdgeCropSquare:
    def __init__(self, edge_crop_size=52):
        self.RandomBottomLeftEdgeCrop = RandomBottomLeftEdgeCrop(left_crop=edge_crop_size, bottom_crop=edge_crop_size)

    def __call__(self, image):
        cropped_img = self.RandomBottomLeftEdgeCrop(image)
        min_dim, max_dim = sorted(cropped_img.shape[1:3])
        offset = random.randint(0, max_dim - min_dim)
        cropped_img = F.crop(cropped_img, offset, 0, min_dim, min_dim)
        return cropped_img

class CropBorder:
    def __init__(self, border_size_rng):
        self.border_size = border_size_rng

    def sample_border_size(self):
        if isinstance(self.border_size, int):
            return self.border_size
        else:
            return random.randint(self.border_size[0], self.border_size[1])
    def __call__(self, img):
        """

        Args:
            img:
            crop: (top left corner) y, x, height, width
        Returns:

        """
        border_size = self.sample_border_size()
        img_cropped = F.crop(img, border_size, border_size, img.shape[-2]-border_size*2, img.shape[-1]-border_size*2)
        return img_cropped

class RandomScaleResize(transforms.Resize):
    def __init__(self, min_scale=0.5, max_scale=3, interpolation=Image.BILINEAR):
        super(RandomScaleResize, self).__init__(size=0, interpolation=interpolation)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img):
        scale = random.uniform(self.min_scale, self.max_scale)
        new_size = [int(dim * scale) for dim in img.size]
        return super(RandomScaleResize, self).resize(img, new_size)

class RandomFlipOrMirror:
    def __init__(self):
        # No parameters needed for this augmentation
        pass

    def __call__(self, image):
        action = random.choice(['flip', 'mirror', 'none', 'both'])

        if action == 'flip':
            # Flip 180 degrees
            image = np.rot90(image, 2)
        elif action == 'mirror':
            # Mirror (horizontal flip)
            image = np.fliplr(image)
        elif action == 'both':
            # Both flip and mirror
            image = np.rot90(image, 2)
            image = np.fliplr(image)
        # 'none' does nothing

        return image

class Mirror:
    def __init__(self):
        # No parameters needed for this augmentation
        pass

    def __call__(self, image):
        return np.ascontiguousarray(np.fliplr(image))


class OneOfTransform(T.Compose):
    def __init__(self, transforms, **kwargs):
        super().__init__(transforms)

    def __call__(self, img):
        transform = random.choice(self.transforms)
        return transform(img)


class Invert:
    def __init__(self):
        pass

    def __call__(self, img):
        #if dtype
        if isinstance(img, torch.Tensor):
            if img.dtype == torch.uint8:
                return 255 - img
            else:
                return 1 - img
        elif isinstance(img, np.ndarray):
            if img.dtype == np.uint8:
                return 255 - img
            else:
                return 1 - img
        elif isinstance(img, Image.Image):
            return ImageOps.invert(img)
