import albumentations as A
import random
from albumentations import Compose, VerticalFlip, HorizontalFlip, Crop, RandomCrop, RandomRotate90
from albumentations.pytorch import ToTensorV2
import random
import cv2  # OpenCV, used for the crop operation
from albumentations import DualTransform, ImageOnlyTransform, NoOp, PadIfNeeded, RandomScale, LongestMaxSize, Rotate
import random
import random
import numpy as np
import torch
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor

class RandomResizeAlbumentations(ImageOnlyTransform):
    def __init__(self, min_scale=0.5, max_scale=2.0, min_pixels=None, max_upscale=2.0, always_apply=False, p=0.5):
        super(RandomResizeAlbumentations, self).__init__(always_apply, p)
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.min_pixels = min_pixels
        self.max_upscale = max_upscale

    def apply(self, img, **params):
        scale_factor = random.uniform(self.min_scale, self.max_scale)
        original_height, original_width = img.shape[:2]
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)

        if self.min_pixels:
            new_height = max(new_height, self.min_pixels)
            new_width = max(new_width, self.min_pixels)

        return cv2.resize(img, (new_width, new_height))


class RandomCropIfTooBigAlbumentations(ImageOnlyTransform):
    def __init__(self, size, always_apply=False, p=0.5):
        super(RandomCropIfTooBigAlbumentations, self).__init__(always_apply, p)
        self.size = size

    def apply(self, img, **params):
        height, width = img.shape[:2]
        crop_h, crop_w = self.size

        if height >= crop_h and width >= crop_w:
            x1 = random.randint(0, width - crop_w)
            y1 = random.randint(0, height - crop_h)
            return img[y1:y1 + crop_h, x1:x1 + crop_w]
        else:
            return img


class ResizeAndPadAlbumentations(ImageOnlyTransform):
    def __init__(self, longest_side, div, always_apply=False, p=0.5):
        super(ResizeAndPadAlbumentations, self).__init__(always_apply, p)
        self.longest_side = longest_side
        self.div = div

    def apply(self, img, **params):
        # Placeholder for resizing logic
        # Placeholder for padding logic
        return img


class PadToBeDivisibleByAlbumentations(ImageOnlyTransform):
    def __init__(self, div, always_apply=False, p=0.5):
        super(PadToBeDivisibleByAlbumentations, self).__init__(always_apply, p)
        self.div = div

    def apply(self, img, **params):
        height, width = img.shape[:2]
        pad_h = (self.div - height % self.div) % self.div
        pad_w = (self.div - width % self.div) % self.div

        if pad_h == 0 and pad_w == 0:
            return img
        else:
            return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[1, 1, 1])


# Remaining transformations such as 'IdentityTransform', 'ToTensorIfNeeded', 'RandomBottomLeftEdgeCrop' can follow the same pattern as above.

class ToTensorIfNeededAlbumentations(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(ToTensorIfNeededAlbumentations, self).__init__(always_apply, p)

    def apply(self, img, **params):
        if not torch.is_tensor(img):
            return ToTensor()(np.array(img))
        else:
            return img


class IdentityTransformAlbumentations(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(IdentityTransformAlbumentations, self).__init__(always_apply, p)

    def apply(self, img, **params):
        return img


class RandomBottomLeftEdgeCropAlbumentations(ImageOnlyTransform):
    def __init__(self, left_crop=37, bottom_crop=52, always_apply=True, p=0.5):
        super(RandomBottomLeftEdgeCropAlbumentations, self).__init__(always_apply, p)
        self.left_crop = left_crop
        self.bottom_crop = bottom_crop

    def apply(self, img, **params):
        if random.choice(['left', 'bottom']) == 'left':
            return img[:, self.left_crop:, :]
        else:
            return img[:-self.bottom_crop, :, :]


class RandomCropIfTooBig(DualTransform):
    def __init__(self, size, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.size = size

    def apply(self, img, **params):
        h, w = img.shape[:2]
        if h >= self.size[0] and w >= self.size[1]:
            x1 = random.randint(0, w - self.size[1])
            y1 = random.randint(0, h - self.size[0])
            return img[y1:y1+self.size[0], x1:x1+self.size[1]]

        return img

class RandomBottomLeftEdgeCrop(A.DualTransform):
    def __init__(self, left_crop=37, bottom_crop=52, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.left_crop = left_crop
        self.bottom_crop = bottom_crop

    def apply(self, img, **params):
        if random.choice(['left', 'bottom']) == 'left':
            return img[:, self.left_crop:, :]
        else:
            return img[:-self.bottom_crop, :, :]

    def get_transform_init_args_names(self):
        return ("left_crop", "bottom_crop")

class RandomBottomLeftEdgeCropSquare(A.DualTransform):
    def __init__(self, edge_crop_size=52, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.edge_crop = RandomBottomLeftEdgeCrop(left_crop=edge_crop_size, bottom_crop=edge_crop_size)

    def apply(self, img, **params):
        cropped_img = self.edge_crop.apply(img)
        min_dim, max_dim = sorted(cropped_img.shape[:2])
        offset = random.randint(0, max_dim - min_dim)
        return cv2.resize(cropped_img[offset:offset+min_dim, :min_dim], (min_dim, min_dim))

    def get_transform_init_args_names(self):
        return ("edge_crop_size",)



