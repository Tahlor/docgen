from albumentations import ImageOnlyTransform
import numpy as np


class AlbumentationsWrapper(object):
    """
    A wrapper to adapt albumentations transforms to work with or without labels within
    torchvision's Compose. Handles images and optionally labels.
    """
    def __init__(self, transform, *args, **kwargs):
        self.transform = transform(*args, **kwargs)
        self._to_dict = transform(*args, **kwargs)._to_dict
        self.__repr__ = transform(*args, **kwargs).__repr__
        self.add_targets = lambda *args, **kwargs: None

    def __call__(self, image, label=None):
        if label is not None:
            transformed = self.transform(image=image, label=label)
            image = transformed['image']
            label = transformed['label']
        else:
            transformed = self.transform(image=image)
            image = transformed['image']
        return (image, label) if label is not None else image


class PILToNumpyTransform(ImageOnlyTransform):
    """An albumentations transform to convert PIL images to NumPy arrays."""

    def __init__(self, always_apply=True, p=1.0):
        #super(PILToNumpyTransform, self).__init__(always_apply, p)
        self.always_apply = always_apply
        self.p = p

    def apply(self, image, **params):
        output = {"image": np.array(image)}
        for k in params:
            output[k] = np.array(params[k])
        return output

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)