"""
Reject certain images if conditions aren't met
"""
import numpy as np
import logging
from hwgen.data.utils import show

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Reject:
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name

    def __call__(self, img, name, *args, **kwargs):
        raise NotImplementedError

class RejectIfEmpty(Reject):

    def __call__(self, img, name=None, **kwargs):
        if not np.asarray(img).any():
            logger.debug("Rejecting image because it is empty")
            return True
        return False

class RejectIfTooManyPixelsAreBelowThreshold(Reject):

        def __init__(self, threshold=0.6,
                     max_percent_of_pixels_allowed_below_threshold=0.4,
                     dataset_name=None):
            super().__init__(dataset_name)
            self.threshold_brightness = threshold
            self.percent_of_pixels = max_percent_of_pixels_allowed_below_threshold

        def __call__(self, img, name=None, **kwargs):
            img = np.asarray(img)
            if len(img.shape) == 3:
                img = img.mean(axis=0)
            dark_pixel_percent = np.sum(img < self.threshold_brightness) / img.size
            if dark_pixel_percent > self.percent_of_pixels:
                show(img)
                logger.debug(f"{self.dataset_name} Rejecting image because too many pixels are below threshold")
                logger.debug(f"Dark pixel percent: {dark_pixel_percent}")
                if name is not None:
                    logger.debug(f"Image name: {name}")

                return True
            return False
