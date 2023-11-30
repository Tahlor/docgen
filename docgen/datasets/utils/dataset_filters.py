"""
Reject certain images if conditions aren't met
"""
import numpy as np
import logging
from hwgen.data.utils import show

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Reject:

    def __call__(self, img, name, *args, **kwargs):
        raise NotImplementedError

class RejectIfEmpty(Reject):

    def __init__(self):
        pass
    def __call__(self, img, name=None, **kwargs):
        if not np.asarray(img).any():
            logger.debug("Rejecting image because it is empty")
            return True
        return False

class RejectIfTooManyPixelsAreBelowThreshold(Reject):

        def __init__(self, threshold_brightness=0.6, max_percent_of_pixels_allowed_below_threshold=0.4):
            self.threshold_brightness = threshold_brightness
            self.percent_of_pixels = max_percent_of_pixels_allowed_below_threshold

        def __call__(self, img, name=None, **kwargs):
            img = np.asarray(img)
            dark_pixel_percent = np.sum(img < self.threshold_brightness) / img.size
            logger.debug(f"Dark pixel percent: {dark_pixel_percent}")
            if dark_pixel_percent > self.percent_of_pixels:
                show(img)
                logger.debug("Rejecting image because too many pixels are below threshold")
                logger.debug(f"Dark pixel percent: {dark_pixel_percent}")
                if name is not None:
                    logger.debug(f"Image name: {name}")

                return True
            return False
