"""
Reject certain images if conditions aren't met
"""
import numpy as np
import logging
from hwgen.data.utils import show

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RejectIfEmpty():

    def __init__(self):
        pass
    def __call__(self, img):
        if not np.asarray(img).any():
            logger.debug("Rejecting image because it is empty")
            return True
        return False

class RejectIfTooManyPixelsAreBelowThreshold():

        def __init__(self, threshold_brightness=0.6, percent_of_pixels=0.4):
            self.threshold_brightness = threshold_brightness
            self.percent_of_pixels = percent_of_pixels

        def __call__(self, img):
            img = np.asarray(img)
            dark_pixel_percent = np.sum(img < self.threshold_brightness) / img.size
            logger.debug(f"Dark pixel percent: {dark_pixel_percent}")
            show(img)
            if dark_pixel_percent > self.percent_of_pixels:
                logger.debug("Rejecting image because too many pixels are below threshold")
                return True
            return False
