from PIL import Image
import numpy as np

class AutoCropper:
    def __init__(self, crop_color=None, crop_mode='both'):
        """
        Initialize the AutoCropper.

        Args:
            image: A PIL Image or NumPy array.
            crop_color: Color to crop out. If None, finds maximum brightness color.
            crop_mode: 'horizontal', 'vertical', or 'both'.
        """
        self.crop_color = crop_color
        self.crop_mode = crop_mode

    def _max_brightness_color(self):
        """ Find the color with maximum brightness in the image. """
        # Convert to grayscale to find maximum brightness
        grayscale = np.mean(self.image, axis=-1)
        return
    def _find_bounding_box(self):
        """ Find the bounding box that excludes the crop color. """

        crop_color = self._max_brightness_color() if self.crop_color is None else self.crop_color

        mask = np.all(self.image != crop_color, axis=-1)
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        if self.crop_mode == 'vertical':
            x_min, x_max = 0, self.image.shape[1]
        elif self.crop_mode == 'horizontal':
            y_min, y_max = 0, self.image.shape[0]

        return x_min, y_min, x_max + 1, y_max + 1

    def process_image(self, image):
        if isinstance(image, Image.Image):
            self.image = np.array(image)
        elif isinstance(image, np.ndarray):
            self.image = image
        else:
            raise TypeError("Image must be a PIL Image or NumPy array")


    def crop(self, image):
        self.process_image(image)
        """ Crop the image based on the specified crop color and mode. """
        x_min, y_min, x_max, y_max = self._find_bounding_box()
        cropped_image = self.image[y_min:y_max, x_min:x_max]
        return Image.fromarray(cropped_image)
