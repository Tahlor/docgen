import sys
from docgen.bbox import BBox
from docgen.render_doc import composite_images_PIL
from PIL import Image

import os
import numpy as np
from PIL import Image
from random import choice
import cv2
from sklearn.utils import shuffle
from docgen.layoutgen.segmentation_dataset.gen import Gen

class CompositeImages(Gen):
    def __init__(self, folders, transforms=None, default_transform=None):
        super().__init__()

        self.folders = folders
        if transforms is None:
            self.transforms = [self.default_transform]
        else:
            self.transforms = transforms
        self.default_transform = default_transform if default_transform else self._default_transform
        self.images = self._load_images()

    def _load_images(self):
        """Load images from folders"""
        images = []
        for folder in self.folders:
            for filename in os.listdir(folder):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img = Image.open(os.path.join(folder, filename))
                    images.append(img)
        return shuffle(images)

    def _default_transform(self, img):
        """Default transformation (resize/rotation/mirror)"""
        # Resize
        max_size = (500, 500)
        img.thumbnail(max_size, Image.ANTIALIAS)
        # Random rotation
        angle = np.random.uniform(-30, 30)
        img = img.rotate(angle)
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img = Image.fromarray(np.fliplr(np.array(img)))
        return img

    def get(self):
        """Get a randomly transformed image"""
        img = choice(self.images)
        transform = choice(self.transforms)
        return transform(img)

    def composite(self, base_image, image2, min_overlap=.5, image_format='HWC', composite_function=np.minimum, overlap_with_respect_to="paste_image"):
        """Composite two images together with given parameters"""
        # Convert PIL images to numpy arrays
        base_image = np.array(base_image)
        image2 = np.array(image2)
        # Get the origin
        origin = self.calculate_origin(base_image, image2, min_overlap)
        # Apply the composite function
        result = composite_function(base_image, image2)
        # Convert back to PIL image
        result = Image.fromarray(result.astype(np.uint8))
        return result, origin

    def calculate_origin(self, base_image, image2, min_overlap=.5):
        """Calculate the origin for compositing images"""
        # Here you can implement your own function for calculating the origin based on image shapes and overlap
        # This is a dummy implementation that just returns (0,0)
        return (0, 0)
