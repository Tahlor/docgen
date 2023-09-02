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
from pathlib import Path
from docgen.image_composition.utils import CalculateImageOriginForCompositing
import albumentations as A

class CompositeImages(Gen):
    def __init__(self, folders, transforms=None, default_transform=None):
        super().__init__()

        self.folders = folders
        self.default_transform = default_transform if default_transform else self._default_transform

        if transforms is None:
            self.transforms = [self.default_transform]
        else:
            self.transforms = transforms

        self.images = self._load_images()
        self.calculate_origin = CalculateImageOriginForCompositing(image_format='HWC')
        self.default_composite_function = seamless_composite

    def _load_images(self):
        """Load images from folders"""
        images = []
        for folder in self.folders:
            # use rglob
            for file_path in Path(folder).rglob("*"):
                if file_path.is_file() and file_path.suffix in [".png", ".jpg", ".jpeg"]:
                    img = Image.open(file_path)
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


    def composite(self, base_image, image2, min_overlap=.5, image_format='HWC',
                        composite_function=None, overlap_with_respect_to="paste_image"):
        """
        Composite two images together with given parameters.

        Composite Functions:
            np.minimum
            cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)
        """
        if composite_function is None:
            composite_function = self.default_composite_function

        # Convert PIL images to numpy arrays
        base_image_np = np.array(base_image)
        image2_np = np.array(image2)

        # Get the origin
        origin_x, origin_y = self.calculate_origin(base_image_np, image2_np, min_overlap)
        result = composite_function(base_image_np, image2_np, origin_x, origin_y)

        # Convert back to PIL image
        result_pil = Image.fromarray(result.astype(np.uint8))

        return result_pil, (origin_x, origin_y)


def numpy_composite(base_image, overlay_image, origin_x, origin_y, composite_function=np.minimum):
    width1, height1, width2, height2 = base_image.shape[1], base_image.shape[0], overlay_image.shape[1], overlay_image.shape[0]

    result_image = base_image.copy()
    overlay_image = overlay_image[
                    max(0, -origin_y):min(height2, height1 - origin_y),
                    max(0, -origin_x):min(width2, width1 - origin_x)]

    overlap = result_image[max(0, origin_y):min(height1, origin_y + height2),
              max(0, origin_x):min(width1, origin_x + width2)]

    result_image[max(0, origin_y):min(height1, origin_y + height2),
    max(0, origin_x):min(width1, origin_x + width2)] = composite_function(overlap, overlay_image)
    return result_image

def seamless_composite(base_image, overlay_image, origin_x, origin_y, *args, **kwargs):
    # Assume that the images are single or three-channel 8-bit images
    # Create a binary mask from the alpha channel of image2
    if overlay_image.shape[2] == 4:  # RGBA image
        mask = overlay_image[..., 3]
    else:  # RGB or grayscale
        mask = np.ones_like(overlay_image[..., 0]) * 255

    # Calculate center coordinates based on the origin and size of image2
    center = (origin_y + overlay_image.shape[1] // 2, origin_x + overlay_image.shape[0] // 2)

    # Apply seamless cloning
    result = cv2.seamlessClone(src=overlay_image, dst=base_image, mask=mask, p=center,
                               flags=cv2.NORMAL_CLONE)
    return result

def demo_image_composition(img1_path, img2_path):
    # Load images using PIL
    # img1 = Image.open(img1_path)
    # img2 = Image.open(img2_path)

    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))

    # Create a CompositeImages object
    folders = [str(Path(img1_path).parent), str(Path(img2_path).parent)]
    compositor = CompositeImages(folders)

    # Test naive_composite
    result_naive, _ = compositor.composite(img1, img2, composite_function=numpy_composite)
    result_naive.show()
    result_naive.save('result_naive_composite.png')

    # Test composite with default seamless cloning
    result_seamless, _ = compositor.composite(img1, img2, composite_function=seamless_composite)
    result_seamless.show()
    result_seamless.save('result_seamless_composite.png')

if __name__ == '__main__':
    #scanned_doc_path = "G:\s3\forms\HTSNet_scanned_documents"
    scanned_doc_path = Path("G:/s3/forms/HTSNet_scanned_documents")
    seal = Path("G:/data/seals/train_images")
    # pick random image from scanned_doc_path
    img1_path = choice(list(scanned_doc_path.glob("*")))
    # pick random image from seal
    img2_path = choice(list(seal.glob("*")))

    demo_image_composition(img1_path, img2_path)
