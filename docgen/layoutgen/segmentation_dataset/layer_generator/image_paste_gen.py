import numpy as np
from PIL import Image
from random import choice
import cv2
from sklearn.utils import shuffle
from docgen.layoutgen.writing_generators import Gen
from pathlib import Path
from docgen.image_composition.utils import CalculateImageOriginForCompositing
from docgen.image_composition.utils import seamless_composite, composite_the_images_numpy

class CompositeImages(Gen):
    """ If you have a bunch of e.g., text datasets, you could produce the composited versions using this object
    """
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
    result_naive, _ = compositor.composite(img1, img2, composite_function=composite_the_images_numpy)
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
