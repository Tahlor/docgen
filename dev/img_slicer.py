import tifffile as tiff
from pathlib import Path
import numpy as np


def slice_images(source_dir, output_dir):
    """
    Slices all PNG and TIFF images in a source directory into four smaller images of 224x224 each.

    Args:
        source_dir (str): The source directory containing the images to slice.
        output_dir (str): The directory where the sliced images will be saved.
    """
    source_dir_path = Path(source_dir)
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    for image_path in source_dir_path.glob('*'):
        if image_path.suffix.lower() in ['.png', '.tiff', '.tif']:
            # Use tifffile for TIFF images and PIL.Image for PNG
            if image_path.suffix.lower() in ['.tiff', '.tif']:
                image = tiff.imread(image_path)
                # Handle multi-channel TIFF images
                if image.ndim == 3:  # Assuming channels are in the first dimension
                    height, width, _ = image.shape
                else:
                    raise ValueError("Unsupported image dimension for TIFF.")
            else:  # Fallback for PNG or other Pillow-supported formats
                from PIL import Image
                image = np.array(Image.open(image_path))
                height, width = image.shape[:2]

            # Assuming the images are at least 448x448, slicing into 224x224
            for i in range(2):
                for j in range(2):
                    left = i * 224
                    top = j * 224
                    # Crop the image, taking into account possible multiple channels
                    cropped_image = image[top:top + 224, left:left + 224]

                    # Construct the output filename
                    output_filename = f"{image_path.stem}_{i * 2 + j + 1}{image_path.suffix}"
                    # Save using tifffile for TIFF images or PIL.Image for PNG
                    if image_path.suffix.lower() in ['.tiff', '.tif']:
                        tiff.imwrite(str(output_dir_path / output_filename), cropped_image)
                    else:  # Fallback for PNG or other formats supported by PIL
                        from PIL import Image
                        Image.fromarray(cropped_image).save(output_dir_path / output_filename)


if __name__ == "__main__":
    import argparse
    import shlex


    def parser(args=None):
        parser = argparse.ArgumentParser()
        parser.add_argument("--source_dir", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        if args is not  None:
            args = parser.parse_args(shlex.split(args))
        else:
            args = parser.parse_args()
        return args


    args = parser()
    slice_images(source_dir=args.source_dir, output_dir=args.output_dir)
