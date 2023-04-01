import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import random
from pathlib import Path

def pdf_to_png(pdf_file_path, output_path, dpi=200):
    """
    Converts a PDF file to PNG images, one image per page.

    Args:
        pdf_file_path (str): Path to the PDF file.
        output_path (str): Directory to save the PNG files.
        dpi (int): Dots per inch resolution of the PNG images.

    Returns:
        list of str: Paths to the generated PNG files.
    """
    output_path = Path(output_path)

    images = convert_from_path(pdf_file_path, dpi=dpi)
    if len(images) == 1:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        image = images[0]
        image.save(output_path, "PNG")
        image_paths = [output_path]
    else:
        image_paths = []
        output_path.mkdir(exist_ok=True, parents=True)
        for i, image in enumerate(images):
            image_path = f"{output_path}/page_{i + 1}.png"
            image.save(image_path, "PNG")
            image_paths.append(image_path)
    return image_paths


def crop_to_content(image_path, border=10):
    """
    Crops an image to the area containing content, adding a random border.

    Args:
        image_path (str): Path to the image file.
        border (int): Fixed border to add around the content (in pixels).

    Returns:
        PIL.Image.Image: Cropped image.
    """
    image = Image.open(image_path)
    # Convert image to numpy array
    image_np = np.array(image)

    # Detect content by finding where the pixels change
    rows = np.any(image_np != 255, axis=1)
    cols = np.any(image_np != 255, axis=0)

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Calculate random borders
    random_border = random.randint(0, border)

    # Adjust the crop area by adding random borders
    ymin = max(0, ymin - random_border)
    ymax = min(image.height, ymax + random_border)
    xmin = max(0, xmin - random_border)
    xmax = min(image.width, xmax + random_border)

    # Crop and return the image
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    return cropped_image

# Example usage of functions can be added here if necessary.
