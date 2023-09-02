import numpy as np
import random
from docgen.bbox import BBox
from typing import Union, Tuple, Dict, Any
from pathlib import Path
from random import choice

def pick_origin(background_shape,
                factor=4):
    """

    Args:
        background_shape: (int,int)
        factor (int): Origin must be less than 1/factor of total space

    Returns:

    """
    return random.randint(0, int(background_shape[0] / factor)), \
    random.randint(0, int(background_shape[1] / factor))

def max_size_given_origin_and_background(origin, background_shape, vertical_buffer_factor=1.):
    """

    Args:
        origin: (int,int)
        background_shape: (int,int)
        vertical_buffer_factor float(0,1): .5 means that the textbox can only be half the height of the background
    Returns:

    """
    return int((background_shape[0] - origin[0])), int((background_shape[1] - origin[1])*vertical_buffer_factor)

def new_textbox_given_background(background_shape,
                                 origin_factor=4,
                                 end_point_factor=2,
                                 font_size=32,
                                 minimum_width_percent=.75):
    """

    Args:
        origin (int,int): y,x
        background_shp: (int,int)
        origin_factor: Origin must be less than 1/factor of total space
        end_point_factor: Textbox must use 1/factor of available space after origin chosen

    Returns:

    """
    min_x = int(background_shape[1] * minimum_width_percent)

    # mostly for picking where to start in a large document
    origin = pick_origin(background_shape, factor=origin_factor)

    # if doing lines, start is highly constrained
    # if font_size > background_shape[0]:
    #     font_size = int(background_shape[0] * random.uniform(.8,.9))
    # if font_size+origin[0]>background_shape[0]:
    #     y = int((background_shape[0]-font_size) * random.uniform(0,1))
    #     origin = (y, origin[1])
    # if minimum_width_percent * background_shape[1] < background_shape[1]-origin[1]:
    #     x = int(background_shape[1] * random.uniform(minimum_width_percent,1))
    #     origin = (origin[0], x)

    max_size = max_size_given_origin_and_background(origin, background_shape)
    return [random.randint(int(max_size[0] / end_point_factor), max_size[0]), \
           random.randint(int(max_size[1] / end_point_factor), max_size[1])], \
           origin

def new_textbox_given_background_line(background_shape,
                                 font_size=32,
                                 minimum_width_percent=.75):
    if font_size > background_shape[1] * .95:
        font_size = int(background_shape[1] * random.uniform(.8,.95))

    y1_max = int(background_shape[1] - font_size)
    x2_min = int(background_shape[0] * minimum_width_percent)
    origin = random.randint(0, background_shape[0] - x2_min), random.randint(0, y1_max)
    #endpoint = (background_shape[0], random.randint(x2_min, background_shape[1]))
    size = max_size_given_origin_and_background(origin, background_shape, vertical_buffer_factor=1)
    bbox = BBox("ul", [origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]], format="XYXY", font_size=font_size)
    return size, origin, font_size, bbox


def convert_most_extreme_value_to_min_max(extreme_x, extreme_y):
    min_start_x = min(0, extreme_x)  # if image2 is wider than image1, then the min start position can be neg
    max_start_x = max(0, extreme_x)  # if image2 is wider than image1, then the max start position is 0
    min_start_y = min(0, extreme_y)  # if image2 is taller than image1, then the min start position can be neg
    max_start_y = max(0, extreme_y)  # if image2 is taller than image1, then the max start position is 0
    return min_start_x, max_start_x, min_start_y, max_start_y

def get_valid_origin(dim_overlap, dim1, dim2):
    part_of_image2_outside_image1 = dim2 > dim_overlap
    both_sides_of_image2_outside_image1 = dim_overlap == dim1
    assert dim_overlap <= dim1
    if part_of_image2_outside_image1 and not both_sides_of_image2_outside_image1:
        neg_excess_dim = abs(dim2 - dim_overlap)
        pos_excess_dim = dim1 - dim_overlap
        origin = random.choice([-neg_excess_dim, pos_excess_dim])
    elif both_sides_of_image2_outside_image1:
        assert dim2 >= dim_overlap
        excess = abs(dim2 - dim_overlap)
        origin = random.randint(-excess, 0)
    else:  # completely inside image1
        assert dim1 >= dim_overlap
        origin = random.randint(0, dim1 - dim_overlap)
    return origin

def compute_paste_origin_from_overlap(height1: int, width1: int, height2: int, width2: int, min_overlap: float,
                                      overlap_with_respect_to: str ="base_image") -> Tuple[int, int]:
    """Calculate the origin to composite two images given their sizes and a minimum overlap.
       Image1 is assumed to be the background image; the coordinates will be where Image2 is pasted onto Image1.
       Overlap is calculated with respect to the base image (Image1) by default, but can be calculated with respect
         to the overlay image (Image2) by setting overlap_with_respect_to to "overlay_image".
    """
    target_overlap_percent = random.uniform(min_overlap, 1.0)
    if overlap_with_respect_to == "base_image":
        overlap_width_baseline, overlap_height_baseline = (width1, height1)
    elif overlap_with_respect_to == "overlay_image":
        overlap_width_baseline, overlap_height_baseline = (width2, height2)
    else:
        raise ValueError(f"overlap_with_respect_to must be either 'base_image' or 'overlay_image', but got {overlap_with_respect_to}")

    overlap_pixel_target = target_overlap_percent * overlap_width_baseline * overlap_height_baseline

    if min(width2, width1) * min(height2, height1) < overlap_pixel_target:
        extreme_x = width1 - width2
        extreme_y = height1 - height2
        min_start_x, max_start_x = (0, extreme_x) if extreme_x >= 0 else (extreme_x, 0)
        min_start_y, max_start_y = (0, extreme_y) if extreme_y >= 0 else (extreme_y, 0)
        origin_x = random.randint(min_start_x, max_start_x)
        origin_y = random.randint(min_start_y, max_start_y)
        width_overlap = min(width2, width1)
        height_overlap = min(height2, height1)

    else:
        max_width = min(width2, width1)
        max_height = min(height2, height1)
        min_width = overlap_pixel_target // max_height
        width_overlap = random.randint(min_width, max_width)
        height_overlap = overlap_pixel_target // width_overlap
        width_overlap = int(min(width_overlap + 1, width2, width1))
        height_overlap = int(min(height_overlap + 1, height2, height1))

        origin_x = get_valid_origin(width_overlap, width1, width2)
        origin_y = get_valid_origin(height_overlap, height1, height2)

    actual_overlap = width_overlap * height_overlap

    params_last_used = {
        'origin_x': origin_x,
        'origin_y': origin_y,
        'width_overlap': width_overlap,
        'height_overlap': height_overlap,
        'overlap_pixel_target': overlap_pixel_target,
        'target_overlap_percent': target_overlap_percent,
        'actual_overlap': actual_overlap,
        'actual_overlap_percent': actual_overlap / (width1 * height1),
        'width1': width1,
        'height1': height1,
        'width2': width2,
        'height2': height2,
    }

    return origin_x, origin_y, params_last_used

def compute_paste_origin_from_overlap_auto(base_image,
                                           overlay_image,
                                           min_overlap: float,
                                           image_format: str,
                                           overlap_with_respect_to: str="base_image") -> Tuple[int, int, Dict[str, Any]]:

    """Wrapper function to handle different image formats (CHW, HWC, PIL)"""
    if image_format == 'CHW':
        x, y = 2, 1
    elif image_format == 'HWC':
        x, y = 1, 0
    elif image_format.upper() == 'PIL':
        x, y = 0, 1
        base_image.shape = base_image.size
        overlay_image.shape = overlay_image.size
    else:
        raise ValueError(f"Unsupported image format: {image_format}")

    height1, width1 = base_image.shape[y], base_image.shape[x]
    height2, width2 = overlay_image.shape[y], overlay_image.shape[x]

    return compute_paste_origin_from_overlap(height1, width1, height2, width2, min_overlap, overlap_with_respect_to=overlap_with_respect_to)


class CalculateImageOriginForCompositing:
    """ Do the compositing locally
        But still use the functions above to compute origin?

        Composite two images together, given a minimum overlap percentage
    Example usage:
        composite = CompositeImages()
        composite(base_image, image2, min_overlap=.5, image_format='HWC', composite_function=np.minimum, overlap_with_respect_to="paste_image")

        OR to calculate the origin:
        composite.calculate_origin(base_image, image2, min_overlap=.5)

    """

    def __init__(self,
                 image_format='HWC',
                 overlap_with_respect_to: str="base_image",
                 ):
        self.params_last_used = None
        self.image_format = image_format
        self.overlap_with_respect_to = overlap_with_respect_to

    def __call__(self, base_image, image2, min_overlap=0.5):
        return self.calculate_origin(base_image, image2, min_overlap)

    def calculate_origin(self, base_image, image2, min_overlap):
        origin_x, origin_y, self.params_last_used = compute_paste_origin_from_overlap_auto(base_image,
                                                                                           image2,
                                                                                           min_overlap,
                                                                                           self.image_format,
                                                                                           self.overlap_with_respect_to)
        return origin_x, origin_y

def np_composite(base_image, overlay_image, origin_x, origin_y, function=np.minimum):
    height1, width1 = base_image.shape[0], base_image.shape[1]
    height2, width2 = overlay_image.shape[0], overlay_image.shape[1]

    result_image = base_image.copy()
    overlay_image = overlay_image[
                    max(0, -origin_y):min(height2, height1 - origin_y),
                    max(0, -origin_x):min(width2, width1 - origin_x)]

    overlap = result_image[max(0, origin_y):min(height1, origin_y + height2),
              max(0, origin_x):min(width1, origin_x + width2)]

    result_image[max(0, origin_y):min(height1, origin_y + height2),
    max(0, origin_x):min(width1, origin_x + width2)] = function(overlap, overlay_image)
    return result_image


class CompositeImages(CalculateImageOriginForCompositing):
    def __init__(self, image_format='HWC', composite_function=np.minimum,
                    overlap_with_respect_to:Union["base_image","paste_image"]="base_image"):
        super().__init__(image_format, composite_function, overlap_with_respect_to)
        self.composite_function = composite_function

    def __call__(self, base_image, overlay_image, min_overlap=0.5):
        origin_x, origin_y = self.calculate_origin(base_image, overlay_image, min_overlap)
        result_image = np_composite(base_image, overlay_image, origin_x, origin_y, self.composite_function)
        return result_image


if __name__ == '__main__':
    import cv2
    scanned_doc_path = Path("G:/s3/forms/HTSNet_scanned_documents")
    seal = Path("G:/data/seals/train_images")
    seal = scanned_doc_path
    img1_path = choice(list(scanned_doc_path.glob("*")))
    img2_path = choice(list(seal.glob("*")))
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    x,y,_ = compute_paste_origin_from_overlap_auto(img1, img2, min_overlap=.98, image_format='HWC')
    print(img1.shape, img2.shape)
    print(x,y)
