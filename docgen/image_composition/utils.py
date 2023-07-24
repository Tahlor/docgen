import numpy as np
import random
from docgen.bbox import BBox
from typing import Union

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

class CompositeImagesDeprecated:
    """ Composite two images together, given a minimum overlap percentage
    Example usage:
        composite = CompositeImages()
        composite(base_image, image2, min_overlap=.5, image_format='HWC', composite_function=np.minimum, overlap_with_respect_to="paste_image")

        OR to calculate the origin:
        composite.calculate_origin(base_image, image2, min_overlap=.5)

    """

    def __init__(self, image_format='HWC', composite_function=np.minimum,
                 overlap_with_respect_to:Union["base_image","paste_image"]="base_image"):
        self.params_last_used = None
        self.image_format = image_format
        self.composite_function = composite_function
        self.overlap_with_respect_to = overlap_with_respect_to


    def calculate_origin(self, base_image, image2, min_overlap):
        if self.image_format == 'CHW':
            x, y = 2, 1
        elif self.image_format == 'HWC':
            x, y = 1, 0
        elif self.image_format.upper() == 'PIL':
            x, y = 0, 1
            base_image.shape = base_image.size
            image2.shape = image2.size
        else:
            raise ValueError(f"Unsupported image format: {self.image_format}")

        height1, width1 = base_image.shape[y], base_image.shape[x]
        height2, width2 = image2.shape[y], image2.shape[x]

        target_overlap_percent = random.uniform(min_overlap, 1.0)

        overlap_width_baseline, overlap_height_baseline = (width1, height1) if self.overlap_with_respect_to == "base_image" \
            else (width2, height2)

        # Calculate the minimum overlap in pixels
        overlap_pixel_target = target_overlap_percent * overlap_width_baseline * overlap_height_baseline

        # Image 2 is smaller than the overlap target in at least 1 dimension
        if min(width2, width1) * min(height2, height1) < overlap_pixel_target:
            extreme_x = base_image.shape[x] - image2.shape[x]  # if this is negative, then this is the min and 0 the max
            extreme_y = base_image.shape[y] - image2.shape[y]

            min_start_x, max_start_x, min_start_y, max_start_y = convert_most_extreme_value_to_min_max(extreme_x,
                                                                                                       extreme_y)
            # Calculate random start positions within the calculated limits
            origin_x = random.randint(min_start_x, max_start_x)
            origin_y = random.randint(min_start_y, max_start_y)

            width_overlap = min(width2, width1)
            height_overlap = min(height2, height1)

        else:
            # Calculate the minimum width required to meet the overlap requirement
            max_width = min(width2, width1)
            max_height = min(height2, height1)

            min_width = overlap_pixel_target // max_height
            width_overlap = random.randint(min_width, max_width)

            height_overlap = overlap_pixel_target // width_overlap

            width_overlap = int(min(width_overlap + 1, width2, width1))
            height_overlap = int(min(height_overlap + 1, height2, height1))
            assert width_overlap * height_overlap >= overlap_pixel_target

            # Choose a random origin ensuring the chosen width and height will be within base_image
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

            origin_x = get_valid_origin(width_overlap, width1, width2)
            origin_y = get_valid_origin(height_overlap, height1, height2)

        self.params_last_used = {
            'origin_x': origin_x,
            'origin_y': origin_y,
            'width_overlap': width_overlap,
            'height_overlap': height_overlap,
            'overlap_pixel_target': overlap_pixel_target,
            'target_overlap_percent': target_overlap_percent,
            'width1': width1,
            'height1': height1,
            'width2': width2,
            'height2': height2,
        }
        return origin_x, origin_y

    def __call__(self, base_image, image2, min_overlap=0.5):
        origin_x, origin_y = self.calculate_origin( base_image, image2, min_overlap)
        width1, height1, width2, height2 = self.params_last_used['width1'], self.params_last_used['height1'], \
                                              self.params_last_used['width2'], self.params_last_used['height2']

        # Paste image2 onto base_image at the chosen origin
        result_image = base_image.copy()
        overlay_image = image2[
                               max(0, -origin_y):min(height2, height1 - origin_y),
                               max(0, -origin_x):min(width2, width1 - origin_x)]

        overlap = result_image[max(0, origin_y):min(height1, origin_y + height2),
                               max(0, origin_x):min(width1, origin_x + width2)]

        result_image[max(0, origin_y):min(height1, origin_y + height2),
        max(0, origin_x):min(width1, origin_x + width2)] = self.composite_function(overlap, overlay_image)
        return result_image


