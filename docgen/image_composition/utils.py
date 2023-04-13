import random
from docgen.bbox import BBox

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