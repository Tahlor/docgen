import random

def pick_origin(background_shape,
                factor=4):
    """

    Args:
        background_shape: (int,int)
        factor (int): Origin must be less than 1/factor of total space

    Returns:

    """
    return random.randint(0, int(background_shape[0] / 4)), \
    random.randint(0, int(background_shape[1] / 4))

def max_size_given_origin_and_background(origin, background_shape):
    """

    Args:
        origin: (int,int)
        background_shape: (int,int)
    Returns:

    """
    return background_shape[0] - origin[0], background_shape[1] - origin[1]

def new_textbox_given_background(background_shape,
                                 origin_factor=4,
                                 end_point_factor=2):
    """

    Args:
        origin: (int,int)
        background_shp: (int,int)
        origin_factor: Origin must be less than 1/factor of total space
        end_point_factor: Textbox must use 1/factor of available space after origin chosen

    Returns:

    """
    origin = pick_origin(background_shape, factor=origin_factor)
    max_size = max_size_given_origin_and_background(origin, background_shape)
    return [random.randint(int(max_size[0] / end_point_factor), max_size[0]), \
           random.randint(int(max_size[1] / end_point_factor), max_size[1])], \
           origin
