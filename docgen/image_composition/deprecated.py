import torch
from typing import Callable

@staticmethod
def composite_the_images_torch2(background_img, img, bckg_x, bckg_y):
    # Shape of the composite_image
    bckg_h, bckg_w = background_img.shape[-2:]
    img_h, img_w = img.shape[-2:]

    # if background_img has 2 dimensions and img has 3, expand the background
    if len(background_img.shape) == 2 and len(img.shape) == 3:
        background_img = background_img.unsqueeze(0)

    # if img has more channels than background, expand the background
    if img.shape[0] > background_img.shape[0] and len(background_img.shape) == 3 == len(img.shape):
        background_img = background_img.expand(img.shape[0], -1, -1)
        # assert torch.all(background_img[0] == background_img[1])

    # Calculate the coordinates of the paste area
    paste_x = bckg_x
    paste_y = bckg_y
    paste_width = min(img_w, bckg_w - bckg_x if bckg_x >= 0 else img_w + bckg_x, bckg_w)
    paste_height = min(img_h, bckg_h - bckg_y if bckg_y >= 0 else img_h + bckg_y, bckg_h)

    # Check for overlap
    if paste_width <= 0 or paste_height <= 0:
        print("No overlap between the images.")
        return background_img

    # Paste the overlap onto the background image
    background_img = background_img.clone()

    slice_background = background_img[..., paste_y:paste_y + paste_height, paste_x:paste_x + paste_width]
    slice_img = img[..., :paste_height, :paste_width]

    # Ensure the slices are of the same shape
    if slice_background.shape == slice_img.shape:
        background_img[..., paste_y:paste_y + paste_height, paste_x:paste_x + paste_width] = torch.min(
            slice_background, slice_img)
    else:
        print("Shapes do not match: slice_background = {}, slice_img = {}".format(slice_background.shape,
                                                                                  slice_img.shape))

    # Return the composite image
    return background_img


