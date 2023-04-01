from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageDraw, ImageFilter

from pathlib import Path

def convert_pdf_bytes_to_img_bytes(pdf):
    images = convert_from_bytes(pdf)
    return images

def crop_image(image_tensor, bbox, format="CHW"):
    x_min, y_min, x_max, y_max = bbox
    if format == "CHW":
        return image_tensor[:, y_min:y_max, x_min:x_max]
    elif format == "HWC":
        return image_tensor[y_min:y_max, x_min:x_max, :]
    else:
        raise ValueError(f"Unknown format: {format}")
def replace_region(original_image, bbox, cropped_image, format="CHW"):
    x_min, y_min, x_max, y_max = bbox
    if format == "CHW":
        original_image[:, y_min:y_max, x_min:x_max] = cropped_image
    elif format == "HWC":
        original_image[y_min:y_max, x_min:x_max, :] = cropped_image
    else:
        raise ValueError(f"Unknown format: {format}")
    return original_image

def paste_image(backgroud_as_bytes, ):
    back_im = backgroud_as_bytes.copy()
    back_im.paste(im2, (100, 50))

    return back_im

# Convert to image
def convert_pdf_to_img_paths(pdf_temp, img_path):
    # from docgen.image_deprecated import pdf2image
    # func = pdf2image.convert

    func = convert_from_path

    pages = func(pdf_temp)
    for i in range(len(pages)):
        print(i)
        path = Path(img_path.format(i))
        pages[i].save(path, 'JPEG')
