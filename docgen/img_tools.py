from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image, ImageDraw, ImageFilter

from pathlib import Path

def convert_pdf_bytes_to_img_bytes(pdf):
    images = convert_from_bytes(pdf)
    return images


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
