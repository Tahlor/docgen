import json
import matplotlib.pyplot as plt
import numpy as np
import PIL
from pathlib import Path
from PIL import PpmImagePlugin, Image

def shape(item):
    """
    Args:
        item:
    Returns:
        x, y
    """
    if isinstance(item, np.ndarray):
        return item.shape[1],item.shape[0]
    elif isinstance(item, PpmImagePlugin.PpmImageFile) or isinstance(item, Image.Image):
        return item.size

def ndim(item):
    return len(shape(item))

def channels(item):
    if isinstance(item, np.ndarray):
        if ndim(item)==2:
            return 1
        elif ndim(item)==3:
            return item.shape[-1]
        else:
            raise Exception
    elif isinstance(item, PpmImagePlugin.PpmImageFile):
        if item.mode == "L":
            return 1
        elif item.mode == "RGB":
            return 3
        else:
            raise Exception

def _resize(im, nR=None, nC=None, square=True):
    """ no interpolation, don't use

    Args:
        im:
        nR:
        nC:
        square:

    Returns:

    """
    if square:
          if nR is None:
              if nC is None:
                return im
              else:
                nR = int(im.shape[0]*nC/im.shape[1])
          else:
              if nC is None:
                nC = int(im.shape[1]*nR/im.shape[0])
    else:
        nR = im.shape[0] if nR is None else nR
        nC = im.shape[1] if nC is None else nC

    nR0 = len(im)     # source number of rows
    nC0 = len(im[0])  # source number of columns
    return [[ im[int(nR0 * r / nR)][int(nC0 * c / nC)] for c in range(nC)] for r in range(nR)]


def display(img, cmap="gray"):
    # if isinstance(img, PpmImagePlugin.PpmImageFile) or isinstance(img, Image.Image):
    #     img.show()
    # else:
    #
    if isinstance(img,list):
        img = img[0]
    fig, ax = plt.subplots(figsize=(10, 10))
    if channels(img)==3:
        cmap = None
    ax.imshow(img, cmap=cmap)
    plt.show()


# def open_image(path):
#     with Path(path).open("rb") as f:
#         return Image.open(f)

def print_all(obj, reset=False):
    """ Try to recursively print an object and attributes in their entirety
        Call printall(obj, reset=True) at top-level
    Args:
        obj:

    Returns:

    """
    global DONE
    if reset:
        DONE = set()
    from collections.abc import Iterable
    if repr(obj) in DONE:
        return
    print("OBJECT", obj)
    try:
        print_all(obj.__dict__)
    except:
        if isinstance(obj, dict):
            for key,value in obj.items():
                print(key)
                print_all(value)
        elif isinstance(obj, Iterable):
            for i in obj:
                print_all(i)
        else:
            print(obj)
    DONE.add(repr(obj))


def incrementer(root, base, make_folder=True):
    new_folder = Path(root) / base
    increment = 0

    while new_folder.exists():
        increment += 1
        increment_string = f"{increment:02d}" if increment > 0 else ""
        new_folder = Path(root / (base + increment_string))

    new_folder.mkdir(parents=True, exist_ok=True)
    return new_folder

def file_incrementer(path, digits=4, create_dir=False, require_digits=True):
    original_path = path = Path(path)
    increment = 1

    if require_digits:
        path = Path(original_path.parent / (original_path.stem + f"_{increment:0{digits}d}" + original_path.suffix))

    while path.exists():
        increment += 1
        increment_string = f"_{increment:0{digits}d}" if increment > 0 else ""
        path = Path(original_path.parent / (original_path.stem + increment_string + original_path.suffix))

    if create_dir:
        path.mkdir(exist_ok=True, parents=True)
    return path

def save_image(img, img_path):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        # cv2.imwrite("filename.png", img)
    #elif isinstance(img, PpmImagePlugin.PpmImageFile) or isinstance(img, Image.Image):
    img.save(img_path)

if __name__ == '__main__':
    f = r"C:\Users\tarchibald\github\docx_localization\temp\french_census_0002\french_census_coco.json"
    with Path(f).open() as ff:
        d = json.load(ff)
    pass
