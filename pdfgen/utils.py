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

def coco_dataset(dict_list, output_path):
    """
        # coco info, images
        # info
            {
            'description': 'COCO 2014 Dataset',
            'url': 'http://cocodataset.org',
            'version': '1.0',
            'year': 2014,
            'contributor': 'COCO Consortium',
            'date_created': '2017/09/01'}
        # images - list
            # {'license': 2, 'file_name': 'COCO_test2014_000000523573.jpg', 'coco_url': 'http://images.cocodataset.org/test2014/COCO_test2014_000000523573.jpg', 'height': 500, 'width': 423, 'date_captured': '2013-11-14 12:21:59', 'id': 523573}
            # {'license': 4, 'file_name': '000000397133.jpg', 'coco_url': 'http://images.cocodataset.org/val2017/000000397133.jpg', 'height': 427, 'width': 640, 'date_captured': '2013-11-14 17:02:52', 'flickr_url': 'http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg', 'id': 397133}
        # licences
        # categories
            # [{'supercategory': 'person', 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}, {'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'id': 90, 'name': 'toothbrush'}]
        # annotations - list
            {'image_id': 179765, 'id': 38, 'caption': 'A black Honda motorcycle parked in front of a garage.'}
        # get handwriting to work

        [{localization, image, image_path, height, width}]
    """
    annotations = []  # image_id, bbox, text, category
    categories = []  # list of all types of fields
    images = []  # id, filename, height, width

    for image_dict in dict_list:
        file = Path(image_dict["image_path"]).name
        id = Path(image_dict["image_path"]).stem
        images.append({"id":id, "filename": file, "height": image_dict["height"], "width": image_dict["width"]})

        for key in image_dict["localization"].keys():
            if "localization_" in key:
                localization_level = key.split("_")[-1]
                for box in image_dict["localization"][key]:
                    annotations.append({"image_id": id,
                                        "bbox":box["bbox"],
                                        "text":box["text"],
                                        "category": f"{localization_level}"
                    })

    info = {'description': 'Synthetic Forms - Pre-alpha Release',
            'url': 'N/A',
            'version': '0.0.0.1',
            'year': 2022,
            'contributor': 'Taylor Archibald',
            'date_created': '2022/07/25'}

    coco={"info":info,
          "images":images,
          "licences": "No license, internal work product, confidential",
          "categories":categories,
          "annotations":annotations
    }

    with output_path.open("w") as ff:
        json.dump(coco,ff)

if __name__ == '__main__':
    f = r"C:\Users\tarchibald\github\docx_localization\temp\french_census_0002\french_census_coco.json"
    with Path(f).open() as ff:
        d = json.load(ff)
    pass
