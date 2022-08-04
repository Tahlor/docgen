from tqdm import tqdm
import time
import numpy as np
import json
from pathlib import Path
import os
import datetime
import copy
from PIL import Image
from pdfgen.utils import *
from pdfgen.bbox import BBox

DEFAULT_COCO_INFO = {'description': 'Synthetic Forms - Pre-alpha Release',
            'url': 'N/A',
            'version': '0.0.0.1',
            'year': 2022,
            'contributor': 'Taylor Archibald',
            'date_created': datetime.datetime.now().strftime("%D")}


def load_json(path):
    with Path(path).open() as ff:
        ocr = json.load(ff)
    return ocr

def save_json(path, arr):
    with Path(path).open("w") as ff:
        json.dump(arr, ff)

def numpy_to_json(path):
    arr = np.load(path, allow_pickle=True)
    save_json(Path(path).with_suffix(".json"), arr)

def delete_extra_images(path):
    if Path(path).stem != "OCR":
        folder = path
        path = path / "OCR.json"
    else:
        folder = Path(path).parent

    ocr_dict = load_json(path)

    for file in folder.rglob("*.jpg"):
        if file.stem not in ocr_dict:
            os.remove(file)
            #time.sleep(0.01)

def ocr_to_coco(ocr_dict, data_set_name="Synthetic Forms - Pre-alpha Release"):
    if isinstance(ocr_dict, Path) or isinstance(ocr_dict, str):
        ocr_dict = load_json(path)

    images = []
    annotations = []

    def process_item(dict, img_idx, category):
        item = {
            "image_id": img_idx,
            "bbox": dict["bbox"],
            "category": f"{category}"
        }
        if "text" in dict:
            item["text"] = dict["text"]
        return item

    for img_id, dict in ocr_dict.items():
        image = {"id":img_id, "filename": img_id+".jpg", "height": dict["height"], "width": dict["width"]}
        images.append(image)

        for section in dict["sections"]:
            for paragraph in section["paragraphs"]:
                for line in paragraph["lines"]:
                    for word in line["words"]:
                        annotations.append(process_item(word, img_id, "word"))
                    annotations.append(process_item(line, img_id, "line"))
                annotations.append(process_item(paragraph, img_id, "paragraph"))
            annotations.append(process_item(section, img_id, "section"))

    info = DEFAULT_COCO_INFO.copy()
    info["description"] = data_set_name

    coco = {
          "info": info,
          "images": images,
          "licences": "No license, internal work product, confidential",
          "categories": ["section","paragraph", "line", "word"],
          "annotations": annotations
    }
    return coco

def offset_and_extend(master_ocr, partial_ocr, base_idx, master_path, partial_path):
    for partial_idx,value in partial_ocr.items():
        new_idx = partial_idx + base_idx
        master_ocr[new_idx] = value
        os.rename(partial_path / f"{partial_idx:07.0f}.jpg", master_path / f"{new_idx:07.0f}.jpg")

def consolidate_data(list_of_paths):
    """ Combine collections - not tested

    """
    master_ocr = {}
    next_idx = 0
    for path in list_of_paths:
        ocr = load_json(path)

        if next_idx:
            next_idx = offset_and_extend(master_ocr, ocr, base_idx=next_idx,
                                         master_path=master_path,
                                         partial_path=path
                                         )
        else:
            next_idx = max(int(x) for x in ocr) + 1
            master_path = path

def load_and_test_image():
    pass


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

    coco = {
          "info": DEFAULT_COCO_INFO,
          "images": images,
          "licences": "No license, internal work product, confidential",
          "categories": categories,
          "annotations": annotations
    }

    with output_path.open("w") as ff:
        json.dump(coco,ff)

def change_key_recursive(reference_dict, my_dict, old_key, new_key):
    for i, ref_val in reference_dict.items():
        if isinstance(ref_val, list):
            for ii, item in enumerate(ref_val):
                if isinstance(item, dict):
                    change_key_recursive(item, my_dict[i][ii], old_key, new_key)
        if isinstance(i,str) and i == old_key:
            my_dict[new_key] = my_dict.pop(old_key)
        elif isinstance(ref_val, dict):
            change_key_recursive(ref_val, my_dict[i], old_key, new_key)

def draw_boxes_paragraph(ocr_format, background_img, origin=[0,0]):
    for paragraph in ocr_format["paragraphs"]:
        for line in paragraph["lines"]:
            for word in line["words"]:
                BBox._draw_box(BBox._offset_origin(word["bbox"], *origin), background_img)
            BBox._draw_box(BBox._offset_origin(line["bbox"], *origin), background_img)
        BBox._draw_box(BBox._offset_origin(paragraph["bbox"], *origin), background_img)

def draw_boxes_sections(ocr_format, background_img):
    for section in ocr_format["sections"]:
        draw_boxes_paragraph(section, background_img)

def fix(path):
    path = Path(path)
    dict = load_json(path)
    # for i,value in dict.items():
    #     dict[i].update({"height": 1152, "width": 768})
    reference_dict = copy.deepcopy(dict)
    print("done copying")
    change_key_recursive(reference_dict, dict, "box", "bbox")
    save_json(path, dict)
    return dict

def fix2(path):
    path = Path(path)
    dict = load_json(path)
    keys = list(dict.keys())
    for key in keys:
        if not (path.parent / (key+".jpg")).exists():
            del dict[key]
    save_json(path, dict)
    return dict


def load_and_draw_and_display(file_path, d=None):
    if d is None:
        d = load_json(Path(file_path).parent / "OCR.json")
    idx = Path(file_path).stem
    img = Image.open(file_path)
    draw_boxes_sections(d[idx], img)
    display(img)

if __name__ == '__main__':
    path = "/home/taylor/anaconda3/DATASET_0021/OCR.json"
    coco = "/home/taylor/anaconda3/DATASET_0021/COCO.json"

    #dict = fix(path="/home/taylor/anaconda3/DATASET_0021/OCR.json")
    #delete_extra_images(path)

    save_json(Path(path).parent / "COCO.json", ocr_to_coco(ocr_dict=path, data_set_name="Handwritten Pages"))
    #load_json(coco)

    if False:
        p = "/home/taylor/anaconda3/DATASET_0021/0036011.jpg"
        load_and_draw_and_display(p)
