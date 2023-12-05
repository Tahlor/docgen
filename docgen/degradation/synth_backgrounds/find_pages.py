import argparse
import cv2
import logging
import numpy as np
import os
import pickle
import sys
import torch
from doc_ufcn import models
from doc_ufcn.main import DocUFCN
from pathlib import Path
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(format="[%(levelname)s] %(message)s", stream=sys.stdout, level=logging.INFO)

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def draw_boxes(image, boxes):
    for box in boxes:
        polygon = box['polygon']
        points = [tuple(point) for point in polygon]
        cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 3)
    return image

class PageSegmenter:
    def __init__(self, model_name, input_folder, output_folder, boxes_file):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path, parameters = models.download_model(model_name)
        self.model = DocUFCN(len(parameters['classes']), parameters['input_size'], self.device)
        self.model.load(model_path, parameters['mean'], parameters['std'], mode='eval')
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.boxes_file = Path(boxes_file)

    def draw_boxes(self, image, boxes):
        for box in boxes:
            polygon = box['polygon']
            points = [tuple(point) for point in polygon]
            cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 3)
        return image

    def process_image(self, image_path):
        image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB)
        detected_polygons, _, _, _ = self.model.predict(image, raw_output=True, mask_output=True, overlap_output=True)

        # Save bounding boxes
        bounding_boxes = detected_polygons

        # Save images with drawn boxes
        overlapped_image = self.draw_boxes(image, detected_polygons[1])  # Assuming class 1 is the target class
        output_image_path = self.output_folder / image_path.relative_to(self.input_folder)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)

        # if jfif use jpg
        if output_image_path.suffix.lower() == ".jfif":
            output_image_path = output_image_path.with_suffix(".jpg")

        cv2.imwrite(str(output_image_path), cv2.cvtColor(overlapped_image, cv2.COLOR_RGB2BGR))

        return bounding_boxes, output_image_path

    def process_all_images(self):
        bounding_boxes = {}
        for image_path in tqdm(list(self.input_folder.rglob('*'))):
            if image_path.is_dir() or image_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jfif"]:
                continue
            if self.output_folder.name in image_path.parts:
                continue

            boxes, output_path = self.process_image(image_path)
            bounding_boxes[str(output_path)] = boxes

        # Save bounding boxes data as pickle file
        with open(self.boxes_file, 'wb') as f:
            pickle.dump(bounding_boxes, f)

        print("Processing complete. Bounding boxes and images saved.")

def parse(args=None):
    parser = argparse.ArgumentParser(description='Document Page Detection')
    parser.add_argument('--image_folder', type=Path, required=True, help='Path to the folder containing images')
    parser.add_argument('--output_folder', type=Path, help='Path to the folder to save results',
                        default=None)
    parser.add_argument('--boxes_file', type=Path, help='Path to save the bounding boxes pickle file', default=None)

    if args is not None:
        import shlex
        args = parser.parse_args(shlex.split(args))
    else:
        args = parser.parse_args()

    # set defaults
    if args.output_folder is None:
        args.output_folder = args.image_folder / "with_segmentations"
    if args.boxes_file is None:
        args.boxes_file = args.image_folder / "bounding_boxes.pkl"
    return args

if __name__ == "__main__":
    args = fr"--image_folder B:/document_backgrounds/with_backgrounds"
    args = parse(args)
    segmenter = PageSegmenter('generic-page', args.image_folder, args.output_folder, args.boxes_file)
    segmenter.process_all_images()
