import json
import h5py
import numpy as np
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_hdf5", help="Path to the input HDF5 file")
    return parser.parse_args()


def main(args):
    with h5py.File(args.input_hdf5, "r") as hf:
        images = hf["images"]
        print(f"Number of images: {len(images)}")


        text_labels = hf["text"]
        ocr_labels = hf["ocr"]
        coco_labels = hf["coco"]

        #print(f"Number of TEXT elements: {text_labels}")

        print(f"Number of TEXT elements: {len(text_labels)}")
        print(f"Number of OCR elements: {len(ocr_labels)}")
        print(f"Number of COCO elements: {len(coco_labels)}")

        # Print the first annotation
        #print(list(text_labels.keys()))
        #print(list(ocr_labels.keys()))
        #print(list(coco_labels.keys()))
        first_key = list(text_labels.keys())[0]
        first_annotation = text_labels[first_key][()]
        print(f"First annotation: {first_annotation}")

        # Save the first image
        first_image_key = list(images)[0]
        first_image_data = hf[first_image_key][()]
        first_image = Image.fromarray(np.uint8(first_image_data))
        first_image.save("first_image.png")

if __name__ == "__main__":
    main(parse_args())


