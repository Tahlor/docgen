import argparse
from pathlib import Path
import cv2
import albumentations as A
from docgen.transforms.transforms import *
from tqdm import tqdm
from multiprocessing import cpu_count
import pickle

def load_bounding_boxes(pkl_path):
    with open(pkl_path, 'rb') as f:
        bounding_boxes = pickle.load(f)
    return bounding_boxes


class ImageTransformer:
    def __init__(self, args):
        self.args = args
        self.input_folders = args.input_folders
        self.output_folder = args.output_folder
        self.group_transform = A.OneOf([
            A.VerticalFlip(p=1),
            A.HorizontalFlip(p=1),
            A.Rotate(limit=(180, 180), p=1),
        ], p=0.75)

        self.pipeline = A.Compose([
            RandomBottomLeftEdgeCropSquare(p=1),
            self.group_transform,
        ])
        self.recursive = args.recursive
        self.rename_to_idx = args.rename_to_idx
        self.overwrite = args.overwrite
        self.bounding_box_file = args.bounding_box_file

    def transform_and_save_images(self):
        for input_folder in self.input_folders:
            self.process_folder(input_folder)

    def get_output(self, input_path, image_path, i):
        relative_path = image_path.relative_to(input_path)
        if self.rename_to_idx:
            img_name = relative_path.parent / f'{i:06d}.jpg'
        else:
            img_name = relative_path.with_suffix('.jpg').name
        relative_output_path_and_name = relative_path.parent / img_name
        return relative_output_path_and_name

    def process_folder(self, input_folder):
        input_path = Path(input_folder)
        output_path = Path(self.output_folder) / input_path.parent.name / input_path.name
        output_path.mkdir(parents=True, exist_ok=True)

        if self.recursive:
            input_path_glob = input_path.rglob
        else:
            input_path_glob = input_path.glob

        if self.bounding_box_file:
            bounding_boxes = load_bounding_boxes(self.bounding_box_file)
        else:
            bounding_boxes = {}

        files = sorted(list(input_path_glob('*')))
        for i, image_path in tqdm(enumerate(files), desc=f'Processing {input_path.name}'):

            relative_output_path_and_name = self.get_output(input_path, image_path, i)
            full_output_path = output_path / relative_output_path_and_name
            if full_output_path.exists() and not self.overwrite:
                continue
            if i == 0:
                print(f"\nTaking input from {image_path}\n and saving output to {full_output_path}.")
                print("Continuing...")

            # Read image
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            # Apply transformation
            if image_path.stem in bounding_boxes:
                bounding_box = bounding_boxes[image_path.stem]

            transformed = self.pipeline(image=image)['image']



            # Save the transformed image
            save_path = full_output_path

            cv2.imwrite(str(save_path), transformed)
            # alternative


def main(args):
    transformer = ImageTransformer(args)
    transformer.transform_and_save_images()

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Apply transformations to images in specified folders')
    parser.add_argument('--input_folders', nargs='*',
                        help='List of input folders containing images')
    parser.add_argument('--output_folder', type=str,
                        help='Output folder where transformed images will be saved')
    parser.add_argument("--recursive", action="store_true",
                        help="If true, the script will recursively walk the input folder")
    parser.add_argument("--threads", type=int, default=cpu_count() - 2,
                        help="The number of threads used when uploading images to s3")
    # rename to idx number
    parser.add_argument("--rename_to_idx", action="store_true",
                        help="If true, the script will rename files to idx number")
    parser.add_argument("--overwrite", action="store_true",
                        help="If true, the script will overwrite existing files")
    parser.add_argument("--bounding_box_file", type=str,
                        help="Path to the bounding box file, not implemented")

    if args is not None:
        import shlex
        return parser.parse_args(shlex.split(args))
    else:
        return parser.parse_args()

if __name__ == '__main__':
    args = "--rename_to_idx"
    args = parse_args(args)
    # If no arguments are provided, use the defaults
    if not args.input_folders or not args.output_folder:
        args.input_folders = [
            # "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/aged blank letter with subtle imprints of text from its other side, as if the ink permeated through the paper; full frame",
            # "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/blank paper with a some random highlighter and marker marks, full frame",
            # "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/blank paper with mold damage, full frame",
            # "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/old paper with many ink marks, crinkles, wrinkles, and imperfections and variable lighting",
            # "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/paper with many ink marks, crinkles, wrinkles, and imperfections and variable lighting",
            # "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/white paper with many ink marks, crinkles, wrinkles, and imperfections and variable lighting",
            # "//?/G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/aged blank letter with imprints of text from the reverse side, as if some vestiges of the ink permeated through the paper, but impossible to read; full frame",
            #r"G:\s3\synthetic_data\resources\backgrounds\synthetic_backgrounds\dalle\document_backgrounds\paper_only\white paper with many ink marks, crinkles, wrinkles, and imperfections and variable lighting"

            "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/old_highlighter_marks",
            "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/old_wrinkled_var_light",
            "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/paper_only/old_liquid_stains",
        ]

        root = "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/with_backgrounds"
        args.input_folders = [
            str(subfolder) for subfolder in Path(root).iterdir() if subfolder.is_dir()
        ]
        args.output_folder = """G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/CROPPED_with_backgrounds"""
        args.bounding_box_file = r"""G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/with_backgrounds/bounding_boxes_FILESTEM_BOUNDINGBOX.pkl"""
    main(args)

