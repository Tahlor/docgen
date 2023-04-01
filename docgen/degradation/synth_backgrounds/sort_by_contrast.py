from tqdm import tqdm
from PIL import Image, ImageStat
import numpy as np
from pathlib import Path
from typing import Tuple
import argparse


def scale_value(value: float, src_range: Tuple[float, float], dst_range: Tuple[int, int]) -> int:
    """
    Scale a value from the source range to the destination range.

    Args:
        value (float): The value to scale.
        src_range (Tuple[float, float]): The source range (min, max).
        dst_range (Tuple[int, int]): The destination range (min, max).

    Returns:
        int: The value scaled to the destination range.
    """
    return int((value - src_range[0]) / (src_range[1] - src_range[0]) * (dst_range[1] - dst_range[0]) + dst_range[0])


def calculate_darkness(image_path: Path) -> float:
    """
    Calculate the darkness of the darkest 5 percent of pixels in the image.

    Args:
        image_path (Path): The path to the image file.

    Returns:
        float: The darkness value.
    """
    with Image.open(image_path) as img:
        # Convert image to grayscale
        grayscale = img.convert("L")
        # Calculate the histogram
        histogram = grayscale.histogram()
        # Calculate the cutoff for the darkest 5 percent
        pixel_count = sum(histogram)
        cutoff = pixel_count * 0.05
        accumulated = 0
        for i, count in enumerate(histogram):
            accumulated += count
            if accumulated >= cutoff:
                break
        # i is the darkness level of the darkest 5% of pixels
        return i / 255  # normalize to the range 0-1


class SortByContrast:

    def __init__(self, args):
        self.args = args
        self.input_folders = args.input_folders
        self.output_folder = args.output_folder
        self.recursive = args.recursive
        self.make_copy = self.output_folder is not None

    def get_output_path(self, input_folder, image_path, scaled_value) -> Path:
        """
        Calculate the output path based on the input path.

        Args:
            input_path (Path): The input path to an image file.

        Returns:
            Path: The corresponding output path.
        """
        relative_path = image_path.relative_to(input_folder)
        if self.output_folder:
            out = Path(self.output_folder) / relative_path
        else:
            out = image_path
        return out.parent / f"{scaled_value}_{out.stem}{out.suffix}"

    def rename_files(self):
        """
        Rename files in the specified folder(s) based on the darkness of the darkest 5 percent of pixels.
        """
        completed_idx = 0
        for input_folder in self.input_folders:
            print(f"Processing folder: {input_folder}")
            folder_path = Path(input_folder)
            if not folder_path.is_dir():
                print(f"The specified path '{folder_path}' is not a directory.")
                continue

            folder_glob = folder_path.rglob if self.recursive else folder_path.glob
            for i, image_path in tqdm(enumerate(folder_glob("*"))):
                if image_path.name[:2].isdigit() and "_" in image_path.name[4]:
                    continue
                if image_path.is_dir():
                    continue
                if image_path.suffix.lower() not in [".jpg", ".jpeg", ".jfif"]:
                    continue
                try:
                    darkness = calculate_darkness(image_path)
                except Exception as e:
                    print(f"Failed to process {image_path}: {e}")
                    continue
                scaled_value = scale_value(darkness, (0, 1), (1, 100))
                new_name = self.get_output_path(input_folder, image_path, scaled_value)

                if completed_idx == 0:
                    print(f"Renamed {image_path} to \n{new_name}")
                    input("Continue?")
                if self.make_copy:
                    image_path.copy(new_name)
                else:
                    image_path.rename(new_name)
                completed_idx += 1

def main(args=None):
    parser = argparse.ArgumentParser(description='Rename image files based on darkness.')
    parser.add_argument('input_folders', type=str, nargs='+', help='Path to the directory/directories with images.')
    parser.add_argument('--output-folder', '-o', type=Path, help='Path to the output directory.')
    parser.add_argument('--recursive', '-r', action='store_true', help='Operate recursively on subdirectories.')

    if args is not None:
        import shlex
        args = parser.parse_args(shlex.split(args))
    else:
        args = parser.parse_args()

    for folder in args.input_folders:
        if not Path(folder).is_dir():
            print("The specified path is not a directory.")
            return

    sorter = SortByContrast(args)
    sorter.rename_files()

if __name__ == "__main__":
    from docgen.windows.utils import map_drive
    if False:
        path = r'G:\s3\synthetic_data\resources\backgrounds\synthetic_backgrounds\dalle\document_backgrounds\paper_only'
        map_drive(path, "B:")

        args = r"'G:\s3\synthetic_data\resources\backgrounds\synthetic_backgrounds\dalle\document_backgrounds\paper_only\white paper with many ink marks, crinkles, wrinkles, and imperfections and variable lighting'"
        args = r"'B:\' -r "
    else:
        path = r'G:\s3\synthetic_data\resources\backgrounds\synthetic_backgrounds\dalle'
        map_drive(path, "B:")
        args = "'B:/document_backgrounds/handwriting' -r"
    main(args)
