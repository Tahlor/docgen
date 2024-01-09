import shlex
import argparse
import shlex
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from hwgen.data.utils import show, display

def load_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

def apply_blur(img, kernel_size=5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def create_mask(img, lower_percentile, upper_percentile):
    lower_threshold = np.percentile(img, lower_percentile)
    upper_threshold = np.percentile(img, upper_percentile)
    threshold = (lower_threshold + upper_threshold) / 2
    mask = (img < threshold).astype(np.uint8)
    return mask

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)

def apply_mask(img, mask):
    inverted_mask = np.invert(mask.astype(bool))
    masked_img = img.copy()
    masked_img[inverted_mask] = 255  # Set non-ink areas to white
    return masked_img


def plot_images_for_review(original, masked, output_path=None, dpi=300):
    plt.figure(figsize=(original.shape[1] / dpi, original.shape[0] / dpi), dpi=dpi)
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')  # Optional: to remove axes for cleaner look

    plt.subplot(1, 2, 2)
    plt.imshow(masked, cmap='gray')
    plt.title('Masked Image')
    plt.axis('off')  # Optional: to remove axes for cleaner look

    if output_path is not None:
        plt.savefig(output_path.with_suffix(".jpg"), dpi=dpi, bbox_inches='tight')
    else:
        plt.show()

def plot_images(masked, output_path=None):
    cv2.imwrite(str(output_path), masked)

def process_images(folder_path, output_folder=None, lower_percentile=10, upper_percentile=70, kernel_size=3):
    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
    subfolders = [x for x in Path(folder_path).iterdir() if x.is_dir()]
    # remove the output folder from the list of subfolders
    if output_folder is not None:
        subfolders = [x for x in subfolders if x != output_folder]
    for subfolder in subfolders:
        for img_file in tqdm(list(Path(subfolder).glob('*.jfif')) + list(Path(subfolder).glob('*.jpg'))):
            img = load_image(str(img_file))
            if False:
                blurred_img = apply_blur(img, kernel_size)
                mask = create_mask(blurred_img, lower_percentile, upper_percentile)
                masked_img = apply_mask(img, mask)
            else:
                binarized_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, 11, 2)

            output_path = Path(output_folder) / f'masked_{img_file.name}' if output_folder is not None else None
            if False:
                plot_images(img, masked_img, output_path)
            else:
                plot_images(masked_img, output_path.with_suffix(".jpg"))
            print(f'Processed image: {img_file.name}')

def main(args):
    parser = argparse.ArgumentParser(description='Process images to separate ink from background.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
    parser.add_argument('--output_folder', type=str, default=None, help='Path to save output images')
    parser.add_argument('--lower_percentile', type=float, default=10, help='Lower percentile for thresholding')
    parser.add_argument('--upper_percentile', type=float, default=40, help='Upper percentile for thresholding')
    parser.add_argument('--kernel_size', type=int, default=5, help='Size of the Gaussian blur kernel')
    if args is not None:
        args = parser.parse_args(shlex.split(args))
    else:
        args = parser.parse_args()

    process_images(args.folder_path, args.output_folder, args.lower_percentile, args.upper_percentile, args.kernel_size)

if __name__ == "__main__":
    args = """'B:/document_backgrounds/handwriting/' 
    --output_folder 'B:/document_backgrounds/handwriting/MASKED'
    """.replace("\n","")
    main(args)
