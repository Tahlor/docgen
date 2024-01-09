from hwgen.data.utils import show, display
import shlex
import argparse
import os
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from tqdm import tqdm
import torch
from torchvision import transforms

def process_image_cuda(image_path, threshold, kernel_size):
    # Load image using PIL and convert to grayscale
    image = Image.open(image_path)

    # Transform to tensor and add batch dimension
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)

    # Take the max of each channel
    image_tensor = image_tensor.max(dim=1, keepdim=True)[0]

    # Use CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_tensor = image_tensor.to(device)

    # Apply threshold
    binary_image = (image_tensor < threshold).float()

    # Create convolution kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)

    # Convolution
    feature_map = torch.nn.functional.conv2d(binary_image, kernel, padding=kernel_size // 2)

    # Check for maximum value
    return torch.any(feature_map == kernel_size ** 2).item()

def process_image(image_path, threshold, kernel_size):
    image = Image.open(image_path)
    image_array = np.asarray(image)

    # Take the max of each channel
    image_array = image_array.max(axis=2)

    # Applying threshold
    binary_image = (image_array < threshold * 255).astype(int)

    # Convolution
    kernel = np.ones((kernel_size, kernel_size), dtype=int)
    feature_map = convolve2d(binary_image, kernel, mode='valid')

    # Check for maximum value
    return np.any(feature_map == kernel_size**2)

def main(input_folder, threshold, kernel_size, output_folder):
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in tqdm(input_path.glob('*.png')):  # Assuming PNG images
        if process_image_cuda(image_file, threshold, kernel_size):
            # Move image to output folder
            #show(Image.open(image_file))
            image_file.rename(output_path / image_file.name)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Process images.')
    parser.add_argument('input_folder', type=str, help='Input folder path')
    parser.add_argument('threshold', type=float, help='Threshold for binary conversion')
    parser.add_argument('kernel_size', type=int, help='Size of the kernel for convolution')
    parser.add_argument('-o', '--output_folder', type=str, help='Output folder path')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(args))

    if args.output_folder is None:
        args.output_folder = f"{args.input_folder}_blocks{args.kernel_size}"

    return args

if __name__ == "__main__":
    args = """ "G:/s3/forms/PDF/IRS/other_elements/images/" 0.97 10 """
    args = """ "G:/s3/forms/PDF/OPM/other_elements/images/" 0.97 10 """
    args = parse_args(args)
    main(args.input_folder, args.threshold, args.kernel_size, args.output_folder)
