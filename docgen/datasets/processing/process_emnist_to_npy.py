"""
I want to download EMNIST and make it so they are black letters on white background, and then save to a npy file.

dataset["UNKNOWN_AUTHOR"]["{the_letter_or_digit}"] = [list of images scaled from 0-1, shape: height, width, 1 channel]

^ this is the same format that HW dataset uses
"""

import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
from tqdm import tqdm
from hwgen.data.utils import show,display


def download_emnist():
    dataset = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=ToTensor())
    return dataset

def process_images(dataset):
    processed_data = {}
    for image, label in tqdm(dataset):
        # Convert to black letters on white background and scale
        image = 1 - image.numpy()[0]
        image = np.expand_dims(image, axis=-1)

        ## mirror and rotate 90 degrees
        image = np.rot90(image, k=3)
        image = np.fliplr(image)
        image = np.ascontiguousarray(image.squeeze())
        author = "UNKNOWN_AUTHOR"

        if label < 10:  # Digits
            letter = str(label)
        else:  # Letters
            letter = chr(label + 87)  # 10 corresponds to 'a', 11 to 'b', etc.

        if author not in processed_data:
            processed_data[author] = {}
        if letter not in processed_data[author]:
            processed_data[author][letter] = []
        processed_data[author][letter].append(image)

    return processed_data

def save_to_npy(processed_data):
    np.save(f'G:/data/standard_hwr/emnist/EMNIST.npy', processed_data)

# Main execution
if __name__ == "__main__":
    emnist_data = download_emnist()
    processed_data = process_images(emnist_data)
    save_to_npy(processed_data)
