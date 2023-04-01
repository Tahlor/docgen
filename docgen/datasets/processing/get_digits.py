import numpy as np
import os

def load_dataset(file_path):
    return np.load(file_path, allow_pickle=True).item()

def filter_digits_only(dataset):
    digits_data = {}
    for author in dataset:
        digits_data[author] = {}
        for letter in dataset[author]:
            # Check if the letter represents a digit
            if letter.isdigit():
                digits_data[author][letter] = dataset[author][letter]
    return digits_data

def save_dataset(data, file_path):
    np.save(file_path, data)

if __name__ == "__main__":
    input_file_path = 'G:/data/standard_hwr/emnist/full/EMNIST.npy'
    output_file_path = 'G:/data/standard_hwr/emnist/digits/EMNIST_digits.npy'

    # Create output directory if it does not exist
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    dataset = load_dataset(input_file_path)
    digits_only_data = filter_digits_only(dataset)
    save_dataset(digits_only_data, output_file_path)
