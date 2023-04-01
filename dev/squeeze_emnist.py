import numpy as np
from tqdm import tqdm
def process_and_save_dataset(path):
    # Load the dataset
    dataset = np.load(path, allow_pickle=True).item()

    # Process each image
    for i, author in enumerate(dataset.keys()):
        for ii, letter in enumerate(tqdm(dataset[author].keys())):
            dataset[author][letter] = [img.squeeze() for img in dataset[author][letter]]
            if i==ii==0:
                print(f"{dataset[author][letter][0].shape}")
    # Resave the dataset
    np.save(path, dataset)

path = "G:/data/standard_hwr/emnist/EMNIST.npy"

process_and_save_dataset(path)
