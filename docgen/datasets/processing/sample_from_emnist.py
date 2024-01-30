import numpy as np

def load_dataset(file_path):
    return np.load(file_path, allow_pickle=True).item()

def thin_dataset(dataset):
    thinned_data = {}
    for author in dataset:
        thinned_data[author] = {}
        for letter in dataset[author]:
            images = dataset[author][letter]
            # Take every tenth image
            thinned_images = images[::10]
            thinned_data[author][letter] = thinned_images
    return thinned_data

def save_thinned_dataset(thinned_data, file_path):
    np.save(file_path, thinned_data)

if __name__ == "__main__":
    file_path = 'G:/data/standard_hwr/emnist/EMNIST.npy'
    dataset = load_dataset(file_path)
    thinned_data = thin_dataset(dataset)
    save_thinned_dataset(thinned_data, file_path.replace('.npy', '_tiny.npy'))
