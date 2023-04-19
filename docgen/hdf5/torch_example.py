from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import h5py
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
from torch.utils.data import Dataset

def show(img):
    from matplotlib import pyplot as plt
    if img.shape[0] == 1:
        img = img.squeeze(0)
    plt.imshow(img, cmap="gray")
    plt.show()

def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._open_file()

def worker_cleanup_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset._close_file()

class HDF5ImageDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.file = None
        self.images = None
        self.labels = None
        self.length = self.compute_length()
    def _open_file(self):
        self.file = h5py.File(self.hdf5_file, 'r')
        self.images = self.file['images']
        self.labels = self.file['labels']

    def compute_length(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            #return len(f['images'])
            return len(f['labels'])

    def _close_file(self):
        self.file.close()
        self.file = None
        self.images = None
        self.labels = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.file is None:
            self._open_file()

        out = {}
        for key in self.labels.keys():
            out[key] = self.labels[key][index]

        img_data = self.images[index]
        img_filename = index
        img_tensor = torch.from_numpy(img_data)
        if img_tensor.dim() == 2:
            img_tensor = img_tensor.unsqueeze(-1)
        img_tensor = img_tensor.permute(2, 0, 1)  # Convert to (C, H, W) format
        out.update({'image': img_tensor, 'filename': img_filename})
        return out


if __name__ == '__main__':
    hdf5_dataset = HDF5ImageDataset('french.hdf5')
    dataloader = DataLoader(hdf5_dataset, batch_size=32, shuffle=True, num_workers=4)
    #dataloader = DataLoader(hdf5_dataset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn,)

    for batch in dataloader:
        output = batch
        images = output['image']
        show(images[0].numpy())
        print(images.shape)
        print(output['text'][0])
        input()