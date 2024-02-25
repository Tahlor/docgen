import random
from torch.utils.data import Dataset
import numpy as np
import warnings

class MetaDataset(Dataset):
    def __init__(self, datasets, weights=None, max_length_override=None, shuffle=True,
                **kwargs):
        self.datasets = datasets
        if weights is not None:
            self.datasets, weights = self.filter_out_0_weights(datasets, weights)
        self.weights = self.normalize_weights(weights)
        self.idx_map = self.compute_idx_to_dataset_index_map()
        self.length = max_length_override if max_length_override else len(self.idx_map)
        self.shuffle = shuffle
        self.counter = 0
        self.key_intersection = self.get_key_intersection()
        print(f"MetaDataset created with length {self.length}")

    def normalize_weights(self, weights):
        if weights is None:
            weights = [1] * len(self.datasets)
        weights = [p / sum(weights) for p in weights]
        return weights

    def filter_out_0_weights(self, img_dataset_config_paths, weights):
        configs = [config for i,config in enumerate(img_dataset_config_paths) if weights[i] > 0]
        weights = self.normalize_weights([w for w in weights if w > 0])
        return configs, weights

    def compute_idx_to_dataset_index_map(self):
        d = {}
        idx = 0
        for i, dataset in enumerate(self.datasets):
            for j in range(len(dataset)):
                d[idx] = (i,j)
                idx += 1
        return d

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.shuffle:
            chosen_dataset = random.choices(self.datasets, self.weights)[0]
            idx = random.randint(0, len(chosen_dataset) - 1)
            out = chosen_dataset[idx % len(chosen_dataset)]
        else:
            if idx >= self.length:
                warnings.warn(f"Index {idx} is out of bounds for dataset of length {self.length}")
            dataset_idx, item_idx = self.idx_map[idx % self.length]
            out = self.datasets[dataset_idx][item_idx]
        if isinstance(out, dict):
            out = {k: v for k, v in out.items() if k in self.key_intersection}
        return out

    def get_key_intersection(self):
        batch_of_examples = [set(dataset[0].keys()) for dataset in self.datasets]
        # get the overlapping keys
        keys = set(batch_of_examples[0])
        for i in range(1, len(batch_of_examples)):
            keys = keys.intersection(batch_of_examples[i])
        return keys


# Example of usage
# dataset1, dataset2 = SomeDataset(), AnotherDataset()
# meta_dataset = MetaDataset([dataset1, dataset2], probabilities=[0.7, 0.3], mix_batches=True)
