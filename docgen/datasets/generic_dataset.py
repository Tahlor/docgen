from torch.utils.data import Dataset
from torchvision.transforms import Compose
import torch


class GenericDataset(Dataset):
    def __init__(self, transform_list=None,
                 max_uniques=None,
                 max_length_override=None,
                 collate_fn=None,
                 ):
        """ * You must sent .length attribute
            * When creating the dataset, stop when length override is reached

        Args:
            transform_list:
            max_uniques: keep epochs the same, but only train on X unique examples
            max_length_override: shorten a dataset epoch to this length
        """
        super().__init__()

        if collate_fn is None:
            collate_fn = GenericDataset.dict_collate_fn
        else:
            collate_fn = collate_fn

        self.max_uniques = int(max_uniques) if max_uniques else None
        self.max_length_override = int(max_length_override) if max_length_override else None
        self.collate_fn = collate_fn

        self.transform_list = transform_list
        if self.transform_list:
            self.transform_composition = Compose(self.transform_list)
        else:
            self.transform_composition = Compose([])

    @property
    def unique_length(self):
        """ Basically to allow overfitting without shortening an epoch

        Returns:

        """
        if self.max_uniques:
            return self.max_uniques
        else:
            return len(self)

    def __len__(self):
        if self.max_length_override:
            return min(self.length, self.max_length_override)
        else:
            return self.length

    def _validate_idx(self, idx):
        """ For looping over the dataset, if you want to overfit and reuse the same examples without changing epoch length

        Args:
            idx:

        Returns:

        """
        idx = (idx % self.unique_length)
        return idx

    @staticmethod
    def dict_collate_fn(batch, tensor_keys=("image", "mask")):
        """

        Args:
            batch:
            tensor_keys: keys in dict that should be collated as a tensor instead of a list

        Returns:
            batch:

        """
        keys = batch[0].keys()
        collated_batch = {}

        for key in keys:
            if tensor_keys and key in tensor_keys:
                collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                collated_batch[key] = [item[key] for item in batch]
        return collated_batch

    @staticmethod
    def tensor_collate_fn(batch):
        return torch.stack(batch, dim=0)
