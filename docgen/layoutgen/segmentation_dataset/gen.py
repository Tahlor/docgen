import sys
from docgen.bbox import BBox
from docgen.render_doc import composite_images_PIL
from PIL import Image
from typing import Dict, Any, Union, Optional
from torch.utils.data import Dataset
import random

class Gen:

    def get(self):
        raise NotImplementedError()
    def __getitem__(self, item):
        return self.get()

    def __len__(self):
        return sys.maxsize

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()

    def get_random_bbox(self, img_size=None, font_size=10):
        if img_size is None:
            img_size = self.img_size
        bbox = BBox("ul", [0,0,*self.img_size]).random_subbox(max_size_x=img_size[1],max_size_y=img_size[0], min_size_x=font_size*6, min_size_y=font_size)
        return bbox

    @staticmethod
    def composite_pil(img, bg, pos, offset):
        return composite_images_PIL(img, bg, pos, offset)

    @staticmethod
    def composite_pil_from_blank(img, bg_size, pos):
        """

        Args:
            img:
            bg_size: PIL size, (width, height)
            pos:

        Returns:

        """
        bg_img = Image.new("RGB", bg_size, (255,255,255))
        return composite_images_PIL(img, bg_img, pos, offset=(0,0))
    
    def pickle_prep(self):
        """ Delete anything that can't be pickled
        """
        if hasattr(self, "generator") and hasattr(self.generator, "image"):
            del self.generator.image

class NaiveGenerator(Gen):
    def __init__(self, img_size):
        self.img_size = img_size

    def get(self):
        return Image.new("RGB", self.img_size, (255,255,255))

class RandomSelectorDataset(Dataset):
    """Dataset to randomly select and return an item from one of the given datasets.

    Attributes:
        datasets (Dict[str, Dataset]): A dictionary containing dataset names as keys and dataset objects as values.
        keys (list): A list of dataset names (keys of the `datasets` dictionary).
        prob (Dict[str, float]): Dictionary specifying the probability of selecting each dataset.
        cumulative_prob (Dict[str, float]): Dictionary specifying the cumulative probability for selecting datasets.

    Args:
        datasets (Dict[str, Dataset]): Dictionary containing dataset names and their corresponding dataset objects.
        prob (Optional[Dict[str, float]]): Optional dictionary containing dataset names and their selection probabilities.
        allow_empty (bool): If True, adds an option to return an empty item.
        empty_prob (float): Probability of selecting the empty dataset. Only valid if `allow_empty=True`.
    """

    def __init__(self, datasets: Dict[str, Dataset], prob: Optional[Dict[str, float]] = None,
                 allow_empty: bool = False, empty_prob: float = 0.0) -> None:
        self.datasets = datasets
        self.keys = list(datasets.keys())
        self.allow_empty = allow_empty
        if prob is None:
            self.prob = {k: 1/len(self.keys) for k in self.keys}
            if self.allow_empty:
                self.prob['none'] = empty_prob
        else:
            self.prob = prob
        self.cumulative_prob = self._calculate_cumulative_prob()

    def _calculate_cumulative_prob(self) -> Dict[str, float]:
        cumulative_prob = 0.0
        cumulative_prob_dict = {}
        for k in self.keys:
            cumulative_prob += self.prob[k]
            cumulative_prob_dict[k] = cumulative_prob
        if self.allow_empty:
            cumulative_prob += self.prob['none']
            cumulative_prob_dict['none'] = cumulative_prob
        return cumulative_prob_dict

    def _select_dataset(self) -> str:
        rand_val = random.random()
        for k, cumulative_prob in self.cumulative_prob.items():
            if rand_val <= cumulative_prob:
                return k
        return 'none' if self.allow_empty else self.keys[-1]  # Fallback

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets.values())

    def __getitem__(self, index: int) -> Dict[str, Union[Dict[str, Any], str]]:
        """Fetches an item from a randomly selected dataset.

        Args:
            index (int): Index of the item to be fetched.

        Returns:
            Dict[str, Union[Dict[str, Any], str]]: A dictionary containing the fetched item and its source dataset name.
        """
        selected_key = self._select_dataset()
        if selected_key == 'none':
            return {'source': 'none', "image": None, "mask": None, }

        selected_dataset = self.datasets[selected_key]
        index = index % len(selected_dataset)  # Ensure the index doesn't exceed the dataset size
        item = selected_dataset[index]
        item["source"] = selected_key
        return item
