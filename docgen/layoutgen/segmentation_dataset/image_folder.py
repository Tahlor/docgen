from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDataset
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose
import logging
from docgen.transforms.transforms import ResizeAndPad
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class NaiveImageFolder(Dataset):
    def __init__(self, img_dir, transform_list=None, max_length=None, color_scheme="RGB", longest_side=None, **kwargs):
        super().__init__()

        self.imgs = list(Path(img_dir).glob("*.png")) + list(Path(img_dir).glob("*.jpg"))
        self.imgs = sorted(self.imgs)
        self.color_scheme = color_scheme

        if len(self.imgs) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.transform_list = transform_list

        if self.transform_list:
            if longest_side:
                raise ValueError("Cannot specify longest_side and transform_list")

        else:
            self.transform_list = []
            if longest_side:
                resize_and_pad = ResizeAndPad(longest_side, 32)
                # resize so longest side is this, pad the other side
                self.transform_list.append(resize_and_pad)

            self.transform_list.append(transforms.ToTensor())

        self.transform_composition = Compose(self.transform_list)

        self.max_length = max_length if max_length is not None else len(self.imgs)

    def __len__(self):
        return min(len(self.imgs), self.max_length)

    def _get(self, idx):
        while True:
            try:
                idx = idx % len(self.imgs)
                img_path = self.imgs[idx]

                # load from png and convert to tensor
                img = Image.open(img_path).convert(self.color_scheme)
                if self.transform_composition is not None:
                    img = self.transform_composition(img)

                return {'image': img, }
            except:
                logger.exception(f"Error loading image {img_path}")
                idx += 1

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:

        """
        return self._get(idx)


    @staticmethod
    def collate_fn(batch):
        return SemanticSegmentationDataset.collate_fn(batch)