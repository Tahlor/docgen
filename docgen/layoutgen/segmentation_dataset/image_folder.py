from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDataset
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class NaiveImageFolder(Dataset):
    def __init__(self, img_dir, transform=None, max_length=None, color_scheme="RGB", **kwargs):
        super().__init__()

        self.imgs = list(Path(img_dir).glob("*.png")) + list(Path(img_dir).glob("*.jpg"))
        self.imgs = sorted(self.imgs)
        self.color_scheme = color_scheme

        if len(self.imgs) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.transform = transform
        if self.transform is None:
            # just convert it to tensor, compose it
            self.transform = Compose([transforms.ToTensor()]
                                     )
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
                if self.transform is not None:
                    img = self.transform(img)

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