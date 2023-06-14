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

class PairedImgLabelImageFolderDataset(Dataset):
    def __init__(self, img_dir,
                 label_folder=None,
                 transform=None,
                 max_uniques=None):
        super().__init__()
        if label_folder is None:
            label_folder = img_dir
        self.max_uniques = max_uniques
        self.imgs = sorted(Path(img_dir).glob("input*.png"))
        self.labels = sorted(Path(label_folder).glob("label*.png"))

        if len(self.imgs) != len(self.labels):
            logger.warning(f"Number of images {len(self.imgs)} does not match number of labels {len(self.labels)}")
        elif len(self.imgs) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.transform = transform
        if self.transform is None:
            # just convert it to tensor, compose it
            self.transform = Compose([transforms.ToTensor()]
                                     )

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
        return len(self.imgs)

    def _get(self, idx):
        while True:
            try:
                idx = idx % self.unique_length
                img_path = self.imgs[idx]
                label_path = self.labels[idx]

                # load from png and convert to tensor
                img = Image.open(img_path).convert("RGB")
                label = Image.open(label_path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                    label = self.transform(label)

                return {'image': img, 'mask': label}
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