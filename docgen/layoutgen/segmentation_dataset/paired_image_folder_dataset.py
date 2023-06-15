import re
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
                 label_dir=None,
                 transform=None,
                 max_uniques=None,
                 length_override=None
                 ):
        super().__init__()
        if label_dir is None:
            label_dir = img_dir
        self.max_uniques = int(max_uniques) if max_uniques else None
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.path_database = self.process_img_file_list(img_dir, label_dir)
        self.length = int(length_override) if length_override else len(self.path_database)

        if len(self.path_database) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.transform = transform
        if self.transform is None:
            # just convert it to tensor, compose it
            self.transform = Compose([transforms.ToTensor()]
                                     )
        elif isinstance(self.transform, list):
            self.transform = Compose(self.transform)

    def process_img_file_list(self, img_dir, label_folder):
        """ Safer to parse the image file to get the index, create a dictionary

        Returns:

        """
        label_folder = Path(label_folder)
        imgs = Path(img_dir).glob("input*.png")
        #self.labels = sorted(Path(label_folder).glob("label*.png"))
        path_database = {}
        for img_path in imgs:
            id = re.search(r"(\d+)$", img_path.stem).group(1)
            label_path = label_folder / f"label_{id}.png"
            if label_path.exists():
                # path_database[int(id)] = {"img_path": img.relative_to(self.img_dir),
                #                           "label_path": label_path.relative_to(self.label_dir)}
                path_database[int(id)] = {"img_path": img_path,
                                          "label_path": label_path}

            else:
                logger.warning(f"Label {label_path} does not exist")

        return path_database

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
        return self.length

    def _get(self, idx):
        while True:
            img_path = ""
            try:
                idx = (idx % self.unique_length)+1
                paths = self.path_database[idx]
                img_path, label_path = paths["img_path"], paths["label_path"]

                # load from png and convert to tensor
                img = Image.open(img_path).convert("RGB")
                label = Image.open(label_path).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                    label = self.transform(label)

                return {'image': img, 'mask': label}
            except:
                logger.exception(f"Error loading image {idx} {img_path}")
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