from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDataset
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose
class PairedImgLabelImageFolderDataset(Dataset):
    def __init__(self, img_folder, label_folder=None, transform=None):
        super().__init__()
        if label_folder is None:
            label_folder = img_folder
        self.imgs = sorted(Path(img_folder).glob("img*.png"))
        self.labels = sorted(Path(label_folder).glob("label*.png"))
        self.transform = transform
        if self.transform is None:
            # just convert it to tensor, compose it
            self.transform = Compose([transforms.ToTensor()]
                                     )

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:

        """
        img_path = self.imgs[idx]
        label_path = self.labels[idx]

        # load from png and convert to tensor
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)

        return {'image': img, 'mask': label}

    @staticmethod
    def collate_fn(batch):
        return SemanticSegmentationDataset.collate_fn(batch)