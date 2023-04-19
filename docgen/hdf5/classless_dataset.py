import os
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import re

re_digit = re.compile(r'\D', re.IGNORECASE)
def str_to_int(s):
    return int(re_digit.sub('', s))

class UnlabeledImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = [
            fname for fname in os.listdir(root_dir)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.root_dir, image_filename)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return {
                "image":image,
                "idx":str_to_int(Path(image_filename).stem),
                }
