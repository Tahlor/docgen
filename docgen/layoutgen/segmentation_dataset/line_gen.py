from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDataset
import torch
import cv2
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset
import torch
import aggdraw
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image


class RandomLineDataset(Dataset):
    def __init__(self, size=(448, 448), shape_count_range=(1, 10), line_thickness_range=(1, 5)):
        super().__init__()
        self.size = size
        self.shape_count_range = shape_count_range
        self.line_thickness_range = line_thickness_range

    def _random_line(self):
        start_point = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        end_point = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        line_thickness = np.random.randint(*self.line_thickness_range)
        return {"xy": (start_point, end_point), "width": line_thickness}

    def draw_alias(self):
        img = Image.new('RGB', self.size, "white")
        draw = ImageDraw.Draw(img)
        for _ in range(np.random.randint(*self.shape_count_range)):
            draw.line(**self._random_line(), fill="black")
        return {'image': torch.from_numpy(np.array(img))}

    def draw_no_alias(self):
        img = Image.new('RGB', self.size, "white")
        draw = aggdraw.Draw(img)
        pen = aggdraw.Pen("black")
        for _ in range(np.random.randint(*self.shape_count_range)):
            xy, width = self._random_line()
            pen.width = width
            draw.line(xy, pen)
        draw.flush()
        return {'image': torch.from_numpy(np.array(img))}

    def __getitem__(self, idx):
        return self.draw_alias()

class BoxDataset(RandomLineDataset):
    def _random_box(self):
        upper_left_point = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        lower_right_point = np.random.randint(upper_left_point[0], self.size[0]), np.random.randint(upper_left_point[1], self.size[1])
        line_thickness = np.random.randint(*self.line_thickness_range)
        return {"xy": (upper_left_point, lower_right_point), "width": line_thickness}

    def draw_no_alias(self):
        img = Image.new('RGB', self.size, "white")
        draw = aggdraw.Draw(img)
        pen = aggdraw.Pen("black")
        for _ in range(np.random.randint(*self.shape_count_range)):
            xy, width = self._random_box()
            pen.width = width
            draw.rectangle(xy, pen)
        draw.flush()
        return {'image': torch.from_numpy(np.array(img))}

    def __getitem__(self, item):
        return self.draw_no_alias()


# to test the class
def test_random_line_dataset():
    dataset = RandomLineDataset()
    for i in range(10):
        sample = dataset[i]
        img = Image.fromarray(sample['image'].numpy().astype('uint8'), 'RGB')
        img.show()


if __name__=="__main__":
    test_random_line_dataset()
