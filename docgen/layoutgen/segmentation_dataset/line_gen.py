from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDatasetImageFolder
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
from docgen.layoutgen.segmentation_dataset.gen import Gen


class LineGenerator(Gen):
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
        return {'image': img}

    def draw_no_alias(self):
        img = Image.new('RGB', self.size, "white")
        draw = aggdraw.Draw(img)
        pen = aggdraw.Pen("black")
        for _ in range(np.random.randint(*self.shape_count_range)):
            xy, width = self._random_line()
            pen.width = width
            draw.line(xy, pen)
        draw.flush()
        return {'image': img}

    def get(self):
        return self.draw_alias()['image']

def test_random_line_dataset():
    dataset = LineGenerator()
    for i in range(10):
        sample = dataset[i]
        img = Image.fromarray(sample['image'].numpy().astype('uint8'), 'RGB')
        img.show()


if __name__=="__main__":
    test_random_line_dataset()
