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
    def __init__(self, img_size=(448, 448), shape_count_range=(2, 5), line_thickness_range=(1, 5)):
        super().__init__()
        self.size = img_size
        self.shape_count_range = shape_count_range
        self.line_thickness_range = line_thickness_range

    def _random_line(self):
        start_point = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        end_point = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        line_thickness = np.random.randint(*self.line_thickness_range)
        return {"xy": (start_point, end_point), "width": line_thickness}

    def _random_line(self, pixel_offset=8):
        orientation = np.random.choice(['horizontal', 'vertical'])

        if orientation == 'horizontal':
            # For nearly horizontal lines, keep the y-coordinates close
            y = np.random.randint(0, self.size[1])
            y_offset = np.random.randint(-pixel_offset, pixel_offset)  # This controls how "nearly" horizontal the line is
            start_point = np.random.randint(0, self.size[0]), y
            end_point = np.random.randint(0, self.size[0]), y + y_offset
        else:
            # For nearly vertical lines, keep the x-coordinates close
            x = np.random.randint(0, self.size[0])
            x_offset = np.random.randint(-pixel_offset, pixel_offset)  # This controls how "nearly" vertical the line is
            start_point = x, np.random.randint(0, self.size[1])
            end_point = x + x_offset, np.random.randint(0, self.size[1])

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
    for i in range(3):
        img = dataset[i]
        img.show()


if __name__=="__main__":
    test_random_line_dataset()
