from docgen.layoutgen.segmentation_dataset.layer_generator.line_gen import LineGenerator
import numpy as np
import aggdraw
from PIL import Image


class BoxGenerator(LineGenerator):
    def __init__(self, *args, shape_count_range=(1,4), **kwargs):
        super().__init__(*args, shape_count_range=shape_count_range, **kwargs)

    def _random_box(self):
        upper_left_point = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        lower_right_point = np.random.randint(upper_left_point[0], self.size[0]), np.random.randint(upper_left_point[1], self.size[1])
        line_thickness = np.random.randint(*self.line_thickness_range)
        return {"xy": (upper_left_point, lower_right_point), "width": line_thickness}

    def draw_no_alias(self):
        img = Image.new('RGB', self.size, "white")
        draw = aggdraw.Draw(img)
        #pen = aggdraw.Pen('black')
        for _ in range(np.random.randint(*self.shape_count_range)):
            box_dict = self._random_box()
            xy, width = box_dict["xy"], box_dict["width"]
            xy = xy[0][0], xy[0][1], xy[1][0], xy[1][1]
            pen = aggdraw.Pen(color=self.random_grayscale(max=self.max_color_brightness), width=width)
            draw.rectangle(xy, pen)
        draw.flush()
        return {'image': img}

    def draw_alias(self):
        return self.draw_no_alias()

    def get(self):
        if self.aliasing:
            return self.draw_alias()["image"]
        else:
            return self.draw_no_alias()["image"]


def box_dataset_test():
    dataset = BoxGenerator()
    for i in range(10):
        sample = dataset[i]
        if isinstance(sample,dict):
            sample = sample["image"]
        #img = Image.fromarray(sample.numpy().astype('uint8'), 'RGB')
        img = sample
        img.show()

if __name__=="__main__":
    box_dataset_test()