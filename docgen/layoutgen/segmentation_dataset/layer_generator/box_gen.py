from docgen.layoutgen.segmentation_dataset.layer_generator.line_gen import LineGenerator
import numpy as np
import aggdraw
from PIL import Image


class BoxGenerator(LineGenerator):
    def _random_box(self):
        upper_left_point = np.random.randint(0, self.size[0]), np.random.randint(0, self.size[1])
        lower_right_point = np.random.randint(upper_left_point[0], self.size[0]), np.random.randint(upper_left_point[1], self.size[1])
        line_thickness = np.random.randint(*self.line_thickness_range)
        return {"xy": (upper_left_point, lower_right_point), "width": line_thickness}

    def draw_no_alias(self):
        img = Image.new('RGB', self.size, "white")
        draw = aggdraw.Draw(img)
        pen = aggdraw.Pen(self.random_grayscale(max=self.max_color_brightness))
        for _ in range(np.random.randint(*self.shape_count_range)):
            xy, width = self._random_box()
            pen.width = width
            draw.rectangle(xy, pen)
        draw.flush()
        return {'image': img}

    def get(self):
        return self.draw_alias()["image"]


def box_dataset_test():
    dataset = LineGenerator()
    for i in range(10):
        sample = dataset[i]
        if isinstance(sample,dict):
            sample = sample["image"]
        #img = Image.fromarray(sample.numpy().astype('uint8'), 'RGB')
        img = sample
        img.show()

if __name__=="__main__":
    box_dataset_test()