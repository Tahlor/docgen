import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from docgen.layoutgen.segmentation_dataset.layer_generator.gen import Gen

class LineGenerator(Gen):
    def __init__(self, img_size=(448, 448), shape_count_range=(2, 5), line_thickness_range=(1, 5), max_color_brightness=0.8):
        self.size = img_size
        self.shape_count_range = shape_count_range
        self.line_thickness_range = line_thickness_range
        self.max_color_brightness = max_color_brightness

    def _random_line(self, pixel_offset=8):
        orientation = np.random.choice(['horizontal', 'vertical'])
        line_style = np.random.choice(['solid', 'dotted', 'dashed'])

        if orientation == 'horizontal':
            y = np.random.randint(0, self.size[1])
            y_offset = np.random.randint(-pixel_offset, pixel_offset)
            start_point = (np.random.randint(0, self.size[0]), y)
            end_point = (np.random.randint(0, self.size[0]), y + y_offset)
        else:
            x = np.random.randint(0, self.size[0])
            x_offset = np.random.randint(-pixel_offset, pixel_offset)
            start_point = (x, np.random.randint(0, self.size[1]))
            end_point = (x + x_offset, np.random.randint(0, self.size[1]))

        line_thickness = np.random.randint(*self.line_thickness_range)

        return {"start": start_point, "end": end_point, "width": line_thickness, "style": line_style}

    def draw_lines(self, aliasing=True):
        fig_size = (self.size[0] / 100, self.size[1] / 100)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        ax.invert_yaxis()  # Invert y-axis to match the coordinate system of PIL
        ax.axis('off')

        for _ in range(np.random.randint(*self.shape_count_range)):
            line_params = self._random_line()
            ax.plot([line_params['start'][0], line_params['end'][0]],
                    [line_params['start'][1], line_params['end'][1]],
                    linewidth=line_params['width'],
                    linestyle=line_params['style'],
                    color=np.random.rand(3) * self.max_color_brightness)

        fig.canvas.draw()

        if aliasing:
            return np.array(fig.canvas.renderer.buffer_rgba())
        else:
            return np.array(fig.canvas.renderer._renderer)

    def draw_alias(self):
        img = self.convert_to_pil(self.draw_lines(aliasing=True))
        return {"image": img}

    def draw_no_alias(self):
        img = self.convert_to_pil(self.draw_lines(aliasing=False))
        return {"image": img}

    def convert_to_pil(self, img_data):
        img = Image.fromarray(img_data, 'RGBA')
        img = img.convert('RGB')
        plt.close()
        return img

    def get(self, aliasing=True):
        img_data = self.draw_lines(aliasing)
        img = self.convert_to_pil(img_data)
        return img

def random_line_dataset_test():
    dataset = LineGenerator()
    for i in range(3):
        img = dataset[i]
        img.show()


if __name__=="__main__":
    random_line_dataset_test()