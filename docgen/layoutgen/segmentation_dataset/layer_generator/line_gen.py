import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from docgen.layoutgen.segmentation_dataset.layer_generator.gen import Gen

DASH_STYLES = ['-', ':', '--', '-.']
DASH_WEIGHTS = [10,30,4,1]
DASH_WEIGHTS = np.array(DASH_WEIGHTS) / np.sum(DASH_WEIGHTS)

CAP_STYLES = {'butt', 'projecting', 'round'}
JOINSTYLES = {'miter', 'round', 'bevel'}

def make_int_into_range(x):
    if isinstance(x, int):
        return (x, x+1)
    else:
        return x

class LineGenerator(Gen):
    def __init__(self,
                 img_size=(448, 448),
                 shape_count_range=(3, 8),
                 line_thickness_range=(1, 5),
                 max_color_brightness=0.8,
                 probability_grayscale=0.5,
                 aliasing=True):
        self.size = img_size
        self.shape_count_range = make_int_into_range(shape_count_range)
        self.line_thickness_range = make_int_into_range(line_thickness_range)
        self.max_color_brightness = make_int_into_range(max_color_brightness)
        self.aliasing = aliasing
        self.probability_grayscale = probability_grayscale

    @staticmethod
    def gaussian_int_with_cap(minimum, maximum, std=None, precision=1):
        maximum = int(maximum * precision)
        minimum = int(minimum * precision)
        rng = max(maximum - minimum, 1)
        mean = rng / 2
        if std is None:
            std = rng / 4
        sample = (int(np.random.normal(mean, std)) % rng) + minimum
        return sample / precision

    def _random_line(self, pixel_offset=8):
        orientation = np.random.choice(['horizontal', 'vertical'])
        if np.random.random() < .5:
            line_style = np.random.choice(DASH_STYLES, p=DASH_WEIGHTS / np.sum(DASH_WEIGHTS))
            dashes = None
        else:
            line_style = None
            dashes = self.random_dash_and_space_length()

        if orientation == 'horizontal':
            y = np.random.randint(0, self.size[1])
            y_offset = self.gaussian_int_with_cap(-pixel_offset, pixel_offset)
            start_x = np.random.randint(0, self.size[0])
            end_x = start_x + np.random.randint(self.size[0] // 4, 3 * self.size[0] // 4)
            start_point = (start_x, y)
            end_point = (end_x % self.size[0], y + y_offset)
        else:
            x = np.random.randint(0, self.size[0])
            x_offset = self.gaussian_int_with_cap(-pixel_offset, pixel_offset)
            start_point = (x, np.random.randint(0, self.size[1]))
            end_point = (x + x_offset, np.random.randint(0, self.size[1]))

        line_thickness = self.random_line_thickness()
        out = {"start": start_point,
                "end": end_point,
                "linewidth": line_thickness,
                "color": self.pick_color() * self.max_color_brightness,
                "dash_capstyle": np.random.choice(list(CAP_STYLES)),
                "dash_joinstyle": np.random.choice(list(JOINSTYLES)),
                "solid_capstyle": np.random.choice(list(CAP_STYLES)),
                "solid_joinstyle": np.random.choice(list(JOINSTYLES)),
                }
        if dashes is not None:
            out["dashes"] = dashes
        else:
            out["linestyle"] = line_style
        return out

    def random_line_thickness(self):
        return self.exponential(self.line_thickness_range, exponential_factor=1.4)

    def random_dash_and_space_length(self):
        dash_length = self.gaussian_int_with_cap(1,8, std=None)
        space_length = self.gaussian_int_with_cap(1, maximum=dash_length, std=None)
        return dash_length, space_length

    def exponential(self, rng, exponential_factor=1.0):
        exp_number = np.random.exponential(exponential_factor)
        random_value = int(exp_number) % (rng[1] - rng[0]) + rng[0]
        return random_value

    def draw_lines(self, aliasing=True):
        fig_size = (self.size[0] / 100, self.size[1] / 100)
        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_xlim(0, self.size[0])
        ax.set_ylim(0, self.size[1])
        ax.invert_yaxis()  # Invert y-axis to match the coordinate system of PIL
        ax.axis('off')

        for _ in range(np.random.randint(*self.shape_count_range)):
            line_params = self._random_line()
            start = line_params.pop('start')
            end = line_params.pop('end')
            ax.plot([start[0], end[0]],
                    [start[1], end[1]],
                    **line_params
                    )
        fig.canvas.draw()
        if aliasing:
            return np.array(fig.canvas.renderer.buffer_rgba())
        else:
            return np.array(fig.canvas.renderer._renderer)

    def pick_color(self):
        if np.random.rand() < self.probability_grayscale:
            grayscale_value = np.random.rand()
            return np.tile(grayscale_value, 3)
        else:
            return np.random.rand(3)

    def convert_to_pil(self, img_data):
        img = Image.fromarray(img_data, 'RGBA')
        img = img.convert('RGB')
        plt.close()
        return img

    def get(self):
        img_data = self.draw_lines(self.aliasing)
        img = self.convert_to_pil(img_data)
        return img


def random_line_dataset_test():
    for aliasing in [True, False]:
        dataset = LineGenerator(line_thickness_range=1, aliasing=aliasing)
        for i in range(1):
            img = dataset[i]
            img.show()


if __name__=="__main__":
    random_line_dataset_test()