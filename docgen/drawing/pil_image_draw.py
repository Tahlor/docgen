import random
from PIL import Image, ImageDraw, ImageFont

class CustomImageDraw(ImageDraw.ImageDraw):
    def __init__(self, *args, max_line_thickness=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_line_thickness = max_line_thickness

    def rectangle(self, xy, fill=None, outline=None, *, styles=None, colors=None, thicknesses=None):
        """
        draw: ImageDraw.Draw object, the drawing context
        xy: tuple, coordinates of the rectangle's bounding box
        fill: color, the color to fill the rectangle, None for no filling
        outline: color, the color of the rectangle outline (default: black)
        width: int, the width of the outline (default: 1)
        styles: list, the line styles of the four sides of the rectangle
        colors: list, the grayscale colors of the four sides of the rectangle
        thicknesses: list, the thicknesses of the four sides of the rectangle
        """

        if outline is None:
            outline = 'black'

        if colors is None:
            colors = [random.randint(0, 255) for _ in range(4)]

        if thicknesses is None:
            thicknesses = [random.randint(1, self.max_line_thickness) for _ in range(4)]

        if fill is not None:
            super().rectangle(xy, fill=fill)

        x1, y1, x2, y2 = xy
        all_points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        for line_idx in range(4):
            x1, y1 = all_points[line_idx]
            x2, y2 = all_points[(line_idx + 1) % 4]

            if line_idx == 0:  # Top side
                x1 -= thicknesses[(line_idx-1) % 4] // 2
                x2 += thicknesses[(line_idx+1) % 4] // 2
            elif line_idx == 1:  # Right side
                y1 -= (thicknesses[(line_idx-1) % 4]-1) // 2
                y2 += (thicknesses[(line_idx+1) % 4]-1) // 2
            elif line_idx == 2:  # Bottom side
                x1 += thicknesses[(line_idx-1) % 4] // 2
                x2 -= thicknesses[(line_idx+1) % 4] // 2
            elif line_idx == 3:  # Left side
                y1 += (thicknesses[(line_idx-1) % 4]-1) // 2
                y2 -= (thicknesses[(line_idx+1) % 4]-1) // 2

            self.line([(x1, y1), (x2, y2)], fill=colors[line_idx], width=thicknesses[line_idx])
