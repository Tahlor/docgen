import random
from PIL import Image, ImageDraw
from docgen.drawing.pil_image_draw import CustomImageDraw

class Grid:
    def __init__(self, width,
                 height,
                 row_height_range=(10, 20),
                 col_width_range=(10, 20),

                 ):
        self.width = width
        self.height = height
        self.row_height_range = row_height_range
        self.col_width_range = col_width_range
        self.image = Image.new("RGB", (width, height), "white")
        self.draw = CustomImageDraw(self.image)
        self.cells = []
        """
        """

    def randomize_cells(self, header_rows=0, subdivided_cells=0, merged_cells=0, extra_cells=0):
        total_height = 0
        while total_height < self.height:
            row_height = random.randint(*self.row_height_range)
            if total_height + row_height > self.height:
                row_height = self.height - total_height
            total_width = 0
            while total_width < self.width:
                cell_width = random.randint(*self.col_width_range)
                if total_width + cell_width > self.width:
                    cell_width = self.width - total_width
                self.cells.append(((total_width, total_height), (total_width + cell_width, total_height + row_height)))
                total_width += cell_width
            total_height += row_height

        # Merged cells
        for _ in range(merged_cells):
            idx = random.randint(0, len(self.cells) - 1)
            next_idx = (idx + 1) % len(self.cells)
            self.cells[idx] = (self.cells[idx][0], (self.cells[next_idx][1][0], self.cells[idx][1][1]))

        # Subdivided cells
        for _ in range(subdivided_cells):
            idx = random.randint(0, len(self.cells) - 1)
            x1, y1 = self.cells[idx][0]
            x2, y2 = self.cells[idx][1]
            self.cells[idx] = ((x1, y1), (x2, (y1 + y2) // 2))
            self.cells.append(((x1, (y1 + y2) // 2), (x2, y2)))

        # Extra cells
        for _ in range(extra_cells):
            idx = random.randint(0, len(self.cells) - 1)
            x1, y1 = self.cells[idx][0]
            x2, y2 = self.cells[idx][1]
            cell_width = random.randint(*self.col_width_range)
            self.cells.append(((x2, y1), (x2 + cell_width, y2)))

    def add_checkboxes_or_circles(self, count):
        for _ in range(count):
            cell = random.choice(self.cells)
            x1, y1 = cell[0]
            x2, y2 = cell[1]
            margin_x = random.randint(2, max(2, (x2 - x1) // 4))  # Margin for x-axis
            margin_y = random.randint(2, max(2, (y2 - y1) // 4))  # Margin for y-axis
            checkbox_or_circle = random.choice(['checkbox', 'circle'])
            shape_count = random.randint(1, 4)  # Random number of shapes to add in a cell
            for _ in range(shape_count):
                if checkbox_or_circle == 'checkbox':
                    size = random.randint(min(5, (x2 - x1) // 2), min(10, (x2 - x1) // 2))
                    self.draw.rectangle(((x1 + margin_x, y1 + margin_y),
                                         (x1 + margin_x + size, y1 + margin_y + size)),
                                        outline="black")
                    margin_x += size + 2  # Update margin for the next shape
                else:  # circle
                    radius = random.randint(min(5, (x2 - x1) // 4), min(10, (x2 - x1) // 4))
                    self.draw.ellipse(((x1 + margin_x, y1 + margin_y),
                                       (x1 + margin_x + 2 * radius, y1 + margin_y + 2 * radius)),
                                      outline="black")
                    margin_x += 2 * radius + 2  # Update margin for the next shape

    def draw_cells(self):
        for cell in self.cells:
            self.draw.rectangle(cell, outline="black")

    def show(self):
        self.draw_cells()
        self.add_checkboxes_or_circles(10)
        self.image.show()

    def get_cells(self):
        return self.cells

if __name__ == '__main__':
    # Example usage:
    grid = Grid(800, 600, row_height_range=(20, 50), col_width_range=(50, 100))
    grid.randomize_cells(header_rows=2, subdivided_cells=5, merged_cells=5, extra_cells=2)
    grid.show()
    print(grid.get_cells())


