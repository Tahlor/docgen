from docgen.tablegen.grid import Grid
from torch.utils.data import Dataset

from torch.utils.data import Dataset

class GridDataset(Dataset):
    def __init__(self, num_samples, grid_width, grid_height, row_height_range, col_width_range,
                 header_rows, subdivided_cells, merged_cells, extra_cells):
        self.num_samples = num_samples
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.row_height_range = row_height_range
        self.col_width_range = col_width_range
        self.header_rows = header_rows
        self.subdivided_cells = subdivided_cells
        self.merged_cells = merged_cells
        self.extra_cells = extra_cells
        self.grid = Grid(self.grid_width, self.grid_height, self.row_height_range, self.col_width_range)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self.grid.randomize_cells(self.header_rows, self.subdivided_cells, self.merged_cells, self.extra_cells)
        image = self.grid.show()
        cell_count = self.grid.get_cells()
        return image, cell_count

