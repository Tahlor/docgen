from docgen.tablegen.grid import Grid
from docgen.layoutgen.segmentation_dataset.layer_generator.gen import Gen

class GridGenerator(Gen):
    def __init__(self, img_size=(512,512)):
        """

        Args:
            img_size: H,W
        """
        height, width = img_size
        self.img_size = img_size
        self.grid = self.generator = Grid(width, height, row_height_range=(40, 120), col_width_range=(40, 400))

    def get(self, img_size=None):
        if img_size is None:
            img_size = self.img_size
        bbox = self.get_random_bbox(img_size=img_size)
        img = self.grid.get((bbox.width, bbox.height),)
        img = self.composite_pil_from_blank(img, img_size, bbox[:2])
        return img


if __name__ == "__main__":
    gen = GridGenerator()
    for i in range(2):
        img = gen.get()
        img.show()
