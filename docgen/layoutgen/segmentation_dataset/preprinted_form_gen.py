from docgen.layoutgen.segmentation_dataset.semantic_segmentation import FlattenPILGenerators
from docgen.layoutgen.segmentation_dataset.grid_gen import GridGenerator
from docgen.layoutgen.segmentation_dataset.line_gen import LineGenerator
from docgen.layoutgen.segmentation_dataset.box_gen import BoxGenerator
from docgen.layoutgen.segmentation_dataset.gen import Gen
class PreprintedFormElementGenerator(Gen):
    def __init__(self, img_size=(448, 448)):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.img_size = img_size
        self.grid_generator = GridGenerator(img_size=img_size)
        self.line_generator = LineGenerator(img_size=img_size)
        self.box_generator = BoxGenerator(img_size=img_size)
        self.combined_generator = FlattenPILGenerators([self.grid_generator, self.line_generator, self.box_generator],
                                                       img_size=img_size)

    def get(self, img_size=None):
        return self.combined_generator.get()
