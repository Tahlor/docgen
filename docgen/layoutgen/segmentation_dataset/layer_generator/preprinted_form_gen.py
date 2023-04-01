from docgen.layoutgen.segmentation_dataset.semantic_segmentation import FlattenPILGenerators
from docgen.layoutgen.segmentation_dataset.layer_generator.grid_gen import GridGenerator
from docgen.layoutgen.segmentation_dataset.layer_generator.line_gen import LineGenerator
from docgen.layoutgen.segmentation_dataset.layer_generator.box_gen import BoxGenerator
from docgen.layoutgen.segmentation_dataset.layer_generator.gen import Gen
class PreprintedFormElementGenerator(Gen):
    def __init__(self, img_size=(448, 448)):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        self.img_size = img_size
        self.grid_generator = GridGenerator(img_size=img_size)
        self.line_generator = LineGenerator(img_size=img_size, shape_count_range=(2,10))
        self.box_generator = BoxGenerator(img_size=img_size)
        self.combined_generator = FlattenPILGenerators([self.grid_generator, self.line_generator, self.box_generator],
                                                       img_size=img_size,
                                                       dataset_probabilities=[.1, .9, .3])

    def get(self, img_size=None):
        return self.combined_generator.get()


if __name__ == "__main__":
    gen = PreprintedFormElementGenerator()
    for i in range(2):
        img = gen.get()
        img.show()