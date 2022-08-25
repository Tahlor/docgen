from torch.utils.data import Dataset, DataLoader, IterableDataset
from pdfgen.degradation.degrade import degradation_function_composition
from pdfgen.utils import handler

TESTING=False

class LayoutDataset(Dataset):
    def __init__(self, layout_generator, render_text_pairs, output_path, lenth=100000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layout_generator = layout_generator
        self.render_text_pairs = render_text_pairs
        self.output_path = output_path
        self.lenth = lenth

    def make_one_image(self, i):
        name = f"{i:07.0f}"
        layout = self.layout_generator.generate_layout()
        image = self.layout_generator.render_text(layout, self.render_text_pairs)
        save_path = self.output_path / f"{name}.jpg"
        image = degradation_function_composition(image)
        image.save(save_path)
        return name, self.layout_generator.create_ocr(layout, id=i, filename=name)

    def __len__(self):
        return self.lenth

    @handler(testing=TESTING, return_on_fail=(None, None))
    def __getitem__(self, i):
        return self.make_one_image(i)

    @staticmethod
    def collate_fn(batch):
        return batch

