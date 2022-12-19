from torch.utils.data import Dataset, DataLoader, IterableDataset
from docgen.degradation.degrade import degradation_function_composition
from docgen.utils import handler
from handwriting.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
from pathlib import Path
import socket
from torch.utils.data import Dataset, DataLoader, IterableDataset

TESTING=True

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

if __name__ == "__main__":
    from textgen.unigram_dataset import Unigrams
    from docgen.rendertext.render_word import RenderImageTextPair
    from docgen.layoutgen.layoutgen import LayoutGenerator

    HWR_FILES = Path("/home/taylor/anaconda3/datasets/HANDWRITING_WORD_DATA/")
    NUMBER_OF_DOCUMENTS=100
    if socket.gethostname()  == "G1G2Q13":
        UNIGRAMS = r"../../textgen/textgen/datasets/unigram_freq.csv"
    else:
        UNIGRAMS = r"/media/data/GitHub/textgen/textgen/datasets/unigram_freq.csv"

    DATASETS = Path("./temp")
    OUTPUT = DATASETS / "FRENCH_BMD_LAYOUTv3"

    lg = LayoutGenerator()
    words = Unigrams(csv_file=UNIGRAMS, newline_freq=0)

    renderer = SavedHandwritingRandomAuthor (
        format="PIL",
        dataset_root=HWR_FILES,
        # dataset_path=HWR_FILE,
        random_ok=True,
        conversion=None,  # lambda image: np.uint8(image*255)
        font_size=32
    )

    render_text_pair = RenderImageTextPair(renderer, words)
    layout_dataset = LayoutDataset(layout_generator=lg,
                                   render_text_pairs=render_text_pair,
                                   output_path=OUTPUT,
                                   lenth=NUMBER_OF_DOCUMENTS)
    for i in layout_dataset:
        pass
