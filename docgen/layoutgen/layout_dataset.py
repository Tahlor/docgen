from torch.utils.data import Dataset, DataLoader, IterableDataset
from docgen.degradation.degrade import degradation_function_composition
from docgen.utils import handler
from hwgen.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
from pathlib import Path
import socket
from torch.utils.data import Dataset, DataLoader, IterableDataset

TESTING = False

class LayoutDataset(Dataset):
    """ Generates layouts with handwriting
        Reason for using a Dataset is to use the multiprocessing DataLoader
    """
    def __init__(self, layout_generator,
                 render_text_pairs,
                 length=100000,
                 degradation_function=None,
                 output_path=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.layout_generator = layout_generator
        self.render_text_pairs = render_text_pairs
        self.length = length
        self.degradation_function = degradation_function
        self.output_path = output_path

    def __len__(self):
        return self.length

    @handler(testing=TESTING, return_on_fail=(None, None, None))
    def __getitem__(self, i):
        name, ocr, image = self.layout_generator.make_one_image(i)

        if self.degradation_function:
            image = self.degradation_function(image)
        if self.output_path:
            name = f"{i:07.0f}"
            save_path = self.output_path / f"{name}.jpg"
            image.save(save_path)
        return name, ocr, image

    @staticmethod
    def collate_fn(batch):
        return batch

if __name__ == "__main__":
    from textgen.unigram_dataset import Unigrams
    from docgen.rendertext.render_word import RenderImageTextPair
    from docgen.layoutgen.layoutgen import LayoutGenerator

    HWR_FILES = None # folder with handwriting .npy files
    UNIGRAMS_PATH = None # folder with unigrams.csv file
    NUMBER_OF_DOCUMENTS=100
    DATASETS = Path("./temp")
    OUTPUT = DATASETS / "FRENCH_BMD_LAYOUTv3"

    lg = LayoutGenerator()
    words = Unigrams(csv_file=UNIGRAMS_PATH, newline_freq=0)

    renderer = SavedHandwritingRandomAuthor(
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
                                   length=NUMBER_OF_DOCUMENTS)
    for i in layout_dataset:
        pass
