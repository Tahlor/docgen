import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import site
import pkg_resources
from docgen.pdf_edit import *
from PIL import Image
from textgen.unigram_dataset import Unigrams
from hwgen.data.saved_handwriting_dataset import SavedHandwritingRandomAuthor
from textgen.rendertext.render_word import RenderWordFont, RenderImageTextPair, RenderWordConsistentFont
from docgen.bbox import BBox
from docgen.layoutgen.segmentation_dataset.layer_generator.gen import Gen

# given a image size:
    # for i in random:
        # pick a font size / hw
        # pick a font / random pre-generated hw word
        # pick some box smaller than this image, larger than the font size
        # choose some number of words
        # fill in

def get_resource(package_name, resource_relative_path):
    """ If the resource is part of a package (i.e., hosted on Github and downloaded on install)
        e.g., unigrams

    Args:
        package_name:
        resource_relative_path:

    Returns:

    """
    resource_path = '/'.join(('datasets', 'unigram_freq.csv'))
    return pkg_resources.resource_filename(package_name, resource_path)

def split_strs_into_tuple(strs):
    if isinstance(strs, str):
        return tuple([int(x) for x in strs.split(",")])
    return strs

class WordGenerator(Gen):
    def __init__(self, img_size=(512,512),
                 font_size_rng=(8,50),
                  word_count_rng=(10,20),
                  **kwargs
        ):

        self.width, self.height = self.img_size = split_strs_into_tuple(img_size)
        self.font_size_rng = split_strs_into_tuple(font_size_rng)
        self.word_count_rng = split_strs_into_tuple(word_count_rng)

    def _get(self, img_size=None):
        if img_size is None:
            img_size = self.img_size
        font_size = random.randint(*self.font_size_rng)
        img = Image.new("RGB", img_size, (255,255,255))
        if random.random() > 0.4:
            bbox = BBox("ul", [0, 0, *img_size])
            box_dict = self.filler.randomly_fill_box_with_words(bbox, img=img,
                                                       max_words=random.randint(*self.word_count_rng),
                                                       allow_overlap=False,
                                                       font_size_override_range=self.font_size_rng
                                                       )
        else:
            bbox = self.get_random_bbox(img_size=img_size, font_size=font_size)
            box_dict = self.filler.fill_box(bbox, img=img,
                                            font_size_override=font_size)

        return box_dict

    def get(self, img_size=None):
        return self._get(img_size=img_size)["img"]


class PrintedTextGenerator(WordGenerator):
    """
            saved_fonts_folder = Path(r"G:/s3/synthetic_data/resources/fonts")
    """
    def __init__(self, img_size=(512,512),
                 font_size_rng=(8, 50),
                 word_count_rng=(10, 20),
                 saved_fonts_folder=None, **kwargs):
        super().__init__(img_size, font_size_rng=font_size_rng, word_count_rng=word_count_rng, **kwargs)
        unigrams = get_resource(package_name="textgen", resource_relative_path="/datasets/unigram_freq.csv")
        clear_fonts_path = Path(saved_fonts_folder) / "clear_fonts.csv"

        words_dataset = Unigrams(csv_file=unigrams)

        self.renderer = RenderWordFont(format="numpy",
                                       font_folder=saved_fonts_folder,
                                       clear_font_csv_path=clear_fonts_path)

        self.render_text_pair = RenderImageTextPair(self.renderer, words_dataset, renderer_text_key="raw_text")
        self.filler = BoxFiller(img_text_word_dict=self.render_text_pair,
                                random_word_idx=True)

class HWGenerator(WordGenerator):
    def __init__(self, img_size=(512,512),
                 font_size_rng=(8, 50),
                 word_count_rng=(10, 20),
                 saved_hw_folder=None, **kwargs):
        """ Generates handwritten text in real time OR combine them from pre-generated .npy files

        Args:
            img_size:
            font_size_rng:
            word_count_rng:
            saved_hw_folder:
            **kwargs:
        """
        super().__init__(img_size, font_size_rng=font_size_rng, word_count_rng=word_count_rng,  **kwargs)
        unigrams = get_resource(package_name="textgen", resource_relative_path="/datasets/unigram_freq.csv")
        if saved_hw_folder is None:
            saved_hw_folder = Path(site.getsitepackages()[0]) / r"hwgen/resources/generated"

        words_dataset = Unigrams(csv_file=unigrams)

        self.renderer_hw = SavedHandwritingRandomAuthor(format="PIL",
                                                        dataset_root=saved_hw_folder,
                                                        random_ok=True,
                                                        conversion=None)

        self.render_text_pair = RenderImageTextPair(self.renderer_hw, words_dataset, renderer_text_key="raw_text")
        self.filler = BoxFiller(img_text_word_dict=self.render_text_pair,
                                random_word_idx=True)


if __name__=="__main__":
    hwgen = HWGenerator((512,256))
    printed_gen = PrintedTextGenerator((512,256))

    for i in range(3):
        box_dict = hwgen.get()
        box_dict["img"].show()

        box_dict = printed_gen.get()
        box_dict["img"].show()

