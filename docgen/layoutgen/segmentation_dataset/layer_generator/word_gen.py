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
from textgen.number_generator import RandomNumberGenerator
from textgen.combined_generator import CombinedGenerator
from textgen.fonts.font_sampler import FontSampler

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

class BoxFillerGen(Gen):
    """ Generates words in a box, with a font size, and a number of words

    """
    def __init__(self, img_size=(512,512),
                 font_size_rng=(8,50),
                  word_count_rng=(10,20),
                  return_format="dict",
                  **kwargs
        ):

        self.width, self.height = self.img_size = split_strs_into_tuple(img_size)
        self.font_size_rng = split_strs_into_tuple(font_size_rng)
        self.word_count_rng = split_strs_into_tuple(word_count_rng)
        self.return_format = return_format
        self.force = False

    def _get(self, img_size=None):
        if img_size is None:
            img_size = self.img_size
        font_size = random.randint(*self.font_size_rng)
        img = Image.new("RGB", img_size, (255,255,255))
        use_random_box = self.force=="random" or (self.force is None and random.random() > 0.4)
        self.force = None

        if use_random_box:
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
        img_dict = self._get(img_size=img_size)
        img_dict["image"] = img_dict.pop("img")
        if self.return_format == "dict":
            return img_dict
        else:
            img = img_dict.pop(["img"])
            img.metadata = img_dict
            return img

    def next_one_random_placement(self):
        self.force = "random"

class PrintedTextGenerator(BoxFillerGen):
    """
            saved_fonts_folder = Path(r"G:/s3/synthetic_data/resources/fonts")
    """
    def __init__(self, img_size=(512,512),
                 font_size_rng=(8, 50),
                 word_count_rng=(10, 20),
                 saved_fonts_folder=None,
                 range_before_font_change=(1,10),
                 probability_of_number=.1,
                 **kwargs):
        super().__init__(img_size, font_size_rng=font_size_rng, word_count_rng=word_count_rng, **kwargs)
        unigrams = get_resource(package_name="textgen", resource_relative_path="/datasets/unigram_freq.csv")
        words_dataset = Unigrams(csv_file=unigrams)
        number_generator = RandomNumberGenerator()

        if saved_fonts_folder is not None:
            saved_fonts_folder = Path(saved_fonts_folder)
            clear_fonts_path = Path(saved_fonts_folder) / "clear_fonts.csv"
            font_characteristics_csv_file = saved_fonts_folder / "ALL_fonts.csv"

            font_class_files = [
                saved_fonts_folder / "FONT_SAMPLES/typewriter_fonts.pkl",
                saved_fonts_folder / "FONT_SAMPLES/APPROVED_fonts.pkl",
            ]
            font_sampler = FontSampler(saved_fonts_folder / "fonts",
                                       font_class_files,
                                       weights=[10, 90],
                                       csv_file=font_characteristics_csv_file)
            font_sampler.sample()
        else:
            font_sampler = clear_fonts_path = None

        combined_gen = CombinedGenerator(
            generators=[words_dataset, number_generator],
            probabilities=[1-probability_of_number, probability_of_number]
        )

        self.renderer = RenderWordConsistentFont(format="numpy",
                                       font_folder=saved_fonts_folder,
                                       clear_font_csv_path=clear_fonts_path,
                                       range_before_font_change=range_before_font_change,
                                       probability_of_all_caps_override=0.2,
                                        font_sampler=font_sampler,
                        )

        self.render_text_pair = RenderImageTextPair(self.renderer, combined_gen, renderer_text_key="raw_text")
        self.filler = BoxFiller(img_text_word_dict=self.render_text_pair,
                                random_word_idx=True)

class HWGenerator(BoxFillerGen):
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
        box_dict["image"].show()

        box_dict = printed_gen.get()
        box_dict["image"].show()

