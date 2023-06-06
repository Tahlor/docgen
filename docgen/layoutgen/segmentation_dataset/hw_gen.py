import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import site
import pkg_resources
from docgen.pdf_edit import *
from PIL import Image
from docgen.pdf_edit import PDF
from textgen.unigram_dataset import Unigrams
from textgen.rendertext.render_word import RenderWordFont
from hwgen.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
import numpy as np
from docgen.utils import utils
from hwgen.data.utils import display
from hwgen.resources import HandwritingResourceManager
from textgen.rendertext.render_word import RenderWordFont, RenderImageTextPair
from docgen.bbox import BBox

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

unigrams = get_resource(package_name="textgen", resource_relative_path="/datasets/unigram_freq.csv")
saved_fonts_folder = Path(site.getsitepackages()[0]) / "rendertext" / "fonts"
saved_fonts_folder = Path(r"G:/s3/synthetic_data/resources/fonts")
clear_fonts_path = saved_fonts_folder / "clear_fonts.csv"
saved_hw_folder = Path(site.getsitepackages()[0]) / r"hwgen/resources/generated"

words_dataset = Unigrams(csv_file=unigrams)
renderer = SavedHandwritingRandomAuthor(format="numpy",
                            dataset_root=saved_hw_folder,
                            random_ok=True,
                            conversion=lambda image: np.uint8(image * 255)
                            )

render_text_pair = RenderImageTextPair(renderer, words_dataset,
                                       renderer_text_key="raw_text")

filler = BoxFiller(img_text_word_dict=render_text_pair,
                        random_word_idx=True)

# create a bbox
bbox = BBox("ul", [3, 3, 350, 50])
box_dict = filler.randomly_fill_box_with_words(bbox,
                                               max_words=random.randint(1, 10),
                                               )
box_dict = filler.fill_box(bbox)

renderer = RenderWordFont(format="numpy",
                          font_dir=saved_fonts_folder,
                          clear_font_csv=clear_fonts_path)
X = renderer.render_word("word")
plt.imshow(X["image"], cmap="gray")
plt.show()
