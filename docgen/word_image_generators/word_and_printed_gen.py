import os
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'
from hwgen.data.hw_generator import HWGenerator
from pathlib import Path
from typing import List, Tuple, Union, Dict
from PIL import Image
from random import randint, choice
from hwgen.data.saved_handwriting_dataset import SavedHandwritingRandomAuthor
from textgen.rendertext.render_word import RenderWordFont, RenderImageTextPair, RenderWordConsistentFont
from typing import List, Tuple, Union, Iterator
from PIL import Image
from hwgen.data.utils import show,display
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class MultiGeneratorFromDict:
    def __init__(self,
                 generators: List[Union[SavedHandwritingRandomAuthor, RenderWordFont]],
                 word_count_ranges: List[Tuple[int, int]],
                 spacing_generator_range: Tuple[int,int]=(0,7),
                 special_args_to_update_object_attributes_that_use_this_generator: List[Tuple[Dict]] = None,
                 ):
        """
        Initialize MultiGenerator. Generates both handwriting and font

        Args:
            generators (List[Union[SavedHandwritingRandomAuthor, RenderWordFont]]): A list of text word_image_generators.
            word_count_ranges (List[Tuple[int, int]]): A list of tuples specifying the range of the number of words for each generator.
            spacing_generator_range (int): Spacing between different word_image_generators' outputs (a multiple of font height,
                mimicking the spacing between pre-printed font text and a fill-in-the-blank handwriting)
            special_args_to_update_object_attributes_that_use_this_generator (List[Tuple[Dict]]):
                A list of dictionaries containing the attributes to update an object that is using the generator
                For instance, we may want more variation when using HW than Font

        """
        assert len(generators) == len(word_count_ranges), "Each generator must have a corresponding word_count_range"
        self.generators = generators
        self.word_count_ranges = word_count_ranges
        self.current_generator_index = 0  # Current generator being used
        self.current_word_count = 0  # Total words generated so far
        self.spacing_generator_range = spacing_generator_range
        self.spacing = 0
        self.reset()
        self.map_from_style_to_generator_idx = {self.get_style(gen): idx for idx, gen in enumerate(self.generators)}
        self.special_args_to_update_object_attributes_that_use_this_generator = special_args_to_update_object_attributes_that_use_this_generator


    def switch_generators(self):
        """
        Switch to the next generator and reset the word count.
        """
        self.current_generator_index = (self.current_generator_index + 1) % len(self.generators)
        self.reset()

    @property
    def just_switched_generators(self):
        return self.current_word_count == 0

    def you_should_update_object_attributes_that_use_this_generator(self):
        if self.just_switched_generators:
            return self.current_special_args
        return False

    @property
    def current_special_args(self):
        if not self.special_args_to_update_object_attributes_that_use_this_generator:
            return None
        return self.special_args_to_update_object_attributes_that_use_this_generator[self.current_generator_index]

    @property
    def current_generator(self):
        return self.generators[self.current_generator_index]

    @property
    def current_generator_style(self):
        return self.get_style(self.current_generator)

    def get_style(self, generator):
        return generator.style if hasattr(generator, "style") else type(generator).__name__

    def override_generator_style(self, style, num_words=None):
        try:
            idx = self.map_from_style_to_generator_idx[style]
            self.current_generator_index = idx
            self.current_word_count = 0
            if num_words:
                self.current_max_count = num_words + 1
            else:
                self.current_max_count = random.randint(*self.word_count_ranges[self.current_generator_index])
            self.spacing = 0
        except:
            logger.error(f"Style {style} not found in generator_style_map")

    def reset(self):
        self.current_word_count = 0
        self.current_max_count = random.randint(*self.word_count_ranges[self.current_generator_index])
        self.spacing = random.randint(*self.spacing_generator_range)

    def parse_generator_result(self, gen_dict):
        return gen_dict["image"], gen_dict["raw_text"]

    def get_added_space(self):
        if self.current_word_count==0:
            return self.spacing
        else:
            return 0

    def generate(self, size=None) -> Dict[str, object]:
        """
        Generate image and text based on the defined word_count_ranges and switching logic.

        Returns:
            Dict[str, object]: A dictionary containing the generated image under "img" and the corresponding text under "text".
        """
        # Get current word_count_range and generator
        generator = self.generators[self.current_generator_index]

        if self.current_word_count < self.current_max_count:
            gen_dict = generator.get(size=size)
            img, raw_text = self.parse_generator_result(gen_dict)
            self.current_word_count += len(raw_text.split())

            if self.current_word_count >= self.current_max_count:
                self.switch_generators()

            return {"img": img,
                    "text": raw_text,
                    "style": generator.style if hasattr(generator, "style") else None,}
        else:
            self.switch_generators()
            return self.generate(size=size)

    def get(self) -> Dict[str, object]:
        """
        Alias for generate() method.

        Returns:
            Tuple[Image.Image, str]: The generated image and the corresponding text.
        """
        return self.generate()

    def __iter__(self) -> Iterator[Dict[str, object]]:
        """
        Make the class iterable.

        Returns:
            Iterator[Tuple[Image.Image, str]]: Iterator for generated images and texts.
        """
        return self

    def __next__(self) -> Dict[str, object]:
        """
        Get the next generated image and text.

        Returns:
            Tuple[Image.Image, str]: The generated image and the corresponding text.
        """
        return self.generate()

    def __getitem__(self, item: int) -> Dict[str, object]:
        """
        Get the generated image and text at a particular index.

        Args:
            item (int): The index at which to generate the image and text.

        Returns:
            Tuple[Image.Image, str]: The generated image and the corresponding text.
        """
        return self.generate()


"""
takes in multiple word_image_generators like SavedHandwritingRandomAuthor, RenderWordFont, etc.
takes in a series of tuples defining the range of the number of words from that generator to use
takes in some parameter to define how much spacing to use when changing word_image_generators

returns:
        return IMG, TEXT


self.renderer_hw = SavedHandwritingRandomAuthor(format="PIL",
                                                dataset_root=saved_hw_folder,
                                                random_ok=True,
                                                conversion=None)

self.renderer = RenderWordFont(format="numpy",
                               font_folder=saved_fonts_folder,
                               clear_font_csv_path=clear_fonts_path)

"""

class MultiGeneratorFromPair(MultiGeneratorFromDict):

    def parse_generator_result(self, gen_tuple):
        img, text = gen_tuple
        return img, text

class MultiGeneratorSmartParse(MultiGeneratorFromDict):
    def parse_generator_result(self, gen_tuple):
        if isinstance(gen_tuple, (list, tuple)):
            img, text = gen_tuple
        elif isinstance(gen_tuple, dict):
            text = gen_tuple.get("raw_text") or gen_tuple.get("text")
            img = gen_tuple.get("image") if "image" in gen_tuple else gen_tuple.get("img")
        else:
            raise Exception
        return img, text


if __name__=='__main__':
    saved_hw_folder = "C:/Users/tarchibald/Anaconda3/envs/docgen_windows/hwgen/resources/generated"
    saved_fonts_folder = Path(r"G:/s3/synthetic_data/resources/fonts")
    clear_fonts_path = Path(saved_fonts_folder) / "clear_fonts.csv"
    hw_generator = SavedHandwritingRandomAuthor(format="PIL",
                                 dataset_root=saved_hw_folder,
                                 random_ok=True,
                                 conversion=None)

    # hw_generator = HWGenerator(next_text_dataset=words_dataset,
    #             batch_size=opts.hw_batch_size,
    #             model=opts.saved_hw_model,
    #             resource_folder=opts.saved_hw_model_folder,
    #             device=opts.device,
    #             style=opts.saved_hw_model,
    #             )

    font_generator = RenderWordConsistentFont(format="numpy",
                   font_folder=saved_fonts_folder,
                   clear_font_csv_path=clear_fonts_path)

    # Then, create a MultiGenerator object
    multi_gen = MultiGeneratorFromDict(
        generators=[hw_generator, font_generator],
        word_count_ranges=[(5, 10), (8, 12)],
    )

    # Generate image and text
    while True:
        img, text = multi_gen.generate()
        if multi_gen.current_word_count == 0:
            show(img)
