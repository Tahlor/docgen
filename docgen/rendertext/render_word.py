from docgen.rendertext.utils.filelock import FileLock, FileLockException
from docgen.rendertext.utils.util import ensure_dir
import threading
from synthetic_text_gen import SyntheticWord
import os, random, re, time
import numpy as np
from docgen.rendertext.utils import img_f
import argparse
from cv2 import resize
from docgen.utils import *
from PIL import Image
from typing import Literal

FONT_DIR = r"C:\Users\tarchibald\github\brian\fslg_documents\data\fonts"
CLEAR_FONTS = r"C:\Users\tarchibald\github\brian\fslg_documents\data\fonts\clear_fonts.csv"

""" 
TODO: Specify font by resizing word image into pixel height
"""

class RenderWord:
    """ !!! HW should inherit from this

    """
    def __init__(self, *args, **kwargs):
        pass

    def render_word(self, word, font=None, size=None):
        """
        Returns:
            word_image
            bbox: what are the bounds of the word?
            font: what font
            size: what font size? or something

        """
        word_image = None
        return {"image":word_image,
                "font":font,
                "size":size,
                "bbox":None}

    def generate(self, phrase):
        """ Words will be completely cropped by default

        Args:
            phrase:

        Returns:

        """
        for word in phrase.split(' '):
            yield self.render_word(word)

    def detect_word_boundary(self, image):
        pass

    def resize_to_height_numpy(self, image, height):
        width = int(image.shape[1] * height / image.shape[0])
        return resize(image, [width, height])


class WordDataset:
    def __init__(self, dataset):
        """ Just a wrapper around an iterator
        """
        super().__init__()
        self.update_dataset(dataset)

    def update_dataset(self, new_dataset):
        """

        Args:
            new_dataset: str or iterable

        Returns:

        """
        if isinstance(new_dataset, str):
            new_dataset = new_dataset.split(' ')
        self.dataset = new_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # return self.cycle(idx)
        return self.dataset[idx]

    def cycle(self, idx):
        while True:
            for x in self.dataset[idx % len(self)]:
                yield x

    def random_word(self):
        return random.choice(self.dataset)

class RenderWordFont(RenderWord):
    def __init__(self,
                 format: Literal['numpy', 'PIL'],
                 font_dir=FONT_DIR,
                 clear_font_csv=CLEAR_FONTS,
                 word_dataset=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gen = SyntheticWord(font_dir, clear=clear_font_csv)
        self.word_dataset = word_dataset if not word_dataset is None else WordDataset(["ipsum", "lorem"])
        self.format = format

    def genImageForWord(self, word, font, fontN, font_name=None, fontN_name=None):
        if re.search(r'\d', word):
            _font = fontN
            _font_name = fontN_name
        else:
            _font = font
            _font_name = font_name
        img, word_new = self.gen.getRenderedText(_font, word)
        if img is None:
            # retry generation
            if _font != fontN:
                _font_name = fontN_name
                img, word_new = self.gen.getRenderedText(fontN, word)

            if img is None:
                _, _, font2, font2_name = self.gen.getFont()
                _font_name = font2_name
                img, word_new = self.gen.getRenderedText(font2, word)

            if img is None:
                return None, None

        img = ((1-img) * 255).astype(np.uint8)
        return img, word_new

    def render_word_loop(self, word, tries=3):
        """ Choose random font

        Args:
            word:
            font:

        Returns:

        """
        for retry in range(tries):
            index = np.random.choice(len(self.gen.fonts))
            filename, hasLower, hasNums, hasBrackets = self.gen.fonts[index]
            img, word_new = self.render_word(word, filename)
            if img is not None:
                break

    def render_word(self, word=None, font=None, size=None):
        """
        Returns:
            word_image
            bbox: what are the bounds of the word?
            font: what font
            size: pixel height


            t_font:
                (<PIL.ImageFont.FreeTypeFont object at 0x000001B0F989C7F0>, array(251, dtype=int64), array(370, dtype=int64), True, True)
            t_font_name:
                'fonts/nk57-monospace-sc-rg.ttf'
            t_fontN: (t_font)
            t_fontN_name:  (t_font_name)
        """
        if word is None:
            word = self.word_dataset.random_word()

        t_font, t_font_name, t_fontN, t_fontN_name = self.gen.getFont(font)
        word_image, word_new = self.genImageForWord(word, t_font, t_fontN)
        if size:
            h = size
            w = int(word_image.shape[1] * h / word_image.shape[0])
            word_image = resize(word_image, [w,h])
        if self.format == "PIL":
            word_image = Image.fromarray(word_image)
        return {"image":word_image,
                "font":font,
                "size":size,
                "bbox":None,
                "raw_text":word}

class RenderImageTextPair:

    def __init__(self, renderer, textgen):
        self.renderer = renderer
        self.textgen = textgen

    @property
    def font_size(self):
        return self.renderer.font_size

    def __len__(self):
        return len(self.textgen)

    def __getitem__(self,i):
        d = self.renderer.render_word(self.textgen[i])
        return d["image"], d["raw_text"]


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    R = RenderWordFont(format="PIL")
    X = R.render_word("word")
    plt.imshow(X["image"], cmap="gray")
    plt.show()
    pass