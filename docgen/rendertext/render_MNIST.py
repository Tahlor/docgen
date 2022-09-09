from setuptools.sandbox import save_path

from docgen.rendertext.render_word import RenderWord
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
import torchvision
import torchvision.datasets as datasets
from docgen.rendertext.render_word import RenderWord
import torch
import torchvision.transforms.functional as F
import warnings
import types
LOCAL_ROOT = Path(os.path.dirname(__file__))
from collections import defaultdict
from typing import Literal

def open_file(path):
    with open(path, "r") as f:
        return "".join(f.readlines())

class RenderEMNIST(RenderWord):
    def __init__(self, path,
                 train_test='train',
                 split: Literal['byclass','bymerge','balanced','letters','digits','mnist']="digits",
                 format: Literal['numpy', 'PIL']="PIL"):
        """ split: 'byclass': Capital, lowercase, and digits
                   'bymerge': Same as byclass, but some upper/lowercase letters have the same label, CIJKLMOPSUVWXYZ
                   'balanced': same as bymerge, but balanced
                   'letters': same as balanced, but no digits
                   'digits': 0-9
                   'mnist': subset of digits
        """
        path = path / (f"{split}_{train_test}.pt")
        self.format = format
        # Load sorted_emnist if PATH is given
        if Path(path).exists():
            self.letters,self.class_to_idx,self.idx_to_class = torch.load(path)
            return

        transform = torchvision.transforms.Compose(
                                                      [  # Fix image orientation
                                                          lambda img: F.rotate(img, -90),
                                                          lambda img: F.hflip(img),
                                                          torchvision.transforms.ToTensor(),
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                                      ])
        dataset = None
        if train_test == 'train':
            dataset = torchvision.datasets.EMNIST(LOCAL_ROOT / 'data/emnist', split=split, train=True, download=True,
                                                  transform=transform)
        elif train_test == 'test':
            dataset = torchvision.datasets.EMNIST(LOCAL_ROOT / 'data/emnist', split=split, train=False, download=True,
                                                  transform=transform)  # , shuffle=True)

        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = dataset.classes #{_index:_class for _class,_index in self.class_to_idx.items()}
        assert max(v for k, v in self.class_to_idx.items()) == len(dataset.classes)-1
        self.class_to_idx[" "] = len(self.idx_to_class)
        self.idx_to_class.append(" ")

        # Store Training Dataset
        letters = defaultdict(list)
        for x in dataset:
            # plot(x[0], x[1])
            img = torchvision.transforms.functional.invert(x[0]).permute(1,2,0)
            char = self.idx_to_class[x[1]]
            letters[char].append(img)

        # Add a space to the image EMNIST dataset
        mm = [(torch.zeros(x[0].shape) + torch.min(x[0]))]
        letters[' '] = mm

        self.letters = letters
        # Save dictionary of lists of char images
        torch.save([self.letters,self.class_to_idx,self.idx_to_class], path)
        return

    def sample(self, char, force_lower=False):
        char = self.idx_to_class[char] if isinstance(char, int) else char
        if force_lower:
            char = char.lower()
        img_idx = random.randrange(0, len(self.letters[char]))
        return self.letters[char][img_idx], img_idx

    def render_digit(self, word=None, size=None):
        if word is None:
            word = random.choice(self.idx_to_class)
        word_image, img_idx = self.sample(word)

        self.output(word, word_image, img_idx, size=size)


    def render_word(self, word, font=None, size=None):
        imgs = []; fonts = []
        for letter in word:
            img, font = self.sample(letter)
            imgs.append(img); fonts.append(font)

        self.output(word, imgs, fonts, size=size)

    def output(self, word, word_image, img_idx, size):
        if size:
            h = size
            w = int(word_image.shape[1] * h / word_image.shape[0])
            word_image = resize(word_image, [w,h])
        if self.format == "PIL":
            word_image = Image.fromarray(word_image)
        return {
            "image": word_image,
            "font": img_idx,
            "size": size,
            "bbox": None,
            "raw_text": word}


if __name__ == '__main__':
    train_emnist = RenderEMNIST(path=LOCAL_ROOT / "data", split="digits")
    display(train_emnist.render_word("1925")["image"])
    pass
