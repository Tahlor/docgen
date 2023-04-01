import numpy as np
import matplotlib.pyplot as plt
from hwgen.data.utils import show,display

"""
Format:
dataset = {author: {word1:[word1_img1, word1_img2], word2:[word2_img1, ...]}}
"""

path="C:/Users/tarchibald/Anaconda3/envs/docgen_windows/hwgen/resources/generated/style_600_IAM_IAM_samples.npy"
path="G:/data/standard_hwr/emnist/EMNIST.npy"
# open it, get .item, print first key/item
dataset = np.load(path, allow_pickle=True).item()
print(dataset.keys())
print(dataset["UNKNOWN_AUTHOR"]["a"][0].squeeze().shape)


print(dataset["600_IAM_IAM"])

