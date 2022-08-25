import ocrodeg
import scipy.ndimage as ndi
import random
from PIL import Image
import numpy as np

"""
COST OF EACH
blur	215	1.0
splotches	2754	12.8
ruled_surface_distortions	600	2.8
random_distortions	3206	14.9
fibrous1	10304	47.9
fibrous2	10554	49.1
"""

def blur(img, s=range(1,3)):
    sigma = random.uniform(*s)
    return ndi.gaussian_filter(img, sigma)

def random_distortions(img, sigmas=[2, 5, 20]):
    sigma = random.choice(sigmas)
    noise = ocrodeg.bounded_gaussian_noise(img.shape, sigma, 5.0)
    distorted = ocrodeg.distort_with_noise(img, noise)
    return distorted

def ruled_surface_distortions(img, mag=5):
    noise = ocrodeg.noise_distort1d(img.shape, magnitude=mag)
    distorted = ocrodeg.distort_with_noise(img, noise)
    return distorted

def splotches(img):
    return ocrodeg.random_blotches(img, 3e-4, 1e-4)

def fibrous1(img):
    return ocrodeg.printlike_multiscale(img)

def fibrous2(img):
    return ocrodeg.printlike_fibrous(img)

all_funcs = [blur, random_distortions, ruled_surface_distortions, splotches, fibrous1, fibrous2]

def degradation_function_composition(img):
    if isinstance(img, Image.Image):
        PIL = True
        mode = img.mode
        img = np.array(img)
    else:
        PIL = False

    if np.max(img)>1.01:
        rescale = True
        img = img / 255.0
    else:
        rescale = False

    number_of_distortions = random.randint(0, len(all_funcs))
    random.shuffle(all_funcs)

    img = apply_all(img, all_funcs[:number_of_distortions])

    if rescale:
        img = img * 255
        img = img.astype(np.uint8)
    if PIL:
        img = Image.fromarray(img, mode=mode)

    return img

def apply_all(img, funcs=all_funcs):
    for f in funcs:
        img = np.clip(img, 0, 1)
        img = f(img)
    img = np.clip(img, 0, 1)
    return img


if __name__ == "__main__":
    noise = np.random.uniform(size=3200*2500).reshape(3200, -1)
    apply_all(noise)
