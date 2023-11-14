"""
Threshold masking for segmentation dataset
"""
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class NaiveMask:
    """ It's not even a mask, it's just pixel intensity
    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, img):
        # convert to grayscale using luminance
        if img.shape[0] == 3:
            img = torch.sum(img, dim=0) / 3
        return img

    def graph(self):
        import matplotlib.pyplot as plt
        x = np.linspace(0, 1, 100)
        y = self.__call__(torch.tensor(x))
        plt.plot(x, y)
        plt.show()

class Mask(NaiveMask):
    def __init__(self, threshold=.5, *args, **kwargs):
        self.threshold01 = threshold if threshold < 1 else threshold / 255

    def __call__(self, img):
        return torch.where(img < self.threshold01, torch.tensor(1), torch.tensor(0))

class SoftMask(Mask):
    def __init__(self, soft_mask_threshold=.3, soft_mask_steepness=20):
        super().__init__()
        self.soft_mask_threshold = soft_mask_threshold
        self.soft_mask_steepness = soft_mask_steepness

    def __call__(self, img):
        mask = 1.0 - img
        transition_point = self.soft_mask_threshold
        steepness = self.soft_mask_steepness
        mask = torch.sigmoid(steepness * (mask - transition_point))
        return mask


if __name__=='__main__':
    from matplotlib import pyplot as plt
    mask = SoftMask()
    mask.graph()
    plt.show()