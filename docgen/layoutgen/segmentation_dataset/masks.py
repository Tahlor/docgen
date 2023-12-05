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
    """ Darker than threshold becomes GT / 1

    """
    def __init__(self, threshold=.5, *args, **kwargs):
        self.threshold01 = threshold if threshold < 1 else threshold / 255

    def __call__(self, img):
        return torch.where(img < self.threshold01, torch.tensor(1), torch.tensor(0))

class GrayscaleMask(Mask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, img):
        if img.shape[0] == 3:
            # Using weighted sum for grayscale conversion
            weights = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)
            img = torch.sum(img * weights, dim=0)
        return img


class SoftMask(Mask):
    def __init__(self, threshold=.7,
                 soft_mask_steepness=40,
                 roundoff_threshold=.02, ):
        """
        Args:
            threshold: Darker than this becomes 1
            soft_mask_steepness:
        """
        super().__init__()
        self.soft_mask_threshold = threshold
        self.soft_mask_steepness = soft_mask_steepness
        self.roundoff_threshold = roundoff_threshold

    def __call__(self, img):
        transition_point = self.soft_mask_threshold
        steepness = self.soft_mask_steepness
        mask = torch.sigmoid(steepness * (transition_point-img))

        mask = torch.where(mask > 1-self.roundoff_threshold, torch.tensor(1), mask)
        mask = torch.where(mask < self.roundoff_threshold, torch.tensor(0), mask)
        return mask


if __name__=='__main__':
    from matplotlib import pyplot as plt
    mask = SoftMask()
    mask.graph()
    plt.show()