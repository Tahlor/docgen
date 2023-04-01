from typing import List, Any
import inspect
import numpy as np
import torch
import albumentations as A

class MetaCompose(A.BasicTransform):
    def __init__(self, transforms: List[Any], **kwargs):
        """
        Initializes the meta compose function with a list of transforms.

        Args:
            transforms (List[Any]): A list of transformation functions or objects.
        """
        super(MetaCompose, self).__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, image, **kwargs):
        """
        Applies the transforms to the input image.

        Args:
            image: The input image.

        Returns:
            The transformed image.
        """
        for transform in self.transforms:
            # Check if the transform accepts a keyword argument (e.g., 'image' for Albumentations)
            # params = inspect.signature(transform).parameters
            # if 'image' in params:
            if isinstance(transform, A.BasicTransform):
                image = transform(image=image)['image']
            else:
                image = transform(image)
        return image


if __name__ == "__main__":
    from torchvision.transforms import ToTensor, RandomHorizontalFlip
    from albumentations import HorizontalFlip, Compose

    # Define a list of transforms from both Albumentations and torchvision
    torch_transforms_list = [
        RandomHorizontalFlip(p=0.5),
    ]
    albumentations_transforms_list = [
        A.HorizontalFlip(p=0.5),
    ]

    meta_compose = MetaCompose(transforms=torch_transforms_list)
    random_image = torch.rand(3, 256, 256)
    transformed_image = meta_compose(random_image)
    print(transformed_image.shape)

    meta_compose = MetaCompose(transforms=albumentations_transforms_list)
    random_np_image = np.random.rand(256, 256, 3)
    transformed_np_image = meta_compose(random_np_image)
    print(transformed_np_image.shape)