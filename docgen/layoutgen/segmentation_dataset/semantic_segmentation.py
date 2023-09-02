import socket
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_image
from typing import List, Optional, Tuple, Union
from PIL import Image
from torchvision.transforms import ToPILImage
from typing import List, Optional, Tuple, Union, Literal, Callable
from docgen.layoutgen.layoutgen import composite_images_PIL
from docgen.transforms.transforms import ResizeAndPad, IdentityTransform
from docgen.image_composition.utils import CompositeImages, CalculateImageOriginForCompositing
#from docdegrade.composite import CompositeImages

to_pil = ToPILImage()
"""
# compose random words, tables, and handwriting into a document
# pre-create full-page different handwriting, tables, and words
# compile on the fly and apply degradations
# specify binarization threshold to create binary mask
# dataset should be degraded, labels are the binary masks stacked as a tensor

# start with just small images, then move to full page

# Two avenues:
# * a dataset that is self-contained that contains all parts
# * a dataset this is composed of other datasets

"""
class SoftMaskConfig:
    def __init__(self, soft_mask_threshold=.3, soft_mask_steepness=20):
        self.soft_mask_threshold = soft_mask_threshold
        self.soft_mask_steepness = soft_mask_steepness

class SemanticSegmentationDataset(Dataset):
    def __init__(self,
                 transforms_before_mask_threshold=None,
                 transforms_after_mask_threshold=None,
                 threshold=.6,
                 overfit_dataset_length=0,
                 size=448,
                 soft_mask_config: Union[SoftMaskConfig, bool]=SoftMaskConfig()
                 ):
        # if threshold is None, just use the pixel intensity as labellogit
        self.transforms_before = transforms_before_mask_threshold
        self.transforms_after = transforms_after_mask_threshold
        self.threshold01 = threshold if threshold < 1 else threshold * 255
        self.overfit_dataset_length = overfit_dataset_length
        self.soft_mask_config = soft_mask_config

        # Default transformations before thresholding
        if self.transforms_before is None:
            self.transforms_before = transforms.Compose([
                #transforms.ToPILImage(),
                transforms.ToTensor(),
                #resize_so_largest_side_is(size=size) if size else IdentityTransform(),
                #pad_image,
                ResizeAndPad(size, 32) if size else IdentityTransform()
            ])
            
        # Default transformations after thresholding
        if self.transforms_after is None:
            self.transforms_after = transforms.Compose([
                #transforms.ToPILImage(),
                #transforms.ToTensor()
            ])

        # self.transforms_after = transforms.Compose([
        #     #to_numpy
        #     lambda x: x.numpy(),
        #     self.transforms_after,
        #     transforms.ToTensor()
        # ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.get_image(idx)

        if self.transforms_before:
            img = self.transforms_before(img)

        # convert to grayscale using luminance
        if img.shape[0] == 3:
            bw_img = img[0] * 0.2126 + img[1] * 0.7152 + img[2] * 0.0722
        else:
            bw_img = img[0]

        if self.soft_mask_config:
            #mask = torch.where(bw_img < self.threshold01, 1 - bw_img, torch.tensor(0))
            mask = 1.0 - bw_img
            transition_point = self.soft_mask_config.soft_mask_threshold
            steepness = self.soft_mask_config.soft_mask_steepness
            mask = torch.sigmoid(steepness * (mask - transition_point))
        else:
            mask = torch.where(bw_img < self.threshold01, torch.tensor(1), torch.tensor(0))

        if self.transforms_after:
            img = self.transforms_after(img)

        sample = {'image': img, 'mask': mask}

        return sample

    @staticmethod
    def collate_fn_simple(batch):
        images = [item['image'] for item in batch]
        masks = [item['mask'] for item in batch]
        return {'image': torch.stack(images, dim=0), 'mask': torch.stack(masks, dim=0)}

    @staticmethod
    def collate_fn(batch, no_tensor_keys=None):
        """

        Args:
            batch:
            no_tensor_keys: keys in dict that should be collated as a list instead of a tensor

        Returns:
            batch:

        """
        keys = batch[0].keys()
        collated_batch = {}

        for key in keys:
            if no_tensor_keys and key in no_tensor_keys:
                collated_batch[key] = [item[key] for item in batch]
            else:
                collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
        return collated_batch

    def get_image(self, idx):
        raise NotImplementedError()

class SemanticSegmentationDatasetGenerative(SemanticSegmentationDataset):
    def __init__(self,
                 generator,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator

    def get_image(self, idx):
        return self.generator.get()

    def __len__(self):
        return sys.maxsize

class SemanticSegmentationDatasetImageFolder(SemanticSegmentationDataset):
    def __init__(self, img_dir,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.img_dir = img_dir
        self.img_paths = self.get_images(img_dir)

    def get_images(self, img_dir):
        return [x for x in Path(img_dir).rglob("*.jpg")]

    def __len__(self):
        return len(self.img_paths)

    def get_image(self, idx):
        if self.overfit_dataset_length > 0:
            idx = idx % self.overfit_dataset_length
        else:
            idx = idx % len(self)

        img_path = self.img_paths[idx]
        img = Image.open(img_path).copy()
        return img


import random


def conservative_random_offset(bck_w, bck_h, img_w, img_h, ):
    if bck_w > img_w and bck_h > img_h:
        start_x = random.randint(0, bck_w - img_w)
        start_y = random.randint(0, bck_h - img_h)
    else: # image is larger than background in at least one dimension
        start_x = random.randint(-img_w // 2, bck_w // 2)
        start_y = random.randint(-img_h // 2, bck_h // 2)
    return start_x, start_y


import random

def more_random_offset(bck_w, bck_h, img_w, img_h, percent=0.9):
    # Calculate the maximum offset, based on the percentage
    max_offset_x = max(0, int((bck_w - img_w) * percent))
    max_offset_y = max(0, int((bck_h - img_h) * percent))

    min_offset_x = -max(0,int(img_w*(1-percent)))
    min_offset_y = -max(0,int(img_h *(1-percent)))

    # The start_x value can now be negative
    start_x = random.randint(min_offset_x, bck_w - img_w)
    start_y = random.randint(min_offset_y, bck_h - img_h)

    return start_x, start_y


class AggregateSemanticSegmentationDataset(Dataset):
    def __init__(self, subdatasets:List[SemanticSegmentationDatasetImageFolder],
                 background_img_properties=None,
                 overfit_dataset_length=0,
                 random_origin_composition=True,
                 mask_default_value=0,
                 img_default_value=1,
                 dataset_length=5000,
                 transforms_after_compositing=None,):
        """

        Args:
            subdatasets:
            background_img_properties: a size tuple, or the property of compositing images (use "min", "max")
            overfit_dataset_length:
            random_origin_composition (bool): randomize image composition
            mask_default_value (int): what value should the mask be by default
            img_default_value (int): if pasting on to a blank image, what value should it be by default
        """
        self.subdatasets = subdatasets
        self.background_img_properties = background_img_properties
        self.overfit_dataset_length = overfit_dataset_length
        self.random_origin_composition = random_origin_composition
        self.mask_default_value = mask_default_value
        self.img_default_value = img_default_value
        self.dataset_length = dataset_length
        self.transforms_after_compositing = transforms_after_compositing
        self.random_origin_function = CalculateImageOriginForCompositing(image_format="CHW") # vs. more_random_offset

    def __len__(self):
        #return min(len(d) for d in self.subdatasets)
        return self.dataset_length

    @staticmethod
    def composite_the_images_torch(background_img, img, bckg_x, bckg_y):
        # Shape of the composite_image
        bckg_h, bckg_w = background_img.shape[-2:]
        img_h, img_w = img.shape[-2:]

        # if background_img has 2 dimensions and img has 3, expand the background
        if len(background_img.shape) == 2 and len(img.shape) == 3:
            background_img = background_img.unsqueeze(0)

        # if img has more channels than background, expand the background
        if img.shape[0] > background_img.shape[0] and len(background_img.shape) == 3 == len(img.shape):
            background_img = background_img.expand(img.shape[0], -1, -1)
            #assert torch.all(background_img[0] == background_img[1])

        # Calculate the coordinates of the paste area
        paste_x = bckg_x
        paste_y = bckg_y
        paste_width = min(img_w, bckg_w - bckg_x if bckg_x >= 0 else img_w + bckg_x, bckg_w)
        paste_height = min(img_h, bckg_h - bckg_y if bckg_y >= 0 else img_h + bckg_y, bckg_h)

        # Check for overlap
        if paste_width <= 0 or paste_height <= 0:
            print("No overlap between the images.")
            return background_img

        # Paste the overlap onto the background image
        background_img = background_img.clone()

        slice_background = background_img[..., paste_y:paste_y + paste_height, paste_x:paste_x + paste_width]
        slice_img = img[..., :paste_height, :paste_width]

        # Ensure the slices are of the same shape
        if slice_background.shape == slice_img.shape:
            background_img[..., paste_y:paste_y + paste_height, paste_x:paste_x + paste_width] = torch.min(
                slice_background, slice_img)
        else:
            print("Shapes do not match: slice_background = {}, slice_img = {}".format(slice_background.shape,
                                                                                      slice_img.shape))

        # Return the composite image
        return background_img

    def composite_the_images3(self, background_img, img, bckg_x, bckg_y):
        # if background_img has 2 dimensions and img has 3, expand the background
        if len(background_img.shape) == 2 and len(img.shape) == 3:
            background_img = background_img.unsqueeze(0)

        # if img has more channels than background, expand the background
        if img.shape[0] > background_img.shape[0] and len(background_img.shape) == 3 == len(img.shape):
            background_img = background_img.expand(img.shape[0], -1, -1)
            #assert torch.all(background_img[0] == background_img[1])

        # Paste the overlap onto the background image
        background_img = background_img.clone()

        bckg_h, bckg_w = background_img.shape[-2:]
        img_h, img_w = img.shape[-2:]

        x_start, x_end = max(bckg_x, 0), min(bckg_x + img_w, bckg_w)
        y_start, y_end = max(bckg_y, 0), min(bckg_y + img_h, bckg_h)

        img_x_start, img_x_end = max(-bckg_x, 0), min(-bckg_x + bckg_w, img_w)
        img_y_start, img_y_end = max(-bckg_y, 0), min(-bckg_y + bckg_h, img_h)

        if x_end <= x_start or y_end <= y_start:
            print("No overlap between the images.")
            # return background_img

        background_img[..., y_start:y_end, x_start:x_end] = img[..., img_y_start:img_y_end, img_x_start:img_x_end]
        return background_img

    def composite_the_images(self, background_img, img, bckg_x, bckg_y, method:Callable=torch.min):

        # # If the images have a different number of channels, expand them to have 3 channels
        # if len(background_img.shape) < 3:
        #     background_img = background_img.unsqueeze(0).repeat(3, 1, 1)
        # if len(img.shape) < 3:
        #     img = img.unsqueeze(0).repeat(3, 1, 1)

        if len(background_img.shape) == 2 and len(img.shape) == 3:
            background_img = background_img.unsqueeze(0)

        # if img has more channels than background, expand the background
        if img.shape[0] > background_img.shape[0] and len(background_img.shape) == 3 == len(img.shape):
            background_img = background_img.expand(img.shape[0], -1, -1)
            #assert torch.all(background_img[0] == background_img[1])


        bckg_h, bckg_w = background_img.shape[-2:]
        img_h, img_w = img.shape[-2:]

        x_start, x_end = max(bckg_x, 0), min(bckg_x + img_w, bckg_w)
        y_start, y_end = max(bckg_y, 0), min(bckg_y + img_h, bckg_h)

        img_x_start, img_x_end = max(-bckg_x, 0), min(-bckg_x + bckg_w, img_w)
        img_y_start, img_y_end = max(-bckg_y, 0), min(-bckg_y + bckg_h, img_h)

        if x_end <= x_start or y_end <= y_start:
            print("No overlap between the images.")
            # return background_img
        # Clone the background image to avoid modifying the original
        background_img = background_img.clone()

        # Take the minimum pixel value between the images in the overlapping area
        background_img[..., y_start:y_end, x_start:x_end] = method(background_img[..., y_start:y_end, x_start:x_end],
                                                                      img[..., img_y_start:img_y_end,
                                                                      img_x_start:img_x_end])

        return background_img

    def __getitem__(self, idx):

        if self.overfit_dataset_length > 0:
            idx = idx % self.overfit_dataset_length

        images_and_masks = [d[idx] for d in self.subdatasets]

        # convert list of dicts to tuple of lists
        images, masks = [x["image"] for x in images_and_masks], [x["mask"] for x in images_and_masks]

        if isinstance(self.background_img_properties, str):
            if self.background_img_properties == 'max':
                # choose size based on total pixels
                # max(np.product(img.shape) for img in images)
                bckg_size = max(images, key=lambda img: np.product(img.shape)).shape
            elif self.background_img_properties == 'min':
                bckg_size = min(images, key=lambda img: np.product(img.shape)).shape
            else:
                raise ValueError("Invalid mode. Choose from 'max', 'min'")
        elif isinstance(self.background_img_properties, (tuple,list)):
            if len(self.background_img_properties) == 2:
                bckg_size = (1,*self.background_img_properties)

            else:
                bckg_size = self.background_img_properties
        else:
            raise ValueError("Invalid background_img_properties. Choose from 'max', 'min' or specify a size tuple")

        composite_image = torch.ones(bckg_size)

        if self.img_default_value:
            composite_image *= self.img_default_value

        composite_masks = []

        for img, mask in zip(images, masks):
            if self.random_origin_composition:
                # if bckg is bigger in both dimensions
                # start_x, start_y = more_random_offset(composite_image.shape[-1],
                #                                       composite_image.shape[-2],
                #                                       img.shape[-1],
                #                                       img.shape[-2])
                start_x, start_y = self.random_origin_function(composite_image, img, .7)
            else:
                start_x = 0
                start_y = 0

            #composite_image[start_y:start_y + img.shape[0], start_x:start_x + img.shape[1]] += img
            composite_image = self.composite_the_images(composite_image, img, start_x, start_y, method=torch.min)

            pasted_mask = torch.zeros(bckg_size[-2:])
            if self.mask_default_value!=0:
                pasted_mask += self.mask_default_value

            pasted_mask = self.composite_the_images(pasted_mask, mask, start_x, start_y, method=torch.max)

            composite_masks.append(pasted_mask)
        composite_masks = torch.stack(composite_masks, dim=0)

        # apply self.transforms_after_compositing
        if self.transforms_after_compositing:
            composite_image = self.transforms_after_compositing(composite_image)

        return {'image': composite_image, 'mask': composite_masks}

    def collate_fn(self, batch):
        images = [item['image'] for item in batch]
        masks = [item['mask'] for item in batch]
        return {'image': torch.stack(images, dim=0), 'mask': torch.stack(masks, dim=0)}


class FlattenPILGenerators(torch.utils.data.Dataset):
    """ Take several generating datasets and flatten the output into 1 image
        ASSUMES PIL size formats
    """
    def __init__(self, datasets,
                 img_size=(512,512),
                 random_offset=True,
                 color_scheme="L"):
        self.datasets = datasets
        self.random_offset = random_offset
        self.img_size = img_size
        self.color_scheme = color_scheme

    def get(self):
        # composite all images
        img = Image.new(self.color_scheme, self.img_size, color='white')
        for d in self.datasets:
            overlay_img = d.get()
            if self.random_offset:
                offset = more_random_offset(img.size[0],img.size[1], overlay_img.size[0],overlay_img.size[1])
            else:
                offset = (0,0)
            img = composite_images_PIL(img, overlay_img, pos=offset)
        return img

class FlattenDatasets(torch.utils.data.Dataset):
    """ Take several generating datasets and flatten the output into 1 image
        for TORCH
    """
    def __init__(self, datasets, random_offset=True):
        self.datasets = datasets
        self.random_offset = random_offset

    def get(self):
        raise NotImplementedError("Not implemented for torch yet")

if __name__=="__main__":
    from docgen.layoutgen.segmentation_dataset.word_gen import HWGenerator, PrintedTextGenerator
    from docgen.layoutgen.segmentation_dataset.grid_gen import GridGenerator
    from docgen.layoutgen.segmentation_dataset.line_gen import LineGenerator
    from docgen.layoutgen.segmentation_dataset.box_gen import BoxGenerator

    if socket.gethostname() == "PW01AYJG":
        saved_fonts_folder = Path(r"G:/s3/synthetic_data/resources/fonts")
        saved_hw_folder = Path("C:/Users/tarchibald/Anaconda3/envs/docgen_windows/hwgen/resources/generated")
    elif socket.gethostname() == "Galois":
        saved_fonts_folder = Path("/media/EVO970/s3/datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/fonts/")
        saved_hw_folder = Path("/media/EVO970/s3/synthetic_data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words")

    # generated version
    hw_generator = HWGenerator(saved_hw_folder=saved_hw_folder)
    printed_text_generator = PrintedTextGenerator(saved_fonts_folder=saved_fonts_folder)

    grid_generator = GridGenerator()
    line_generator = LineGenerator()
    box_generator = BoxGenerator()

    form_generator = FlattenPILGenerators([grid_generator, line_generator, box_generator])

    # form elements
    form_dataset = SemanticSegmentationDatasetGenerative(form_generator)
    hw_dataset = SemanticSegmentationDatasetGenerative(hw_generator)
    printed_dataset = SemanticSegmentationDatasetGenerative(printed_text_generator)

    aggregate_dataset = AggregateSemanticSegmentationDataset([form_dataset, hw_dataset, printed_dataset],
                                                             background_img_properties='max',

                                                             )

    dataloader = torch.utils.data.DataLoader(aggregate_dataset, batch_size=2, collate_fn=aggregate_dataset.collate_fn)

    for i, batch in enumerate(dataloader):
        print(batch['image'].shape)
        print(batch['mask'].shape)
        # show images
        for j in range(batch['image'].shape[0]):
            img = batch['image'][j]
            mask = batch['mask'][j]

            img_pil = transforms.ToPILImage()(img)
            mask_pil1 = transforms.ToPILImage()(mask[0])
            mask_pil2 = transforms.ToPILImage()(mask[1])

        if i > 15:
            break
