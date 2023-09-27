from docgen.layoutgen.segmentation_dataset.utils.dataset_sampler import LayerSampler
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
from docgen.transforms.transforms_torch import ResizeAndPad, IdentityTransform
from docgen.image_composition.utils import CompositeImages, CalculateImageOriginForCompositing
from docgen.image_composition.utils import seamless_composite, composite_the_images_numpy, composite_the_images_torch
from docgen.image_composition.utils import compute_paste_origin_from_overlap_auto
from docdegrade.degradation_objects import RandomDistortions, RuledSurfaceDistortions, Blur, Lighten, Blobs, \
    BackgroundMultiscaleNoise, BackgroundFibrous, Contrast, ConditionalContrast


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

# If the pixel percentage of the image that is identified as text (based on some threshold), then throw it out 
# TEXT: G:\s3\forms\PDF\GDC\text\images\
# FORM ELEMENTS: G:\s3\forms\PDF\GDC\other_elements\images\ - require pretty dark threshold, on at least 1 channel
# 

# Aggregate: should be given a list of descriptions of the layers to be composited; each dataset should have one of these layer descriptions
# We can use a "meta" random choose that decides which if any dataset it's going to sample
# It can choose to sample NONE of them, in which case we just omit that layer
# COMBINED FORMS: what if we want it to predict TEXT+FORM element combined?

# Another experiment: don't predict the mask, just predict the B&W original pixel value
# When I save the dataset, just save the raw pixels and let the mask calculated at runtime?

        if self.mask_object:
            #mask = torch.where(bw_img < self.threshold01, 1 - bw_img, torch.tensor(0))
            mask = 1.0 - bw_img
            transition_point = self.mask_object.soft_mask_threshold
            steepness = self.mask_object.soft_mask_steepness
            mask = torch.sigmoid(steepness * (mask - transition_point))
        else:
            mask = torch.where(bw_img < self.threshold01, torch.tensor(1), torch.tensor(0))

* Figure out BEFORE augmentations, these include rotation, size, some kinds of distortions, color, etc. When do we convert it to a tensor?
    # Let's avoid all kinds of numpy augmentations
* Random chooser
* Config maker!
* Crop the top and bottom of images
* Crop the bing ones
* Random rotation / mirror


https://www.1001fonts.com/handwriting-fonts.html
https://fonts.google.com/?category=Handwriting

"""


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

class Mask(NaiveMask):
    def __init__(self, threshold=.5, *args, **kwargs):
        self.threshold01 = threshold if threshold < 1 else threshold * 255

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

def compose(transform_list):
    if isinstance(transform_list, (list,tuple)):
        return transforms.Compose(transform_list)
    else:
        return transform_list
class SemanticSegmentationDataset(Dataset):
    def __init__(self,
                 layer_contents: Union[tuple,str,list],
                 transforms_before_mask_threshold=None,
                 transforms_after_mask_threshold=None,
                 overfit_dataset_length=0,
                 size=448,
                 mask_maker: Union[Mask, bool]=None,
                 layer_position=50,
                 sample_weight=1.0,
                 name="generic",
                 percent_overlap=0.7,
                 composite_function=None,
                 ):
        """

        Args:
            layer_contents: e.g. ('text', 'form_elements', 'handwriting')
            transforms_before_mask_threshold:
            transforms_after_mask_threshold:
            overfit_dataset_length:
            size:
            mask_maker:
            layer_position: 0 is the bottom layer, 1 is the next layer, etc., optional for AGGREGATOR
            sample_weight: how much to weight this dataset when sampling in AGGREGATOR
            name: name of the dataset
            percent_overlap: how much of the image should overlap with the previous layer in AGGREGATOR
            composite_function: function to use to composite the images in AGGREGATOR
        """
        # sort it so that the order is consistent
        self.layer_contents = (layer_contents,) if isinstance(layer_contents, str) else tuple(sorted(layer_contents))
        self.layer_position = layer_position
        # if threshold is None, just use the pixel intensity as labellogit
        self.transforms_before = compose(transforms_before_mask_threshold)
        self.transforms_after = compose(transforms_after_mask_threshold)
        self.overfit_dataset_length = overfit_dataset_length
        self.mask_maker = mask_maker if mask_maker else Mask()
        self.sample_weight = sample_weight
        self.name = name
        self.percent_overlap = percent_overlap
        self.composite_function = composite_function

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

        mask = self.mask_maker(bw_img)

        if self.transforms_after:
            img = self.transforms_after(img)

        sample = {'image': img, 'mask': mask}

        return sample

    from hwgen.data.utils import show

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
                 layer_contents,
                 generator,
                 *args,
                 **kwargs):
        super().__init__(layer_contents, *args, **kwargs)
        self.generator = generator

    def get_image(self, idx):
        return self.generator.get()

    def __len__(self):
        return sys.maxsize

class SemanticSegmentationDatasetImageFolder(SemanticSegmentationDataset):
    def __init__(self, layer_contents, img_dir,
                 *args,
                 **kwargs):
        super().__init__(layer_contents, *args, **kwargs)
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
    def __init__(self, subdatasets:List[SemanticSegmentationDataset],
                 background_img_properties=None,
                 overfit_dataset_length=0,
                 random_origin_composition=True,
                 mask_default_value=0,
                 img_default_value=1,
                 dataset_length=5000,
                 transforms_after_compositing=None,
                 output_channel_content_names=None,
                 layout_sampler=None,
                 size=448,
                 composite_function: Union[seamless_composite, composite_the_images_torch]=composite_the_images_torch,):

        """

        Args:
            subdatasets:
            background_img_properties: a size tuple, or the property of compositing images (use "min", "max")
            overfit_dataset_length:
            random_origin_composition (bool): randomize image composition
            mask_default_value (int): what value should the mask be by default
            img_default_value (int): if pasting on to a blank image, what value should it be by default
            dataset_length (int): how many images to generate in one epoch
            transforms_after_compositing:
            output_channel_content_names (list): list of names for each layer
            layout_sampler (LayerSampler): a LayerSampler object that will be used to sample the layers

        """
        self.size = size
        self.subdatasets = sorted(subdatasets, key=lambda x: (x.layer_position, x.name))
        self.background_img_properties = background_img_properties
        self.overfit_dataset_length = overfit_dataset_length
        self.random_origin_composition = random_origin_composition
        self.mask_default_value = mask_default_value
        self.img_default_value = img_default_value
        self.dataset_length = dataset_length
        self.transforms_after_compositing = transforms_after_compositing
        self.random_origin_function = CalculateImageOriginForCompositing(image_format="CHW") # vs. more_random_offset
        self.composite_function = composite_function
        #self.layer_contents_names = layer_contents_names if layer_contents_names else self.get_layer_contents_to_output_channel_map()
        self.layout_sampler = layout_sampler
        self.output_channel_content_names = self.get_layer_contents_to_output_channel_map(
            ) if output_channel_content_names is None else output_channel_content_names
        self.config = self.get_combination_config(
            [subdataset.layer_contents for subdataset in self.subdatasets])


        # Exclude naive masks from the channels to be visualized
        naive_mask_datasets = [x for x in self.subdatasets if type(x.mask_maker) is NaiveMask]
        self.naive_mask_channels = list(set([self.get_channel_idx(x.layer_contents) for x in naive_mask_datasets]))
        self.naive_mask_channels.sort()
        self.channels_to_be_visualized = list(set(range(len(self.output_channel_content_names))) - set(self.naive_mask_channels))



    def get_channel_idx(self, layer_name_tuple):
        return self.output_channel_content_names.index((layer_name_tuple,) if isinstance(layer_name_tuple, str) else tuple(layer_name_tuple))

    def confirm_mask_consistency(self):
        """ Make sure no naive masks are combined with non-naive masks

        Returns:

        """
        mask_types = [None] * len(self.output_channel_content_names)
        for dataset in self.subdatasets:
            for layer_name in dataset.layer_contents:
                channel = self.get_channel_idx(layer_name)
                if mask_types[channel] is None:
                    mask_types[channel] = "naive" if type(dataset.mask_maker)==NaiveMask else "normal"
                else:
                    if mask_types[channel] != type(dataset.mask_maker):
                        raise ValueError(f"Mask types are inconsistent for channel {channel} ("
                                         f"{mask_types[channel]} vs. {type(dataset.mask_maker)})")
        return mask_types


    def get_layer_contents_to_output_channel_map(self):
        """ You need to make sure that ALL unique elements are found (i.e., they might only appear in a combined thing,
                but we still have a spot for them (not essential, but for simplicity)

        Returns:

        """
        self.unique_layers = sorted(list(set([item for dataset in self.subdatasets for item in dataset.layer_contents])))
        self.combined_layers = sorted(
            list(set(
            [item.layer_contents for item in self.subdatasets if len(item.layer_contents) > 1]
            )), key=lambda x: (len(x), x)
        )
        return [(x,) if isinstance(x,str) else x for x in self.unique_layers] + self.combined_layers


    def get_combination_config(self, list_of_subdataset_layer_tuples):
        """ Given a list of tuples of like (("hw", "text"), ("noise"), ("hw", "text", "noise")), return a config,
            with mapping the tuple to the output channel index from self.output_channel_content_names
            so hw -> 0, text -> 1, noise -> 2, (hw, text) -> 3, (hw, text, noise) -> 4
            so the config would be ((0,1), 3), ((0,1,2), 4)

        Args:
            list_of_subdataset_layer_tuples (list): list of tuples of layer names

        Returns:
            tuple: config

        """

        config = []
        for layer_tuple in list_of_subdataset_layer_tuples:
            if len(layer_tuple) > 1:
                # Getting indices for the individual layers from the tuple
                channels = tuple(self.get_channel_idx(layer) for layer in layer_tuple)
                combined_channel = self.get_channel_idx(layer_tuple)
                config.append((channels, combined_channel))

        return tuple(config)

    def __len__(self):
        #return min(len(d) for d in self.subdatasets)
        return self.dataset_length


    def __getitem__(self, idx: int) -> dict:
        """Get item for a given index."""
        if self.overfit_dataset_length > 0:
            idx = idx % self.overfit_dataset_length

        if self.layout_sampler is None:
            chosen_datasets = self.subdatasets
        else:
            chosen_datasets = self.layout_sampler.sample(replacement=False)

        # sort chosen datasets by layer position to ensure background is layered first
        chosen_datasets = sorted(chosen_datasets, key=lambda x: x.layer_position)

        images_and_masks = [d[idx] for d in chosen_datasets]
        images, masks = [x["image"] for x in images_and_masks], [x["mask"] for x in images_and_masks]

        # Calculate background size
        if self.size is not None:
            bckg_size = (3, self.size, self.size)
        else:
            bckg_size = self._calculate_background_size(images)

        composite_image = torch.ones(bckg_size) * self.img_default_value
        composite_masks = torch.zeros((len(self.output_channel_content_names), *bckg_size[-2:]))
        if self.mask_default_value:
            composite_masks += self.mask_default_value

        composite_masks[len(self.unique_layers):] = -100

        layers = [x.layer_contents for x in chosen_datasets]
        combined_layers_in_current_iteration = [x.layer_contents for x in chosen_datasets if len(x.layer_contents) > 1]
        combined_layer_elements_in_current_iteration = set([item for sublist in combined_layers_in_current_iteration for item in sublist])

        for i, (dataset, (img, mask)) in enumerate(zip(chosen_datasets, zip(images, masks))):
            #start_x, start_y = self._determine_image_start_position(composite_image, img)
            start_x, start_y, details = compute_paste_origin_from_overlap_auto(composite_image,
                                                                               img,
                                                                               min_overlap=dataset.percent_overlap,
                                                                               image_format="CHW")

            # For mask, choose channel, paste at appropriate location, then take the max with existing mask
            channel = self.get_channel_idx(dataset.layer_contents)
            pasted_mask = torch.zeros(bckg_size[-2:]) + self.mask_default_value
            pasted_mask = composite_the_images_torch(pasted_mask, mask, start_x, start_y, method=torch.max)
            composite_masks[channel] = torch.max(composite_masks[channel], pasted_mask)

            if i == 0: # no composition needed
                composite_image = composite_the_images_torch(composite_image, img, start_x, start_y)
            else:
                # from hwgen.data.utils import show, display
                # from torchvision.transforms import ToTensor
                # from torch import Tensor
                # composite_image = composite_the_images_torch(background_img, img, bckg_x, bckg_y, method=torch.mul)
                # C = ConditionalContrast()
                # x = self.composite_function(composite_image, Tensor(C(img.numpy().transpose([1,2,0]))), start_x, start_y)
                # show(x)
                # if "handwriting" in dataset.layer_contents:
                #     binary_mask_from_pasted = torch.where(pasted_mask > .5, torch.tensor(1), torch.tensor(0)).unsqueeze(0).expand_as(composite_image)
                #     # compute average darkness of background where masked
                #
                #     average_background_darkness = torch.mean(composite_image[binary_mask_from_pasted == 1])
                #     average_foreground_darkness = torch.mean(img[:,][binary_mask_from_pasted == 1])
                #     if torch.
                if hasattr(dataset, "composite_function") and dataset.composite_function:
                    composite_image = dataset.composite_function(composite_image, img, start_x, start_y)
                else:
                    composite_image = self.composite_function(composite_image, img, start_x, start_y)

        # Process the combined layers.
        for combined_layer in combined_layers_in_current_iteration:
            # loop through constituent channels, taking the max
            channels = [self.get_channel_idx(layer) for layer in combined_layer] + [self.get_channel_idx(combined_layer)]
            combined_channel = self.get_channel_idx(combined_layer)
            composite_masks[combined_channel] = torch.max(composite_masks[combined_channel], torch.max(composite_masks[channels], dim=0)[0])
            #composite_masks[combined_channel] = torch.max(torch.index_select(composite_masks, 0, torch.tensor(channels)),dim=0)[0]

        for combined_layer_element in combined_layer_elements_in_current_iteration:
            channel = self.get_channel_idx(combined_layer_element)
            composite_masks[channel] = -100

        if self.transforms_after_compositing:
            composite_image = self.transforms_after_compositing(composite_image)


        return {'image': composite_image, 'mask': composite_masks}

    def _calculate_background_size(self, images: List[torch.Tensor]) -> Tuple:
        """Calculate the background size for compositing."""
        if isinstance(self.background_img_properties, str):
            if self.background_img_properties == 'max':
                bckg_size = max(images, key=lambda img: np.product(img.shape)).shape
            elif self.background_img_properties == 'min':
                bckg_size = min(images, key=lambda img: np.product(img.shape)).shape
            else:
                raise ValueError("Invalid mode. Choose from 'max', 'min'")
        elif isinstance(self.background_img_properties, (tuple, list)):
            if len(self.background_img_properties) == 2:
                bckg_size = (1, *self.background_img_properties)
            else:
                bckg_size = self.background_img_properties
        else:
            raise ValueError("Invalid background_img_properties. Choose from 'max', 'min' or specify a size tuple")
        return bckg_size

    def _determine_image_start_position(self, composite_image: torch.Tensor, img: torch.Tensor) -> Tuple[int, int]:
        """Determine the start position for the image on the composite background."""
        if self.random_origin_composition:
            start_x, start_y = self.random_origin_function(composite_image, img, .7)
        else:
            start_x, start_y = 0, 0
        return start_x, start_y
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
                #offset = more_random_offset(img.size[0],img.size[1], overlay_img.size[0],overlay_img.size[1])
                x,y,details = compute_paste_origin_from_overlap_auto(img, overlay_img, min_overlap=.8, image_format="PIL")
                offset = (x,y)
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


    all_generators = [form_generator, hw_generator, printed_text_generator]
    layout_sampler = LayerSampler(all_generators,
                                  [d.sample_weight if hasattr(d, "sample_weight") else 1 for d in all_generators]
                                  )

    aggregate_dataset = AggregateSemanticSegmentationDataset(all_generators,
                                                             background_img_properties='max',
                                                             layout_sampler=layout_sampler,
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
