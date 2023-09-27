from pathlib import Path
import logging
import yaml
from enum import Enum
from hwgen.data.utils import show
from tqdm import tqdm
from docgen.layoutgen.segmentation_dataset.word_gen import HWGenerator, PrintedTextGenerator
from docgen.layoutgen.segmentation_dataset.grid_gen import GridGenerator
from docgen.layoutgen.segmentation_dataset.line_gen import LineGenerator
from docgen.layoutgen.segmentation_dataset.box_gen import BoxGenerator
from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDatasetGenerative, \
AggregateSemanticSegmentationDataset, FlattenPILGenerators, SoftMask, Mask, NaiveMask, SemanticSegmentationDatasetImageFolder
from docgen.layoutgen.segmentation_dataset.paired_image_folder_dataset import PairedImgLabelImageFolderDataset
import torch
import socket
from pathlib import Path
# get number of cores
import multiprocessing
import logging
import random
import numpy as np
# import torch vision transforms
from torchvision import transforms
from PIL import Image
from docgen.layoutgen.segmentation_dataset.image_paste_gen import CompositeImages
from docgen.layoutgen.segmentation_dataset.image_folder import NaiveImageFolder
from docgen.layoutgen.segmentation_dataset.gen import RandomSelectorDataset
from docgen.image_composition.utils import encode_channels_to_colors
from docgen.layoutgen.segmentation_dataset.utils.dataset_sampler import LayerSampler
import tifffile
from torchvision.transforms import ToTensor
from docgen.windows.utils import map_drive
from docdegrade.torch_transforms import ToNumpy, CHWToHWC, HWCToCHW, RandomChoice, Squeeze
from docgen.transforms.transforms_torch import ResizeAndPad, IdentityTransform, RandomResize, RandomCropIfTooBig, \
    ResizeLongestSide, RandomEdgeCrop
from docgen.layoutgen.segmentation_dataset.preprinted_form_gen import PreprintedFormElementGenerator
from typing import List, Union, Dict, Any
from docdegrade.degradation_objects import RandomDistortions, RuledSurfaceDistortions, Blur, Lighten, Blobs, \
    BackgroundMultiscaleNoise, BackgroundFibrous, Contrast, ConditionalContrast
from easydict import EasyDict as edict
from docgen.image_composition.utils import seamless_composite, composite_the_images_torch, CompositerTorch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class TransformType(Enum):
    TONUMPY = ToNumpy
    RANDOMCHOICE = RandomChoice
    IDENTITYTRANSFORM = IdentityTransform
    RANDOMRESIZE = RandomResize
    RANDOMCROPIFTOOBIG = RandomCropIfTooBig
    RANDOMDISTORTIONS = RandomDistortions
    RULEDSURFACEDISTORTIONS = RuledSurfaceDistortions
    BLUR = Blur
    LIGHTEN = Lighten
    BLOBS = Blobs
    BACKGROUND_MULTISCALE_NOISE = BackgroundMultiscaleNoise
    BACKGROUND_FIBROUS = BackgroundFibrous
    CONTRAST = Contrast
    CONDITIONALCONTRAST = ConditionalContrast
    SQUEEZE = Squeeze
    CHWTOHWC = CHWToHWC
    HWCTOCHW = HWCToCHW
    RESIZEANDPAD = ResizeAndPad
    TOTENSOR = ToTensor
    RESIZELONGESTSIDE = ResizeLongestSide
    RANDOMEDGECROP = RandomEdgeCrop

class DatasetType(Enum):
    HWGENERATOR = HWGenerator
    NAIVEIMAGEFOLDER = NaiveImageFolder
    SEMANTICSEGMENTATIONDATASETGENERATIVE = SemanticSegmentationDatasetGenerative
    PREPRINTEDFORMELEMENTGENERATOR = PreprintedFormElementGenerator
    PRINTEDTEXTGENERATOR = PrintedTextGenerator

class CompositeFunctions(Enum):
    """ Because these are functions and not classes, you'll need to do:
            member = getattr(CompositeFunctions, 'SEAMLESS_COMPOSITE')

    """
    SEAMLESS_COMPOSITE = seamless_composite
    COMPOSITE_THE_IMAGES_TORCH = composite_the_images_torch
    COMPOSITERTORCH = CompositerTorch
    TORCHMAX = torch.max
    TORCHMIN = torch.min
    TORCHMUL = torch.mul

class Masks(Enum):
    #SoftMask, Mask, NaiveMask
    SOFTMASK = SoftMask
    MASK = Mask
    NAIVEMASK = NaiveMask

def parse_config(config_file_path: Path):
    with open(config_file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config.get("drive_mappings"):
        for drive_letter, target_path in config.get("drive_mappings").items():
            map_drive(target_path, drive_letter)
    config = edict(config)
    config.output_path = Path(config.get("output_path"))
    (config.output_path).mkdir(exist_ok=True, parents=True)

    if not config.get("workers") and config.get("workers") != 0:
        config.workers = multiprocessing.cpu_count() - 2

    return config


def create_individual_dataset(config):
    dataset_type = config.get("type")
    sample_weight = config.get("sample_weight")
    transform_defs = config.get("transforms")
    transforms = create_transforms(transform_defs)
    base_dataset_kwargs = config.get("base_dataset_kwargs", {})

    # composite function
    composite_function = parse_composite_function(config.get("composite_function", None))

    # get mask.type
    mask_config = config.get("mask", {"type": "softmask", "kwargs": {}})
    mask_type = Masks[mask_config["type"].upper()].value
    mask_maker = mask_type(**mask_config.get("kwargs", {}))

    dataset_cls = DatasetType[dataset_type.upper()].value
    dataset = dataset_cls(**base_dataset_kwargs)

    segmentation_dataset = SemanticSegmentationDatasetGenerative(layer_contents=config.get("layer_contents"),
                                                 generator=dataset,
                                                 name=config.get("name"),
                                                 percent_overlap=config.get("percent_overlap", 0.7),
                                                 mask_maker=mask_maker,
                                                 transforms_before_mask_threshold=transforms,
                                                                 sample_weight=sample_weight,
                                                                 composite_function=composite_function,
                                                                 layer_position=config.get("layer_position", 50)
                                                                 )
    return segmentation_dataset


def create_transforms(transforms_list: List[Union[str, Dict[str, Any]]]) -> List[Any]:
    """
    Create a list of transform objects based on the configuration list.

    Args:
        transforms_list (List[Union[str, Dict[str, Any]]]): List of transform configurations from YAML.

    Returns:
        List[Any]: List of initialized transform objects.
    """
    if not transforms_list:
        return []
    transform_objects = []

    for transform_item in transforms_list:
        if isinstance(transform_item, str):
            # Simple transform
            transform_cls = TransformType[transform_item.upper()].value
            transform_objects.append(transform_cls())
        elif isinstance(transform_item, dict):
            # Complex transform (with arguments or meta-transform like RandomChoice)
            for key, value in transform_item.items():
                if key == 'RandomChoice':
                    inner_transform_objects = [TransformType[item.upper()].value() for item in value]
                    transform_objects.append(RandomChoice(inner_transform_objects))
                else:
                    transform_cls = TransformType[key.upper()].value
                    if value is None:
                        transform_objects.append(transform_cls())
                    elif isinstance(value, dict):
                        transform_objects.append(transform_cls(**value))
                    else:
                        transform_objects.append(transform_cls(value))

    return transform_objects


def parse_composite_function(composite_function_def):
    if composite_function_def is None:
        return None
    elif isinstance(composite_function_def, dict):
        composite_function_class_name = composite_function_def["type"]
        composite_function_args = composite_function_def.get("kwargs", {})
        method = composite_function_def.get("method", {})
        composite_function_class = CompositeFunctions[composite_function_class_name.upper()].value
        method = CompositeFunctions[method.upper()].value
        composite_function = composite_function_class(method=method,**composite_function_args)
    else:
        composite_function = getattr(CompositeFunctions, composite_function_def.upper())
    return composite_function

def create_aggregate_dataset(config):
    generators = []
    overrides = config.get("dataset_override", None)
    for dataset_config in config['datasets']:
        if overrides and not dataset_config.get("name") in overrides:
            continue
        dataset = create_individual_dataset(dataset_config)
        if dataset is None:
            input(f"Problem with dataset config: {dataset_config}")
            continue
        transforms = create_transforms(dataset_config.get("transforms", []))
        dataset.transforms = transforms
        generators.append(dataset)

    layout_sampler = LayerSampler(generators,
                                  [d.sample_weight if hasattr(d, "sample_weight") and d.sample_weight else 1 for d in generators],
                                    **config.get("layout_sampler_kwargs", {}),
                                  )
    after_transforms = config.get("transforms_after_compositing", [])

    composite_function_def = config.get("composite_function", "seamless_composite")
    composite_function = parse_composite_function(composite_function_def)

    aggregate_dataset = AggregateSemanticSegmentationDataset(generators,
                                                             background_img_properties='max',
                                                             dataset_length=config.get("dataset_length", 100000),
                                                             transforms_after_compositing=after_transforms,
                                                             layout_sampler=layout_sampler,
                                                             size=config.get("output_img_size", 448),
                                                             composite_function=composite_function
                                                             )
    return aggregate_dataset

if __name__ == "__main__":
    config_path = Path("./config/default.yaml")
    config = parse_config(config_path)
    create_aggregate_dataset(config)
