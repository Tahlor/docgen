from docgen.utils.yaml_utils import MySafeYAMLDumper, ReallySafeDumper
import yaml
from enum import Enum
from docgen.layoutgen.segmentation_dataset.layer_generator.word_gen import HWGenerator, PrintedTextGenerator
from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationCompositionDatasetGenerative, \
AggregateSemanticSegmentationCompositionDataset
import torch
from pathlib import Path, WindowsPath, PosixPath
# get number of cores
import multiprocessing
import logging
# import torch vision transforms
from docgen.datasets.image_folder import NaiveImageFolder
from docgen.layoutgen.segmentation_dataset.utils.dataset_sampler import LayerSampler
from torchvision.transforms import ToTensor
from docgen.windows.utils import map_drive, unmap_drive
from docdegrade.torch_transforms import ToNumpy, CHWToHWC, HWCToCHW, RandomChoice, Squeeze
from docgen.transforms.transforms_torch import ResizeAndPad, IdentityTransform, RandomResize, RandomCropIfTooBig, \
    ResizeLongestSide, RandomBottomLeftEdgeCrop, CropBorder, RandomFlipOrMirror
from docgen.transforms.transforms_torch import *
from docgen.layoutgen.segmentation_dataset.layer_generator.preprinted_form_gen import PreprintedFormElementGenerator
from typing import List, Union, Dict, Any
from docdegrade.degradation_objects import RandomDistortions, RuledSurfaceDistortions, Blur, Lighten, Blobs, \
    BackgroundMultiscaleNoise, BackgroundFibrous, Contrast, ConditionalContrast, ColorJitter, RandomRotate
from easydict import EasyDict as edict
from docgen.image_composition.utils import seamless_composite, composite_the_images_torch, CompositerTorch
from docgen.datasets.utils.dataset_filters import RejectIfEmpty, RejectIfTooManyPixelsAreBelowThreshold
from docgen.layoutgen.segmentation_dataset.masks import Mask, NaiveMask, SoftMask, GrayscaleMask
def easydict_representer(dumper, data):
    return dumper.represent_dict(data.items())

def to_yaml_path(dumper, data):
    print(f"to_yaml_path called for {data}")  # Debug print
    return dumper.represent_scalar('tag:yaml.org,2002:str', str(data))

def tuple_to_list_representer(dumper, data):
    return dumper.represent_list(list(data))

yaml.add_representer(tuple, tuple_to_list_representer)
yaml.add_representer(edict, easydict_representer)
Path.to_yaml = to_yaml_path
yaml.add_representer(Path, Path.to_yaml)
yaml.add_representer(WindowsPath, Path.to_yaml)
yaml.add_representer(PosixPath, Path.to_yaml)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class TransformType(Enum):
    TONUMPY = ToNumpy
    RANDOMCHOICE = RandomChoice
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
    RANDOMBOTTOMLEFTEDGECROP = RandomBottomLeftEdgeCrop
    COLORJITTER = ColorJitter
    CROPBORDER = CropBorder
    RANDOMROTATE = RandomRotate
    RANDOMFLIPORMIRROR = RandomFlipOrMirror
    IDENTITYTRANSFORM = IdentityTransform

class DatasetType(Enum):
    HWGENERATOR = HWGenerator
    NAIVEIMAGEFOLDER = NaiveImageFolder
    SEMANTICSEGMENTATIONCOMPOSITIONDATASETGENERATIVE = SemanticSegmentationCompositionDatasetGenerative
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
    GRAYSCALEMASK = GrayscaleMask

class DatasetFilters(Enum):
    REJECTIFEMPTY = RejectIfEmpty
    REJECTIFTOOMANYPIXELSAREBELOWTHRESHOLD = RejectIfTooManyPixelsAreBelowThreshold

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

    config.mask_null_value = config.get("mask_null_value", -100)

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

    if dataset_type.upper() == "NAIVEIMAGEFOLDER":
        filter_configs = config.get("filters", [])
        if filter_configs:
            instantiated_filters = [DatasetFilters[list(f.keys())[0].upper()].value(**list(f.values())[0], dataset_name=config.name) for f in filter_configs]
            base_dataset_kwargs["filters"] = instantiated_filters
        base_dataset_kwargs["transform_list"] = transforms
        base_dataset_kwargs["return_format"] = "dict"
        transforms = None

    dataset = dataset_cls(**base_dataset_kwargs)

    segmentation_dataset = SemanticSegmentationCompositionDatasetGenerative(layer_contents=config.get("layer_contents"),
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
        composite_function_args = composite_function_def.get("kwargs") or {}
        method = composite_function_def.get("method") or {}
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
        transforms = create_transforms(dataset_config.get("transforms") or [])
        dataset.transforms = transforms
        generators.append(dataset)

    layout_sampler = LayerSampler(generators,
                                  [d.sample_weight if hasattr(d, "sample_weight") and d.sample_weight else 1 for d in generators],
                                    **(config.get("layout_sampler_kwargs") or {}),
                                  )
    after_transforms = config.get("transforms_after_compositing") or []

    composite_function_def = config.get("composite_function") or "seamless_composite"
    composite_function = parse_composite_function(composite_function_def)

    aggregate_dataset = AggregateSemanticSegmentationCompositionDataset(generators,
                                                             background_img_properties='max',
                                                             dataset_length=config.get("dataset_length", 100000),
                                                             transforms_after_compositing_img_only=after_transforms,
                                                             transforms_after_compositing_img_and_mask=None,
                                                             layout_sampler=layout_sampler,
                                                             size=config.get("output_img_size", 448),
                                                             composite_function=composite_function,
                                                             mask_null_value=config.mask_null_value,
                                                             background_bounding_boxes_pkl_path=config.get("background_bounding_boxes_pkl_path", None),
                                                             use_ignore_index=config.get("use_ignore_index", True),
                                                             )
    # save out 1) the config used to create this dataset and 2) aggregate_dataset.config
    config.combined_channel_mapping = aggregate_dataset.config
    config.output_channel_content_names = aggregate_dataset.output_channel_content_names
    save_config(config.output_path / "config.yaml", config)
    return aggregate_dataset

def save_config(config_path, config):
    with open(config_path, 'w') as file:
        safe_config = {
            "output_channel_content_names":config.output_channel_content_names
        }
        yaml.dump(safe_config, file)

def convert_config(config):
    """
    Convert non-serializable items to strings within a configuration dictionary.
    """
    if isinstance(config, dict):
        return {k: convert_config(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_config(item) for item in config]
    try:
        yaml.safe_dump(config)
        return config
    except yaml.representer.RepresenterError:
        return str(config)  # Convert non-serializable objects to strings


if __name__ == "__main__":
    config_path = Path("./config/default.yaml")
    config = parse_config(config_path)
    create_aggregate_dataset(config)
