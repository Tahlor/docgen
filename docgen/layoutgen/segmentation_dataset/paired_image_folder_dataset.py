from docgen.utils.utils import time_function
from docgen.layoutgen.segmentation_dataset.cache import Cache
import inspect
import albumentations as A
from itertools import chain
import warnings
import torch
import numpy as np
import re
from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationCompositionDataset
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Compose
import logging
from tifffile import imread, imsave, TiffFile
import json
from docgen.datasets.generic_dataset import GenericDataset
from hwgen.data.utils import show, display
import yaml
from docgen.utils.channel_mapper import SimpleChannelMapper
import random
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class MetaPairedImgLabelImageFolderDataset(Dataset):
    def __init__(self, img_dataset_config_paths, weights=None, shuffle=True, *args, **kwargs):
        if weights is None:
            weights = [1] * len(img_dataset_config_paths)
        self.img_dataset_config_paths, self.weights = self.filter_out_0_weights(img_dataset_config_paths, weights)
        self.datasets = [PairedImgLabelImageFolderDataset(img_dataset_config_paths=config, *args, **kwargs) for config in self.img_dataset_config_paths]
        self.length = sum(len(d) for d in self.datasets)
        self.counter = 0
        self.shuffle = shuffle

    def filter_out_0_weights(self, img_dataset_config_paths, weights):
        configs = [config for i,config in enumerate(img_dataset_config_paths) if weights[i] > 0]
        weights = self.normalize_weights([w for w in weights if w > 0])
        return configs, weights

    def normalize_weights(self, weights):
        if weights is None:
            weights = [1] * len(self.datasets)
        weights = [p / sum(weights) for p in weights]
        return weights

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.shuffle:
            chosen_dataset = random.choices(self.datasets, self.weights)[0]
            return chosen_dataset[idx % len(chosen_dataset)]
        else:
            for dataset in self.datasets:
                if idx < len(dataset):
                    return dataset[idx]
                idx -= len(dataset)


class PairedImgLabelImageFolderDataset(GenericDataset):

    def __init__(self, img_dirs=None,
                 label_dirs=None,
                 transform_list=None,
                 paired_mask_transform_obj=None,
                 max_uniques=None,
                 max_length_override=None,
                 use_ignore_channel_if_available=True,
                 active_channels=None,
                 img_dataset_config_paths=None,
                 img_glob_patterns="*input.png",
                 label_name_patterns="{}_label.tiff",
                 img_id_regexes="(\d+)",
                 output_channel_names=None,
                 input_channel_class_names=None,
                 use_cache=True,
                 cache_size=300,
                 max_cache_reuse=10,
                 **kwargs,
                 ):
        """
        Constructor for the class.

        Args:
            img_dirs: [preferably put in img_dataset_config_paths]
                list of dirs with imgs
            label_dirs: [preferably put in img_dataset_config_paths]
                list of dirs with label imgs
            transform_list:
            paired_mask_transform_obj: takes in img, label, returns degraded img, label.
            max_uniques:
            max_length_override:
            use_ignore_channel_if_available: if the label has an ignore channel, use it to replace values with -1
                on other channels; these should be excluded from the loss
                the ignore channel is the last channel
                whether or not exists should be specified in the tiff metadata.

            active_channels (list): a list of the channels that are active
                for instance, maybe you have an image with only "text" and "background" channels
                you don't want to compute loss for an "hw" channel, since it is not active in the current dataset
                so you would set active_channels = [0, 1]
                datasets made with docgen will usually specify which channels are active in the metadata.

            img_dataset_config_paths (str / Path): a path to a yaml file with the img dataset configs (see below);
                it will prioritize using this over the options below.

            img_glob_patterns (list or str): [preferably put in img_dataset_config_paths]
                a list of patterns to use to find the image file, like "*input.png".

            label_name_patterns (list or str or None): [preferably put in img_dataset_config_paths]
                a list of patterns to use to find the label file, like "label_{}.png"
                None: if they have the same name but are in different folders
                However, if the extension is different, you should still use this.

            img_id_regexes (list or str or None): [preferably put in img_dataset_config_paths]
                a list of regexes to use to extract the image id from the filename, like "\d+"
                The corresponding label will be found by replacing the {} in label_name_patterns with the image id
                Used in conjunction with label_name_patterns.

            output_channel_names (list): [text, handwriting, form_elements, noise ...]

            input_channel_class_names (list of lists): [preferably put in img_dataset_config_paths]
                for each folder, what the corresponding channel_names are
                these will be remapped to match the output_channel_names
        """

        super().__init__(
            max_uniques=max_uniques,
            max_length_override=max_length_override,
            transform_list=transform_list,
            collate_fn=PairedImgLabelImageFolderDataset.collate_fn,
        )
        self.max_length_override = int(max_length_override) if max_length_override else None
        self.max_uniques = int(max_uniques) if max_uniques else None
        if self.max_uniques or self.max_length_override:
            self.max_uniques = min(self.max_uniques, self.max_length_override)

        self.img_dataset_config_paths = [img_dataset_config_paths] if isinstance(img_dataset_config_paths, (str, Path)) else img_dataset_config_paths
        self.number_of_datasets = len(self.img_dataset_config_paths) if self.img_dataset_config_paths is not None else len(img_dirs)
        parallel_configs = self.process_dataset_configs(img_dataset_config_paths=self.img_dataset_config_paths,
                                                        img_dirs=img_dirs,
                                                        label_dirs=label_dirs,
                                                        img_glob_patterns=img_glob_patterns,
                                                        label_name_patterns=label_name_patterns,
                                                        img_id_regexes=img_id_regexes,
                                                        input_channel_class_names=input_channel_class_names, )

        self.img_dirs, self.label_dirs, self.img_id_regexes, self.img_glob_patterns, self.label_name_patterns, self.input_channel_class_names, self.img_dataset_configs = parallel_configs

        self.path_database = self.process_img_file_list(img_dirs=self.img_dirs,
                                                        label_dirs=self.label_dirs,
                                                        img_glob_patterns=self.img_glob_patterns,
                                                        label_name_patterns=self.label_name_patterns,
                                                        img_id_regexes=self.img_id_regexes
                                                        )

        self.output_channel_names = output_channel_names
        self.channel_mappers = self.create_channel_mappers(
            output_channel_names=output_channel_names,
            input_channel_names_for_each_folder=self.input_channel_class_names,
        )

        self.active_channels = active_channels
        self.use_ignore_channel_if_available = use_ignore_channel_if_available

        if len(self.path_database) == 0:
            raise ValueError(f"No images found in {img_dirs}")

        self.paired_mask_transform_obj = paired_mask_transform_obj
        self.transform_list = transform_list
        if self.transform_list is None:
            # just convert it to tensor, compose it
            self.transform_list = Compose([transforms.ToTensor()]
                                          )
        if use_cache:
            print(f"USING CACHE!!! Size: {cache_size} Reuse: {max_cache_reuse}")
            self.cache_object = self.build_cache(cache_size, max_cache_reuse)
        else:
            self.cache_object = None
    def build_cache(self, cache_size=100, max_cache_reuse=5):
        return Cache(cache_size, max_cache_reuse)

    def create_channel_mappers(self, output_channel_names, input_channel_names_for_each_folder):
        if output_channel_names is not None and input_channel_names_for_each_folder is not None:
            channel_mappers = []
            for i in range(self.number_of_datasets):
                if input_channel_names_for_each_folder[i] is not None:
                    channel_mapper = SimpleChannelMapper(output_channel_names=output_channel_names,
                                                     input_channel_names=input_channel_names_for_each_folder[i])
                else:
                    channel_mapper = None
                channel_mappers.append(channel_mapper)
        else:
            channel_mappers = [None] * self.number_of_datasets

        return channel_mappers

    def get_all_files_from_folder_as_list(self, folders, img_glob_pattern="*"):
        folders = [folders] if isinstance(folders, (str, Path)) else folders
        imgs = list(chain.from_iterable(Path(dir).glob(f"{img_glob_pattern}") for dir in folders))
        return imgs

    def process_dataset_configs(self,
                                img_dataset_config_paths,
                                img_dirs,
                                label_dirs,
                                img_glob_patterns,
                                label_name_patterns,
                                img_id_regexes,
                                input_channel_class_names):
        """ Loop through the img_dataset_config_paths if not None
            If not None, load the img_dataset_config_paths from the yaml file

            These are all either
                                img_dir,
                                label_dir,
                                img_glob_patterns,
                                label_name_patterns,
                                img_id_regexes
        """
        assert img_dirs is not None or img_dataset_config_paths is not None, "Must provide img_dirs or img_dataset_config_paths"

        def get_path(reference_path, path):
            if str(path).startswith("."):
                return Path(reference_path).parent / path
            return path

        if isinstance(img_id_regexes, str) or img_id_regexes is None:
            img_id_regexes = [img_id_regexes] * self.number_of_datasets
        if isinstance(label_name_patterns, str) or label_name_patterns is None:
            label_name_patterns = [label_name_patterns] * self.number_of_datasets
        if isinstance(img_glob_patterns, str) or img_glob_patterns is None:
            img_glob_patterns = [img_glob_patterns] * self.number_of_datasets
        if isinstance(img_dirs, str) or img_dirs is None:
            img_dirs = [img_dirs] * self.number_of_datasets
        if isinstance(label_dirs, str):
            img_dirs = [label_dirs] * self.number_of_datasets
        is_list_of_lists = isinstance(input_channel_class_names, list) and \
                           isinstance(input_channel_class_names[0], list)

        img_dataset_configs = [None] * self.number_of_datasets

        if input_channel_class_names is None or is_list_of_lists:
            input_channel_class_names = [input_channel_class_names] * self.number_of_datasets

        if label_dirs is None:
            label_dirs = img_dirs[:]

        if img_dataset_config_paths:
            for idx in range(self.number_of_datasets):
                img_dataset_config_path = Path(img_dataset_config_paths[idx])
                if img_dataset_config_path is not None:
                    if img_dataset_config_path.exists():
                        img_dataset_config = yaml.safe_load(img_dataset_config_path.read_text())
                        print(f"Loaded img_dataset_config from {img_dataset_config_path}\n{img_dataset_config}")
                        img_dirs[idx] = get_path(img_dataset_config_path, img_dataset_config.get("img_dir", img_dirs[idx]))
                        label_dirs[idx] = get_path(img_dataset_config_path, img_dataset_config.get("label_dir", label_dirs[idx]))
                        img_glob_patterns[idx] = img_dataset_config.get("img_glob_pattern", img_glob_patterns[idx])
                        label_name_patterns[idx] = img_dataset_config.get("label_name_pattern", label_name_patterns[idx])
                        img_id_regexes[idx] = img_dataset_config.get("img_id_regex", img_id_regexes[idx])
                        input_channel_class_names[idx] = img_dataset_config.get("input_channel_class_names", input_channel_class_names[idx])
                        img_dataset_configs[idx] = img_dataset_config
                    else:
                        raise Exception(f"Couldn't find {img_dataset_config_path}")

        img_id_regexes = [re.compile(regex) if regex is not None else None for regex in img_id_regexes]

        assert len(img_dirs) == len(label_dirs) == len(img_id_regexes) == len(img_glob_patterns) == len(label_name_patterns) == len(input_channel_class_names), \
            "All lists must have the same length"
        return img_dirs, label_dirs, img_id_regexes, img_glob_patterns, label_name_patterns, input_channel_class_names, img_dataset_configs


    def process_img_file_list(self,
                              img_dirs,
                              label_dirs,
                              img_glob_patterns,
                              label_name_patterns,
                              img_id_regexes,
                              ):
        """ The strategy is: get all the INPUT files in the IMG folder, then look for the corresponding label in the
                LABEL folder (possibly using a different pattern). If the label has a different extension, include
                that in the label format


        img_dirs: List of image directories.
        label_dirs: List of label directories.
        strategies: List of strategies ('regex' or 'direct') for each directory.
        regexes: Optional list of regex patterns for each directory if using 'regex' strategy.
        """

        path_database = []
        self.file_id_to_idx = {}

        for folder_idx, (img_dir,
                         label_dir,
                         regex,
                         label_name_pattern,
                         img_glob_pattern,
                         ) in \
                enumerate(zip(img_dirs,
                              label_dirs,
                              img_id_regexes,
                              label_name_patterns,
                              img_glob_patterns,
                              )
                          ):
            imgs = self.get_all_files_from_folder_as_list(img_dir, img_glob_pattern)

            for img_path in imgs:
                if not img_path.is_file():
                    continue
                if regex is not None:
                    # Extract the id from the filename using regex.
                    match = regex.search(img_path.name)
                    try:
                        img_id = match.group(1)
                    except:
                        continue

                else:
                    img_id = img_path.stem

                label_name = label_name_pattern.format(img_id)
                label_path = Path(label_dir) / label_name

                if label_path.exists():
                    path_database.append({"img_path": img_path,
                                          "label_path": label_path,
                                          "folder_idx": folder_idx})
                    self.file_id_to_idx[img_id] = len(path_database) - 1

                    if self.max_uniques and len(path_database) >= self.max_uniques:
                        break
                else:
                    logger.warning(f"No match found for file {img_path}")

        self.length = len(path_database)
        return path_database

    #@time_function
    def _load_idx(self, idx):
        idx = self._validate_idx(idx)
        paths = self.path_database[idx]
        img_path, label_path, folder_idx = paths["img_path"], paths["label_path"], paths["folder_idx"]

        # load from png and convert to tensor
        img = Image.open(img_path).convert("RGB")
        label_dict = read_label_img(label_path)
        label = label_dict["label"]
        img = np.array(img)
        label = np.array(label)

        config = self.img_dataset_configs[folder_idx]
        gt_options = config.get("gt_options", {}) if config is not None else {}
        label = process_label(label, gt_options=gt_options)

        return img, label, label_dict, img_path, label_path, folder_idx, config, gt_options

    def load_idx(self, idx):
        if self.cache_object is not None:
            return_tuple = self.cache_object.get(idx, random_ok=True)
            if return_tuple is None:
                return_tuple = self._load_idx(idx)
                self.cache_object.put(idx, return_tuple)
            return return_tuple
        else:
            return self._load_idx(idx)

    #@time_function
    def _get(self, idx):
        while True:
            try:
                img, label, label_dict, img_path, label_path, folder_idx, config, gt_options = self.load_idx(idx)
                orig_height, orig_width = img.shape[:2]

                if self.transform_list is not None:
                    img, label = run_transforms(self.transform_list, img, label)
                if self.paired_mask_transform_obj is not None and self.paired_mask_transform_obj:
                    img, label = self.paired_mask_transform_obj(img, label)

                metadata = label_dict.get("metadata")
                if metadata and metadata.get("HasIgnoreChannel") and self.use_ignore_channel_if_available:
                    mask_with_ignore = label[:-1]
                    ignore_mask = (label[-1] == 1)  # Create the boolean mask from the last channel
                    #print(f"IGNORE MASK ACTIVE {ignore_mask.sum()} pixels in {img_path}")
                    mask_expanded = ignore_mask.unsqueeze(0).expand_as(
                        mask_with_ignore)  # Expand the mask to match dimensions

                    # Exclude form_elements from ignore
                    exclude_form_elements_index = config.get("input_channel_class_names",[]).index(["form_elements"])
                    if exclude_form_elements_index is not None:
                        mask_expanded[exclude_form_elements_index] = False  # Exclude specific class from ignore

                    mask_with_ignore[mask_expanded] = -1  # Replace with -1 where the mask is True

                    label = mask_with_ignore

                if label_dict.get("label_active_channels") is not None:
                    active_channels = label_dict["label_active_channels"]
                else:
                    active_channels = self.active_channels

                if self.channel_mappers[folder_idx] is not None:
                    channel_mapper = self.channel_mappers[folder_idx]
                    label = channel_mapper(label)
                    if active_channels is None:
                        active_channels = list(range(label.shape[0]))

                    active_channels = channel_mapper.convert_idx_config(active_channels)
                else:
                    print("No mapper???")

                # show(img, title=f"Image {img_path.parent.name}/{img_path.stem}")
                # for i in range(label.shape[0]):
                #     show(label[i], title=f"Layer: {channel_mapper.output_channel_names[i]}"
                #                          f"Range: {label[i].min()} - {label[i].max()}")


                return_dict = {
                        'image': img,
                        'mask': label,
                        "name": img_path.stem,
                        "label_active_channels": active_channels,
                        "path": img_path,
                        }

                h,w = img.shape[-2:]
                if (h,w) != (orig_height, orig_width):
                    return_dict["original_hw"] = orig_height, orig_width

                return return_dict
            except Exception as e:
                logger.exception(f"Error loading image {idx} {img_path}")
                idx += 1
                if not self.continue_on_error:
                    raise e

    @property
    def label_output_class_names(self):
        return self.output_channel_names

    def label_input_class_names(self, idx):
        idx = self._validate_idx(idx)
        paths = self.path_database[idx]
        channel_mapper = self.channel_mappers[paths["folder_idx"]]
        if channel_mapper is not None:
            return channel_mapper.input_channel_names
        else:
            return None

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:

        """
        return self._get(idx)

    @staticmethod
    #@time_function
    def collate_fn(batch, tensor_keys=["image", "mask"]):
        if isinstance(batch[0], list):
            return batch
        keys = batch[0].keys()
        collated_batch = {}

        for key in keys:
            if tensor_keys and key in tensor_keys:
                if batch[0].get(key) is not None:
                    collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                collated_batch[key] = [item.get(key) for item in batch]

        # if "label_active_channels" in collated_batch:
        #     collated_batch["active_channel_mask"] = PairedImgLabelImageFolderDataset.generate_active_channel_mask(collated_batch["mask"],
        #                                                                                              collated_batch["label_active_channels"])
        #     collated_batch["mask"] *= collated_batch["active_channel_mask"]
        return collated_batch

    @staticmethod
    def generate_active_channel_mask(input_img, active_channels_list=None):
        """
            Convert a list of active channels of length batch_size to a mask of shape batch_size, num_channels, 1, 1
            designating which channels are active. We want a mask we can use to 0 out inactive channels.

        Args:
            input_img: we get an input of batch, channels, h, w
            active_channels_list: len batch, each is a tuple with the channels that should be active

        Returns:

        """
        batch_size, num_channels, height, width = input_img.shape
        mask = torch.zeros(batch_size, num_channels, 1, 1)

        # Set active channel indices to 1
        if active_channels_list and active_channels_list[0] is not None:
            for idx, active_channels in enumerate(active_channels_list):
                mask[idx, list(active_channels), ...] = 1
        else:
            mask[:, :, ...] = 1
        return mask

def process_label(label, gt_options={}):
    if label.dtype == bool:
        label = label.astype(np.float32)  # to 0's and 1's

    if gt_options.get("rescale_to_01") or label.dtype==np.uint8:
        label = label.astype(np.float32) / label.max()

    if gt_options.get("invert"):
        label = 1 - label

    # add a channel dim if needed in numpy
    if len(label.shape) == 2:
        label = label[..., np.newaxis]

    return label

def run_transforms(transform_list, img, label):
    if isinstance(transform_list, A.Compose):
        if transform_list.additional_targets:
            img_dict = transform_list(image=img, mask=label)
            img, label = img_dict["image"], img_dict["mask"]
        else:
            img = transform_list(image=img)["image"]
    elif isinstance(transform_list, Compose): # not compatible with albumentations additional targets, so loop through it ourselves and use run_transforms on it
        for transform in transform_list.transforms:
            img, label = run_transforms(transform, img, label)
    elif isinstance(transform_list, list):
        for transform in transform_list:
            img, label = run_transforms(transform, img, label)
    elif isinstance(transform_list, A.OneOf):
        img, label = transform_list(image=img, mask=label)
    # else if callable
    elif callable(transform_list):
        if "label" in inspect.signature(transform_list).parameters:
            img, label = transform_list(image=img, label=label)
        else:
            img = transform_list(img)
    else:
        raise ValueError(f"Unknown transform type {transform_list}")
    return img, label

def read_label_img(path):
    path = Path(path)
    # if TIFF, open as TIFF
    loaded_img = None
    if path.suffix.startswith(".tif"):
        try:
            with TiffFile(path) as tif:
                label_chw = tif.asarray()
                metadata = parse_metadata(tif, keys_to_find="ActiveChannels")
            label_active_channels = metadata.get("ActiveChannels")
            loaded_img = np.transpose(label_chw, (1, 2, 0))
        except Exception as e:
            warnings.warn(f"Couldn't open {path}\n{e}")
    if loaded_img is None:
        loaded_img = Image.open(path)
        label_active_channels = metadata = None

    return {
        "label": loaded_img,
        "metadata": metadata,
        "label_active_channels": label_active_channels
    }


def parse_metadata(tif_binary, keys_to_find="ActiveChannels"):
    """
    Parse the image description metadata from a TIFF file.

    Args:
        tiff_path (str): The path to the TIFF file.

    Returns:
        dict: A dictionary containing the parsed metadata.
    """
    # Attempt to retrieve the image description metadata
    image_description = tif_binary.pages[0].tags.get('ImageDescription', None)
    if image_description:
        metadata_json = image_description.value
        # Deserialize the JSON string back into a Python dictionary
        metadata = json.loads(metadata_json)
        img_description = metadata.get("image_description")
        if img_description:
            img_description_dict = parse_image_description(img_description, keys_to_find)
            if isinstance(img_description_dict, dict):
                metadata.update(img_description_dict)
        return metadata

def parse_image_description(img_description, keys_to_find):
    try:
        metadata = json.loads(img_description)
        keys_to_find = [keys_to_find] if isinstance(keys_to_find, str) else keys_to_find

        for key in keys_to_find:
            if key in metadata:
                metadata[key] = metadata[key]
        return metadata
    except:
        warnings.warn(f"Could not parse metadata: {img_description}")
        return img_description