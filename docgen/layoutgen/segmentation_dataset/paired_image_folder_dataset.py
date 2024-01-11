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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

class PairedImgLabelImageFolderDataset(GenericDataset):

    def __init__(self, img_dir,
                 label_dir=None,
                 transform_list=None,
                 paired_mask_transform_obj=None,
                 max_uniques=None,
                 max_length_override=None,
                 label_name_pattern="label_{}.png",
                 **kwargs,
                 ):
        """

        Args:
            img_dir:
            label_dir:
            transform_list:
            paired_mask_transform_obj: takes in img, label, returns degraded img, label
            max_uniques:
            max_length_override:
            label_name_pattern:
        """
        super().__init__(
            max_uniques=max_uniques,
            max_length_override=max_length_override,
            transform_list=transform_list,
            collate_fn=PairedImgLabelImageFolderDataset.collate_fn,
        )
        if label_dir is None:
            label_dir = img_dir
        self.max_uniques = int(max_uniques) if max_uniques else None
        self.label_dir = label_dir
        self.img_dir = img_dir
        self.label_name_pattern = label_name_pattern
        self.max_length_override = int(max_length_override) if max_length_override else None
        self.path_database = self.process_img_file_list(img_dir, label_dir)

        if len(self.path_database) == 0:
            raise ValueError(f"No images found in {img_dir}")

        self.paired_mask_transform_obj = paired_mask_transform_obj
        self.transform_list = transform_list
        if self.transform_list is None:
            # just convert it to tensor, compose it
            self.transform_list = Compose([transforms.ToTensor()]
                                          )
        elif isinstance(self.transform_list, list):
            self.transform_list = Compose(self.transform_list)

    def process_img_file_list(self, img_dir, label_folder):
        """
        Safer to parse the image file to get the index, create a dictionary
        """
        label_folder = Path(label_folder)
        imgs = Path(img_dir).glob("*input*.png")

        path_database = []
        self.file_id_to_idx = {}
        for img_path in imgs:
            # Extract the id fro    m the filename.
            # The regular expression will match both "12345_input" and "input_12345".
            match = re.search(r"(\d+)", img_path.stem)
            if match:
                id = match.group(1)

                label_name_pattern = self.label_name_pattern.format(id)
                label_path = label_folder / label_name_pattern
                path_database.append({"img_path": img_path,
                                      "label_path": label_path})
                self.file_id_to_idx[id] = len(path_database) - 1
                if self.max_length_override and len(path_database) >= self.max_length_override:
                    break
            else:
                logger.warning(f"No id found in filename {img_path}")

        self.length = len(path_database)
        return path_database

    def _get(self, idx):
        while True:
            img_path = "img_path not found"
            try:
                idx = self._validate_idx(idx)
                paths = self.path_database[idx]
                img_path, label_path = paths["img_path"], paths["label_path"]

                # load from png and convert to tensor
                img = Image.open(img_path).convert("RGB")
                label_dict = read_label_img(label_path)
                label = label_dict["label"]

                if self.transform_list is not None:
                    img = self.transform_list(img)
                    label = self.transform_list(label)
                if self.paired_mask_transform_obj is not None and self.paired_mask_transform_obj:
                    img, label = self.paired_mask_transform_obj(img, label)

                return_dict = {'image': img,
                        'mask': label,
                        "name": img_path.stem,
                        "ignore_channel_idx": label.shape[0]-1 if label_dict["metadata"]["HasIgnoreChannel"] else None
                        }
                if "label_active_channels" in label_dict:
                    return_dict["label_active_channels"] = label_dict["label_active_channels"]

                return return_dict
            except Exception as e:
                logger.exception(f"Error loading image {idx} {img_path}")
                idx += 1
                if not self.continue_on_error:
                    raise e

    def __getitem__(self, idx):
        """

        Args:
            idx:

        Returns:

        """
        return self._get(idx)

    @staticmethod
    def collate_fn(batch, tensor_keys=["image", "mask"]):
        keys = batch[0].keys()
        collated_batch = {}

        for key in keys:
            if tensor_keys and key in tensor_keys:
                collated_batch[key] = torch.stack([item[key] for item in batch], dim=0)
            else:
                collated_batch[key] = [item[key] for item in batch]

        collated_batch["active_channel_mask"] = PairedImgLabelImageFolderDataset.generate_active_channel_mask(collated_batch["mask"],
                                                                                                              collated_batch["label_active_channels"])
        if collated_batch["ignore_channel_idx"] is not None:
            # if False:
            #     mask = np.ones_like(collated_batch["mask"], dtype=bool)
            #     mask[:, collated_batch["ignore_channel_idx"]] = 1
            # else:
            collated_batch["mask"][:,:-1] *= collated_batch["active_channel_mask"][:,:-1]


        else:
            collated_batch["mask"] *= collated_batch["active_channel_mask"]
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

def read_label_img(path):
    path = Path(path)
    # if TIFF, open as TIFF
    if path.suffix == ".tiff":
        with TiffFile(path) as tif:
            label_chw = tif.asarray()
            metadata = parse_metadata(tif, keys_to_find="ActiveChannels")
        label_active_channels = metadata.get("ActiveChannels")
        label = np.transpose(label_chw, (1, 2, 0))
    else:
        label = Image.open(path)
        label_active_channels = metadata = None
    return {
        "label": label,
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