import traceback
import numpy as np
import json
from docgen.layoutgen.segmentation_dataset.create_dataset.parse_config import parse_config
from docgen.layoutgen.segmentation_dataset.create_dataset.parse_config import create_aggregate_dataset
from tqdm import tqdm
import torch
import socket
from pathlib import Path
# get number of cores
import logging
# import torch vision transforms
from docgen.image_composition.utils import encode_channels_to_colors
import tifffile
from torchvision.transforms import ToPILImage
import re
import pickle
from hwgen.data.utils import show, display

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

PARENT = Path(__file__).parent

"""
    # Idea: predict noise pixel levels, invert, apply soft-MASK, add to predicted HW-mask, compare to comparable GT;
        # UGH too much work, just use the HW generator for now

    MAJOR ISSUES:
        Some before transforms leave a gray box
        Make sure background is fully covered - use the improved COMPOSITION thing
        HAVE to do something about GREY BOXES FOR FORM ELEMENTS    
"""

def get_config(config=None):
    if not config is None:
        return config
    if socket.gethostname() == "PW01AYJG":
        path = PARENT / "config" / "default.yaml"
    elif socket.gethostname() == "Galois":
        path = PARENT / "config" / "Galois" / "default.yaml"
    else:
        raise Exception("Unknown host {}".format(socket.gethostname()))
    return path


class DatasetSaver:
    def __init__(self, config_path=None):
        self.config = parse_config(get_config(config_path))
        self.to_pil = ToPILImage()
        self.lossless = False
        self.metadata_path = self.config.output_path / "metadata.pkl"
        self.metadata_dict = self.load_metadata()

    def load_metadata(self):
        if self.metadata_path.exists():
            with open(self.metadata_path, 'rb') as file:
                print(f"Loading metadata from {self.metadata_path}")
                metadata_dict = pickle.load(file)
        else:
            metadata_dict = {}
        return metadata_dict

    def save_dataset(self):
        logger.info(f"Saving dataset to {self.config.get('output_folder')}")
        dataset = create_aggregate_dataset(self.config)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1, collate_fn=dataset.collate_fn, num_workers=self.config.workers
        )
        all_files = list(self.config.output_path.rglob("*.png"))
        logger.info(f"Found {len(all_files)} files in output folder")

        if self.config.overwrite or len(all_files) == 0:
            step = 0
        else:
            max_file = max(all_files, key=lambda x: int(re.findall(r"\d+", x.name)[0]))
            step = int(re.findall(r"\d+", max_file.name)[0])

        self.channels_to_be_visualized = dataset.channels_to_be_visualized

        try:
            self.save_all(dataloader, step)
        except Exception as e:
            traceback.print_exc()
            logger.info("Saving dataset interrupted by user")
            logger.info(e)
            raise e

        self.save_metadata()

    def save_all(self, dataloader, step):
        for i, batch in tqdm(enumerate(dataloader)):
            step += 1
            img_path = self.config.output_path / f"{step:07d}_input.png"
            label_path = self.config.output_path / f"{step:07d}_label.tiff"
            label_path_visual = self.config.output_path / f"{step:07d}_label_visual.jpg"
            inputs, labels = batch['image'], batch['mask']
            labels = labels.numpy().transpose(0, 2, 3, 1) if len(labels.shape) == 4 else labels.numpy().transpose(1, 2,
                                                                                                                  0)
            for j in range(inputs.shape[0]):
                input_img = inputs[j]
                label = labels[j]

                # save as images
                input_img = self.to_pil(input_img)
                input_img.save(img_path)

                # exclude non-naive channels (like NOISE) from the label
                visualized_img = encode_channels_to_colors(label[:, :, self.channels_to_be_visualized])
                self.to_pil(visualized_img).save(label_path_visual)

                if "active_channel_indices" in batch:
                    active_channels = batch["active_channel_indices"][j]
                else:
                    active_channels = [i for i in range(label.shape[2]) if
                                       not label[0, 0, i] == self.config.mask_null_value]

                label_formatted = label.copy().transpose([2, 0, 1])
                ignore_channel_index = batch["ignore_index"][j] if "ignore_index" in batch else None
                if self.lossless:
                    lossless_tiff_save(label_formatted, label_path, active_channels=active_channels, ignore_channel_index=ignore_channel_index)
                else:
                    jpg_tiff_save(label_formatted, label_path, active_channels=active_channels, ignore_channel_index=ignore_channel_index)

            # Collect metadata for each batch (excluding image and mask)
            batch_metadata = {k: v for k, v in batch.items() if k not in ['image', 'mask']}
            self.metadata_dict[img_path.name] = batch_metadata
            if i % 1000 == 0:
                self.save_metadata()

    def save_metadata(self):
        with open(self.metadata_path, 'wb') as file:
            pickle.dump(self.metadata_dict, file)


def prep_metadata(active_channels, has_ignore_channel=True):
    # channels
    # any channel where MASK_NULL_VALUE appears
    metadata = {'ActiveChannels': tuple(active_channels), 'HasIgnoreChannel': has_ignore_channel}
    metadata_json = json.dumps(metadata)
    return {'image_description': metadata_json}


def jpg_tiff_save(label_formatted, label_path, active_channels, ignore_channel_index=None):
    """ MUST BE UINT8, NO SAVING -100 VALUES

    When you read this in with PIL.Image and combine with ToTensor, it will scale to 0-1

    Returns:

    """
    # multiply by 255 if active, otherwise 0
    channel_dim = label_formatted.shape[0]
    metadata = prep_metadata(active_channels)

    if ignore_channel_index:
        active_channels.add(ignore_channel_index)
    bit_scale = np.asarray([255 if i in active_channels else 0 for i in range(channel_dim)])[:,None,None]
    label_formatted *= bit_scale
    label_formatted = label_formatted.astype("uint8")
    tifffile.imwrite(label_path, label_formatted,
                     compression="jpeg",
                     compressionargs={'level': 20},
                     metadata=metadata)


def lossless_tiff_save(label_formatted, label_path, active_channels, ignore_channel_index=None):
    """ IMPORTANT: When you read this in with PIL.Image and combine with ToTensor, an uint8 0-255 image will scale to 0-1
        However, an int16 will NOT be rescaled (so -100,255 will still be -100,255)

    Args:
        label_formatted:
        label_path:
        active_channels:

    Returns:

    """
    metadata = prep_metadata(active_channels)

    if ignore_channel_index:
        active_channels.add(ignore_channel_index)

    label_formatted = label_formatted.astype("int16")
    tifffile.imwrite(label_path, label_formatted,
                     compression='zstd',
                     compressionargs={"level": 9},
                     metadata=metadata)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Create a dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    d = DatasetSaver(args.config)
    d.save_dataset()
