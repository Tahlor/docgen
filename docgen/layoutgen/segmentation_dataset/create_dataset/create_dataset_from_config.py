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


def save_dataset(config_path=None):
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    config = parse_config(get_config(config_path))
    logger.info("Saving dataset to {}".format(config.get("output_folder")))
    dataset = create_aggregate_dataset(config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, num_workers=config.workers)

    # get max file in folder using regex
    all_files = list(config.output_path.rglob("*.png"))
    logger.info("Found {} files in output folder".format(len(all_files)))

    # get all indices where the mask is naive, i.e., don't visualize NOISE
    channels_to_be_visualized = dataset.channels_to_be_visualized

    if config.overwrite:
        step = 0
    elif len(all_files) > 0:
        import re
        max_file = max(all_files, key=lambda x: int(re.findall(r"\d+", x.name)[0]))
        step = int(re.findall(r"\d+", max_file.name)[0])
    else:
        step = 0

    for i, batch in tqdm(enumerate(dataloader)):
        step+=1
        img_path = config.output_path / f"{step:07d}_input.png"
        label_path = config.output_path / f"{step:07d}_label.tiff"
        label_path_visual = config.output_path / f"{step:07d}_label_visual.jpg"

        inputs, labels = batch['image'], batch['mask']

        # convert to numpy and switch to HWC, handle batch or not
        labels = labels.numpy().transpose(0, 2, 3, 1) if len(labels.shape) == 4 else labels.numpy().transpose(1, 2, 0)

        for j in range(inputs.shape[0]):
            input_img = inputs[j]
            label = labels[j]

            # save as images
            input_img = to_pil(input_img)
            input_img.save(img_path)

            # exclude non-naive channels (like NOISE) from the label
            visualized_img = encode_channels_to_colors(label[:, :, channels_to_be_visualized])
            to_pil(visualized_img).save(label_path_visual)

            if "active_channel_indices" in batch:
                active_channels = batch["active_channel_indices"][j]
            else:
                active_channels = [i for i in range(label.shape[2]) if not label[0, 0, i] == config.mask_null_value]

            label_formatted = label.copy().transpose([2,0,1])
            if False:
                lossless_tiff_save(label_formatted, label_path, active_channels=active_channels)
            else:
                jpg_tiff_save(label_formatted, label_path, active_channels=active_channels)

            # file_size = os.path.getsize(label_path)
            # print(file_size)
            # # read in
            # label_read = tifffile.imread(label_path)

def prep_metadata(active_channels):
    # channels
    # any channel where MASK_NULL_VALUE appears
    metadata = {'ActiveChannels': tuple(active_channels)}
    metadata_json = json.dumps(metadata)
    return {'image_description': metadata_json}


def jpg_tiff_save(label_formatted, label_path, active_channels):
    """ MUST BE UINT8, NO SAVING -100 VALUES

    When you read this in with PIL.Image and combine with ToTensor, it will scale to 0-1

    Returns:

    """
    # multiply by 255 if active, otherwise 0
    channel_dim = label_formatted.shape[0]
    bit_scale = np.asarray([255 if i in active_channels else 0 for i in range(channel_dim)])[:,None,None]
    label_formatted *= bit_scale
    metadata = prep_metadata(active_channels)
    label_formatted = label_formatted.astype("uint8")
    tifffile.imwrite(label_path, label_formatted,
                     compression="jpeg",
                     compressionargs={'level': 20},
                     metadata=metadata)


def lossless_tiff_save(label_formatted, label_path, active_channels):
    """ IMPORTANT: When you read this in with PIL.Image and combine with ToTensor, an uint8 0-255 image will scale to 0-1
        However, an int16 will NOT be rescaled (so -100,255 will still be -100,255)

    Args:
        label_formatted:
        label_path:
        active_channels:

    Returns:

    """
    metadata = prep_metadata(active_channels)
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
    save_dataset(args.config)
