import os
from docgen.layoutgen.segmentation_dataset.create_dataset.parse_config import parse_config
from docgen.layoutgen.segmentation_dataset.create_dataset.parse_config import create_aggregate_dataset
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
from docgen.transforms.transforms_torch import ResizeAndPad, IdentityTransform, RandomResize, RandomCropIfTooBig
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

PARENT = Path(__file__).parent

"""
    # Idea: predict noise pixel levels, invert, apply soft-MASK, add to predicted HW-mask, compare to comparable GT;
    # UGH too much work, just use the HW generator for now

    Some before transforms leave a gray box
    CROP bing images
    export CONFIG somehow?
    IGNORE naive mask for now? nah you can predict it, just can't combine it
    FIX default transform - if NONE, NO TRANSFORMS SHOULD BE PERFORMED!
    CROPPING and resizing the images/seals
    Make sure background is fully covered - use the improved COMPOSITION thing
    PULLING the same image every time
    Make sure the HANDWRITING is fully composited and visible on the degraded paper :(
        # Each layer dataset can be background or foreground, do the background ones first?
    Would be good if we layer the paper on a background of white/black noise or something; or maybe use the with background dataset and segmentation (too much)
    
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

    # get all indices where the mask is naive
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

            # exclude naive_indices from the label
            visualized_img = encode_channels_to_colors(label[:, :, channels_to_be_visualized])
            to_pil(visualized_img).save(label_path_visual)

            label_formatted = (label * 255).astype('uint8').transpose([2,0,1])
            tifffile.imwrite(label_path, label_formatted, compression="jpeg", compressionargs={'level': 20})
            # file_size = os.path.getsize(label_path)
            # print(file_size)
            # # read in
            # label_read = tifffile.imread(label_path)


if __name__ == "__main__":
    save_dataset()
