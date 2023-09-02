from tqdm import tqdm
from docgen.layoutgen.segmentation_dataset.word_gen import HWGenerator, PrintedTextGenerator
from docgen.layoutgen.segmentation_dataset.grid_gen import GridGenerator
from docgen.layoutgen.segmentation_dataset.line_gen import LineGenerator
from docgen.layoutgen.segmentation_dataset.box_gen import BoxGenerator
from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDatasetGenerative, \
AggregateSemanticSegmentationDataset, FlattenPILGenerators, SoftMaskConfig
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

workers = multiprocessing.cpu_count() - 2
workers = int(multiprocessing.cpu_count()/2)
workers = 0

SIZE=448

def create_generating_dataset(saved_fonts_folder=None, saved_hw_folder=None, dataset_length=5000,
                              transforms_before=None,
                              transforms_after=None,
                              transforms_before2=None,):
    """ Create a dataset-generator with a single word generator and a single handwriting generator

    Args:
        saved_fonts_folder:
        saved_hw_folder:
        dataset_length:

    Returns:

    """
    if socket.gethostname() == "PW01AYJG":
        saved_fonts_folder = Path(r"G:/s3/synthetic_data/resources/fonts")
        saved_hw_folder = Path("C:/Users/tarchibald/Anaconda3/envs/docgen_windows/hwgen/resources/generated")
        saved_seals_folder = Path("G:/s3/synthetic_data/resources/seals")
        saved_preprinted_folder = Path("G:/s3/forms/HTSNet_scanned_documents")
        saved_background_folder = Path("G:/s3/synthetic_data/resources/backgrounds")
        saved_images = Path("G:/s3/synthetic_data/resources/images")


    elif socket.gethostname() == "Galois":
        saved_fonts_folder = Path("/media/EVO970/s3/datascience-computervision-l3apps/HWR/synthetic-data/python-package-resources/fonts/")
        saved_hw_folder = Path("/media/EVO970/s3/synthetic_data/python-package-resources/generated-handwriting/single-word-datasets/iam-cvl-32px-top10k-words")
        saved_seals_folder = None # TODO

    # image folder version
    # reportlab = r"G:\s3\synthetic_data\reportlab\training\train"
    # hw = r"G:\s3\synthetic_data\multiparagraph"
    # dataset1 = SemanticSegmentationDatasetImageFolder(img_dir=reportlab)
    # dataset2 = SemanticSegmentationDatasetImageFolder(img_dir=hw)

    # generated versionPairedImgLabelImageFolderDataset
    hw_generator = HWGenerator(saved_hw_folder=saved_hw_folder,)
    printed_text_generator = PrintedTextGenerator(saved_fonts_folder=saved_fonts_folder,
                                                  font_size_rng=(8, 40),
                                                  word_count_rng=(10,20)
                                                  )

    logger.info("Generating Grid, Line, and Box Generators")
    grid_generator = GridGenerator()
    line_generator = LineGenerator()
    box_generator = BoxGenerator()
    seal_generator = SemanticSegmentationDatasetGenerative(NaiveImageFolder(saved_seals_folder),
                                                              transforms_before_mask_threshold=transforms_before,
                                                              #transforms_after_mask_threshold=transforms_after,
                                                              size=SIZE)
    background_generator = SemanticSegmentationDatasetGenerative(NaiveImageFolder(saved_background_folder),
                                                                transforms_before_mask_threshold=transforms_before,
                                                                #transforms_after_mask_threshold=transforms_after,
                                                                size=SIZE)
    preprinted_generator = SemanticSegmentationDatasetGenerative(NaiveImageFolder(saved_preprinted_folder),
                                                                    transforms_before_mask_threshold=transforms_before,
                                                                    #transforms_after_mask_threshold=transforms_after,
                                                                    size=SIZE)
    image_generator = SemanticSegmentationDatasetGenerative(NaiveImageFolder(saved_images),
                                                                    transforms_before_mask_threshold=transforms_before,
                                                                    #transforms_after_mask_threshold=transforms_after,
                                                                    size=SIZE)
    image_dataset = RandomSelectorDataset({
        "seal": seal_generator,
        "image": image_generator
    })

    form_generator = FlattenPILGenerators([grid_generator, line_generator, box_generator])

    # form elements
    logger.info("Generating Semantic Segmentation Datasets")
    form_dataset = SemanticSegmentationDatasetGenerative(form_generator,
                                                         transforms_before_mask_threshold=transforms_before,
                                                         #transforms_after_mask_threshold=transforms_after,
                                                         size=SIZE)
    # get next item
    # m = next(iter(form_dataset))
    # print(m["image"].shape)
    # grid_generator.pickle_prep()

    hw_dataset = SemanticSegmentationDatasetGenerative(hw_generator,
                                                         transforms_before_mask_threshold=transforms_before2,
                                                         #transforms_after_mask_threshold=transforms_after,
                                                         size=SIZE,
                                                         soft_mask_config=SoftMaskConfig(soft_mask_threshold=.2,
                                                                                            soft_mask_steepness=45)

                                                       )
    printed_dataset = SemanticSegmentationDatasetGenerative(printed_text_generator,
                                                            transforms_before_mask_threshold=transforms_before,
                                                            #transforms_after_mask_threshold=transforms_after,
                                                            size=SIZE)

    aggregate_dataset = AggregateSemanticSegmentationDataset([form_dataset,
                                                              hw_dataset,
                                                              printed_dataset,
                                                              seal_generator,
                                                              background_generator,
                                                              preprinted_generator,
                                                              image_generator],
                                                             background_img_properties='max',
                                                             dataset_length=dataset_length,
                                                             transforms_after_compositing=transforms_after,
                                                             )

    return aggregate_dataset

def get_path():
    if socket.gethostname() == "PW01AYJG":
        path = Path(r"C:\Users\tarchibald\github\document_embeddings\document_embeddings\segmentation\dataset")
    elif socket.gethostname() == "Galois":
        path = Path(r"/media/EVO970/s3/synthetic_data/semantic_segmentation_dataset/v2_100k")
    else:
        raise Exception("Unknown host {}".format(socket.gethostname()))
    return path

def get_path_novel():
    if socket.gethostname() == "PW01AYJG":
        path = Path(r"C:\Users\tarchibald\github\document_embeddings\document_embeddings\segmentation\dataset\v5_100k")
    elif socket.gethostname() == "Galois":
        path = Path(r"/media/EVO970/s3/synthetic_data/semantic_segmentation_dataset/v3_100k")
    else:
        raise Exception("Unknown host {}".format(socket.gethostname()))
    return path


def saved_dataset(path=None):
    if path is None:
        path = get_path()
    return PairedImgLabelImageFolderDataset(path)



def save_dataset():
    from torchvision.transforms import ToPILImage
    to_pil = ToPILImage()
    output = get_path_novel()
    output.mkdir(exist_ok=True, parents=True)
    logger.info("Saving dataset to {}".format(output))
    transforms_before, transforms_after, transforms_before2 = get_transforms()
    dataset = create_generating_dataset(dataset_length=100000,
                                        transforms_before=transforms_before,
                                        transforms_after=transforms_after,
                                        transforms_before2=transforms_before2,)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn, num_workers=workers)
    # get max file in folder using regex
    all_files = list(output.rglob("*.png"))
    REGEN_ALL = True

    logger.info("Found {} files in output folder".format(len(all_files)))

    if REGEN_ALL:
        step = 0
    elif len(all_files) > 0:
        import re
        max_file = max(all_files, key=lambda x: int(re.findall(r"\d+", x.name)[0]))
        step = int(re.findall(r"\d+", max_file.name)[0])
    else:
        step = 0

    for i, batch in tqdm(enumerate(dataloader)):
        step+=1
        img_path = output / f"{step:07d}_input.png"
        label_path = output / f"{step:07d}_label.png"

        if False and img_path.exists():
            continue

        inputs, labels = batch['image'], batch['mask']

        for j in range(inputs.shape[0]):
            input = inputs[j]
            label = labels[j]

            # save as images
            to_pil(input).save(img_path)
            to_pil(label).save(label_path)

def get_transforms():
    """
        Don't use:
        BlurThreshold
        BackgroundFibrous (randomly inverts, will def mess up masks if done before, won't composite correctly also)
    """
    from docdegrade.degradation_objects import RandomDistortions, RuledSurfaceDistortions, Blur, Lighten, Blobs, \
        BackgroundMultiscaleNoise, BackgroundFibrous, Contrast
    from docgen.transforms.transforms import ResizeAndPad, IdentityTransform
    from docdegrade.torch_transforms import ToNumpy, CHWToHWC, HWCToCHW, RandomChoice, Squeeze

    #exclude = [BlurThreshold(), BackgroundFibrous()]
    before = [RandomDistortions(), RuledSurfaceDistortions()]
    either = [Blur(), Lighten(), Contrast()]
    before += either
    after = [Blobs(), BackgroundMultiscaleNoise(), BackgroundFibrous(), Contrast()]

    transforms_before = transforms.Compose([
        # ocnvert PIL IMage to numpy
        ToNumpy(),
        RandomChoice(before),
        ResizeAndPad(SIZE, 32) if SIZE else IdentityTransform(), # this converts to Tensor
    ])
    transforms_before2 = transforms.Compose([
        ToNumpy(),
        ResizeAndPad(SIZE, 32) if SIZE else IdentityTransform(), # this converts to Tensor
    ])

    transforms_after = transforms.Compose([
        #lambda x: x.permute(1, 2, 0),
        CHWToHWC(),
        Squeeze(),
        RandomChoice(after),
        HWCToCHW(),
        #lambda x: x.numpy(),
        #transforms.ToTensor()
    ])
    return transforms_before, transforms_after, transforms_before2
    #return None,None


if __name__ == "__main__":
    save_dataset()
