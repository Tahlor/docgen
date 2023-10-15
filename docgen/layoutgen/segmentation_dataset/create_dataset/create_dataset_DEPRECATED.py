from tqdm import tqdm
from docgen.layoutgen.writing_generators import HWGenerator, PrintedTextGenerator
from docgen.layoutgen.writing_generators import GridGenerator
from docgen.layoutgen.writing_generators import LineGenerator
from docgen.layoutgen.writing_generators import BoxGenerator
from docgen.layoutgen.segmentation_dataset.semantic_segmentation import SemanticSegmentationDatasetGenerative, \
AggregateSemanticSegmentationDataset, FlattenPILGenerators, SoftMask, NaiveMask
from docgen.layoutgen.segmentation_dataset.paired_image_folder_dataset import PairedImgLabelImageFolderDataset
import torch
import socket
from pathlib import Path
# get number of cores
import multiprocessing
import logging
# import torch vision transforms
from torchvision import transforms
from docgen.datasets.image_folder import NaiveImageFolder
from docgen.image_composition.utils import encode_channels_to_colors
from docgen.layoutgen.segmentation_dataset.utils.dataset_sampler import LayerSampler
import tifffile
from docgen.windows.utils import map_drive
from docdegrade.torch_transforms import ToNumpy, RandomChoice
from docgen.transforms.transforms_torch import RandomResize, RandomCropIfTooBig
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

workers = multiprocessing.cpu_count() - 2
workers = int(multiprocessing.cpu_count()/2)
workers = 0

SIZE=448

"""
    # Idea: predict noise pixel levels, invert, apply soft-MASK, add to predicted HW-mask, compare to comparable GT;
    # UGH too much work, just use the HW generator for now

    Some before transforms leave a gray box
    CROP bing images
    export CONFIG somehow?
    IGNORE naive mask for now? nah you can predict it, just can't combine it
    FIX default transform - if NONE, NO TRANSFORMS SHOULD BE PERFORMED!
    IMPLEMENT CONFIG
    IMPLEMENT actual transforms of resizing etc.
    CROPPING and resizing the images/seals
    Make sure background is fully covered - use the improved COMPOSITION thing
    PULLING the same image every time
    Make sure the HANDWRITING is fully composited and visible on the degraded paper :(
        # Each layer dataset can be background or foreground, do the background ones first?
    Tiffs are still too big
    Would be good if we layer the paper on a background of white/black noise or something; or maybe use the with background dataset and segmentation (too much)
    
    
"""

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
        saved_preprinted_text_and_elements_folders = Path("G:/s3/forms/HTSNet_scanned_documents")
        map_drive(target_path=r"G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle", drive_letter="B:")
        saved_background_folder = Path("B:/document_backgrounds/paper_only")
        saved_images = Path("G:/s3/synthetic_data/resources/images")
        saved_pdf_text_images = [#"G:/s3/forms/PDF/IRS/text/images",
                                 #"G:/s3/forms/PDF/GDC/text/images",
                                 "G:/s3/forms/PDF/OPM/text/images",
                                 "G:/s3/forms/PDF/SSA/text/images"]
        saved_pdf_form_elements = ["G:/s3/forms/PDF/IRS/other_elements/images",
                                   "G:/s3/forms/PDF/GDC/other_elements/images",
                                   "G:/s3/forms/PDF/OPM/other_elements/images",
                                   "G:/s3/forms/PDF/SSA/other_elements/images"]
        saved_fbmd_handwriting = "G:/s3/french_bmd/transcription_pairs/transcriptions/version3/train_images"

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

    # dataset2 = SemanticSegmentationDatasetImageFolder(img_dir=saved_images, layer_contents="images")

    seal_generator = SemanticSegmentationDatasetGenerative(layer_contents=("seals","text"),
                                                           generator=NaiveImageFolder(saved_seals_folder),
                                                              transforms_before_mask_threshold=[],
                                                              size=SIZE)

    background_generator = SemanticSegmentationDatasetGenerative(layer_contents=("noise"),
                                                                generator=NaiveImageFolder(saved_background_folder),
                                                                mask_maker=NaiveMask(),
                                                                transforms_before_mask_threshold=[RandomResize(min_scale=.5, max_scale=2.0, min_pixels=SIZE),
                                                                                                  RandomCropIfTooBig(size=SIZE)],
                                                                size=SIZE)

    preprinted_generator = SemanticSegmentationDatasetGenerative(generator=NaiveImageFolder(saved_preprinted_text_and_elements_folders),
                                                                    layer_contents=("text", "form_elements"),
                                                                    transforms_before_mask_threshold=[RandomCropIfTooBig(size=SIZE)],
                                                                    size=SIZE)
    # image_generator = SemanticSegmentationDatasetGenerative(generator=NaiveImageFolder(saved_images),
    #                                                                 layer_contents=("images"),
    #                                                                 transforms_before_mask_threshold=transforms_before,
    #                                                                 size=SIZE)
    # pdf_text_generator = SemanticSegmentationDatasetGenerative(generator=NaiveImageFolder(saved_pdf_text_images),
    #                                                                 layer_contents=("text"),
    #                                                                 transforms_before_mask_threshold=transforms_before,
    #                                                                 size=SIZE)
    # pdf_form_elements_generator = SemanticSegmentationDatasetGenerative(generator=NaiveImageFolder(saved_pdf_form_elements),
    #                                                                 layer_contents=("form_elements"),
    #                                                                 transforms_before_mask_threshold=transforms_before,
    #                                                                 size=SIZE)
    # fbmd_handwriting_generator = SemanticSegmentationDatasetGenerative(generator=NaiveImageFolder(saved_fbmd_handwriting),
    #                                                                 layer_contents=("hw"),
    #                                                                 transforms_before_mask_threshold=transforms_before,
    #                                                                 size=SIZE)

    form_generator = FlattenPILGenerators([grid_generator, line_generator, box_generator])

    # form elements
    logger.info("Generating Semantic Segmentation Datasets")
    form_dataset = SemanticSegmentationDatasetGenerative(layer_contents=("form_elements"),
                                                         generator=form_generator,
                                                         transforms_before_mask_threshold=transforms_before,
                                                         #transforms_after_mask_threshold=transforms_after,
                                                         size=SIZE)
    # get next item
    # m = next(iter(form_dataset))
    # print(m["image"].shape)
    # grid_generator.pickle_prep()

    hw_dataset = SemanticSegmentationDatasetGenerative(generator=hw_generator,
                                                            layer_contents=("hw"),
                                                             transforms_before_mask_threshold=transforms_before2,
                                                             size=SIZE,
                                                             mask_maker=SoftMask(soft_mask_threshold=.2,
                                                             soft_mask_steepness=45)

                                                       )
    printed_dataset = SemanticSegmentationDatasetGenerative(generator=printed_text_generator,
                                                            layer_contents=("text"),
                                                            transforms_before_mask_threshold=transforms_before,
                                                            size=SIZE)

    # all_generators2 = [form_dataset,
    #                     hw_dataset,
    #                     printed_dataset,
    #                     background_generator,
    #                     preprinted_generator,
    #                     pdf_form_elements_generator,
    #                     pdf_text_generator,
    #                     seal_generator,
    #                     image_generator,
    #                                                          ],

    all_generators = [preprinted_generator, form_dataset,  hw_dataset, printed_dataset, background_generator, seal_generator]
    layout_sampler = LayerSampler(all_generators,
                                  [d.sample_weight if hasattr(d, "sample_weight") else 1 for d in all_generators]
                                  )

    aggregate_dataset = AggregateSemanticSegmentationDataset(all_generators,
                                                             background_img_properties='max',
                                                             dataset_length=dataset_length,
                                                             transforms_after_compositing=transforms_after,
                                                             layout_sampler=layout_sampler,

                                                             )

    print(aggregate_dataset.config)
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
        path = Path(r"C:\Users\tarchibald\github\document_embeddings\document_embeddings\segmentation\dataset\v6_100k")
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
    transforms_before, transforms_after, transforms_before2, transforms_before_image_folders = get_transforms()
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
        label_path = output / f"{step:07d}_label.tiff"
        label_path_visual = output / f"{step:07d}_label_visual.jpg"

        if False and img_path.exists():
            continue

        inputs, labels = batch['image'], batch['mask']

        # convert to numpy and switch to HWC, handle batch or not
        labels = labels.numpy().transpose(0, 2, 3, 1) if len(labels.shape) == 4 else labels.numpy().transpose(1, 2, 0)

        for j in range(inputs.shape[0]):
            input_img = inputs[j]
            label = labels[j]

            # save as images
            input_img = to_pil(input_img)
            input_img.save(img_path)
            visualized_img = encode_channels_to_colors(label)
            to_pil(visualized_img).save(label_path_visual)
            tifffile.imwrite(label_path, label)

def get_transforms():
    """
        Don't use:
        BlurThreshold
        BackgroundFibrous (randomly inverts, will def mess up masks if done before, won't composite correctly also)
    """
    from docdegrade.degradation_objects import RandomDistortions, RuledSurfaceDistortions, Blur, Lighten, Blobs, \
        BackgroundMultiscaleNoise, BackgroundFibrous, Contrast
    from docgen.transforms.transforms_torch import ResizeAndPad, IdentityTransform

    #exclude = [BlurThreshold(), BackgroundFibrous()]
    before = [RandomDistortions(), RuledSurfaceDistortions()]
    either = [Blur(), Lighten(), Contrast()]
    before += either
    after = [Blobs(), BackgroundMultiscaleNoise(), BackgroundFibrous(), Contrast()]

    transforms_before_generative = transforms.Compose([
        # convert PIL IMage to numpy
        ToNumpy(),
        # transforms.ToTensor(),
        RandomChoice(before),
        ResizeAndPad(SIZE, 32) if SIZE else IdentityTransform(), # this converts to Tensor
    ])

    transforms_before_image_folders = transforms.Compose([
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

        # NO DEGRADATION
        # CHWToHWC(),
        # ToNumpy(),
        # Squeeze(),
        # RandomChoice(after),
        # ToTensor(),

        # HWCToCHW(),
        #lambda x: x.numpy(),
        #transforms.ToTensor()
    ])
    return transforms_before_generative, transforms_after, transforms_before2, transforms_before_image_folders
    #return None,None


if __name__ == "__main__":
    save_dataset()
