from tqdm import tqdm
import argparse
from docgen.dataset_utils import draw_boxes_sections_COCO, load_json
import sys
import socket
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def parser_args(args=None):
    parser = argparse.ArgumentParser(description='Process COCO format data')

    parser.add_argument('--coco_json_path', type=str, required=True,
                        help='The COCO format file')
    parser.add_argument('--category_id', type=int, default=None,
                        help='Category id to be processed')
    parser.add_argument('--background_img', type=str, default=None,
                        help='Background image file')
    parser.add_argument('--draw_boxes', type=bool, default=True,
                        help='Whether to draw boxes or not')
    parser.add_argument('--draw_segmentations', type=bool, default=False,
                        help='Whether to draw segmentations or not')
    parser.add_argument('--image_root', type=str, default=".",
                        help='Root directory of images')
    parser.add_argument('--image_ids', nargs='+', type=str, default=None,
                        help='List of image ids to be processed')

    if args is not None:
        import shlex
        args = parser.parse_args(shlex.split(args))
    else:
        args = parser.parse_args()

    return args

def read_json_or_npy(json_path):
    logger.info(f"Reading {json_path}")
    if json_path.endswith(".json"):
        with open(json_path, "r") as f:
            data = load_json(f)
    elif json_path.endswith(".npy"):
        import numpy as np
        data = np.load(json_path, allow_pickle=True).item()
    else:
        raise ValueError("Unknown file type: {}".format(json_path))
    logger.info("Done loading")
    return data

def main(args=None):
    args = parser_args(args)

    # read in the coco format file /
    coco_format = read_json_or_npy(args.coco_json_path)

    for image_id in tqdm(args.image_ids):
        logger.info(f"Processing image_id: {image_id}")
        img = draw_boxes_sections_COCO(coco_format,
                             args.category_id,
                             args.background_img,
                             args.draw_boxes,
                             args.draw_segmentations,
                             args.image_root,
                             image_id=image_id)
        img.show()

if __name__ == "__main__":
    args1 = r"""--coco_json_path 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.0.0\COCO.json' 
                   --image_root 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.0.0' 
                   --image_ids 0000001 0010000 0019999"""

    args = r"""--coco_json_path 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.1.0\COCO.json' 
               --image_root 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.1.0' 
               --image_ids 0000001 0052999 0053000 0053001 0099999"""

    args = r"""--coco_json_path 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.1.0\singles\COCO_1000.json' 
               --image_root 'G:\s3\synthetic_data\FRENCH_BMD\FRENCH_BMD_LAYOUTv2.1.0' 
               --image_ids 0000001 0000999 """

    args = args.replace("\n","")
    if not args or len(sys.argv) > 1:
        args = sys.argv[1:]
    main(args)
