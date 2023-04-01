from pathlib import Path
from projects.french_bmd.french_bmd_layoutgen import main, parser, save_dataset
import json
from easydict import EasyDict as edict
from docgen.dataset_utils import draw_gt_layout, save_json, ocr_dataset_to_coco, load_json

def process(path):
    path = Path(path)
    ocr_dataset = load_json(path)

    opts = edict({
        "coco_path": path.parent / "COCO.json",
    })

    # ocr_dataset, i, opts, format="OCR", add_iteration_suffix=True
    save_dataset(ocr_dataset, 0, opts, format="COCO", add_iteration_suffix=False)

if __name__ == "__main__":
    # load ocr config
    path = "G:/s3/synthetic_data/FRENCH_BMD/v0-/OCR.json"
    process(path)