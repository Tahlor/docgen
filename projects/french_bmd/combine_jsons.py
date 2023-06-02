from docgen.dataset_utils import consolidate_data
from pathlib import Path

if __name__=='__main__':
    path = "/media/EVO970/data/synthetic/french_bmd_0092"
    for json_type in "OCR","COCO":
        consolidate_data( [p for p in Path(path).glob(f"{json_type}*.json")] )
