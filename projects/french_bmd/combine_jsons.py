# from docgen.dataset_utils import consolidate_data
# from pathlib import Path
#
# if __name__=='__main__':
#     path = "/media/EVO970/data/synthetic/french_bmd_0005"
#     for json_type in "OCR","COCO":
#         consolidate_data( [p for p in Path(path).glob(f"{json_type}*.json")] )
import sys
import socket
from docgen.datasets.utils.combine_jsons import FindJSONS

if __name__ == "__main__":
    args = []

    if socket.gethostname() == "PW01AYJG":
        input_folder = f"/media/EVO970/data/synthetic/french_bmd_0092/"
        input_folder = f"G:/s3/synthetic_data/one_line/english"
        args = f""" {input_folder} --output_folder . --overwrite
        """
    elif socket.gethostname() == "Galois":
        input_folder = f"/media/EVO970/data/synthetic/french_bmd_0005"
        args = f"{input_folder} --overwrite --save_npy"

    if sys.argv[1:]:
        args = None

    hf = FindJSONS(args)
    hf.main()

