from pathlib import Path
import json
import argparse
import shlex
import logging

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)


def get_collection_files_matching_base_path(base_path, exclude_base_path=False):
    """
    """
    base_path = Path(base_path)
    folder = base_path.parent
    file_name = base_path.stem
    matching_file_names = list(folder.glob(f"**/{file_name}*"))
    if exclude_base_path and base_path in matching_file_names:
        matching_file_names.remove(base_path)
    if matching_file_names:
        logger.info(f"Found files matching {str(base_path)}\n{matching_file_names}")
    else:
        logger.info(f"No files found matching {str(base_path)}")
    return matching_file_names

def get_last_file_in_collection_matching_base_path(base_path):
    """
    """
    matching_files = get_collection_files_matching_base_path(base_path)
    if len(matching_files) == 0:
        return None
    else:
        return sorted(matching_files)[-1]

def append_json_files_in_collection_matching_base_path(base_path,
                                                       exclude_base_path,
                                                       dict_format="dict",
                                                       ):
    """

    Args:
        base_path:
        exclude_base_path:
        dict_format (str): dict, COCO

    Returns:

    """
    matching_json_files = get_collection_files_matching_base_path(base_path, exclude_base_path)
    output_json = {}
    for f in matching_json_files:
        logger.info(f"Adding {f} to master dict")
        with Path(f).open("r") as f:
            new_dict = json.load(f)
        if dict_format=="dict":
            append_dict(output_json, new_dict)
        elif dict_format=="COCO":
            append_coco(output_json, new_dict)
        else:
            raise NotImplementedError(f"Unrecognized {dict_format}")
    return output_json

def append_dict(master_dict, new_dict):
    master_dict.update(new_dict)

def append_coco(master_coco, new_coco):
    if master_coco:
        master_coco["images"] += new_coco["images"]
        master_coco["annotations"] += new_coco["annotations"]
        #master_coco["categories"] += list(set( master_coco["categories"] + new_coco["categories"] ))
        master_coco["info"].update(new_coco["info"])
    else:
        master_coco.update(new_coco)

def save_dict_as_json(dic, json_path):
    logger.info(f"Saving as {json_path}")
    with Path(json_path).open("w") as f:
        json.dump(dic, f)

def create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # base_path name
    parser.add_argument("--base_path", type=str, help="Path to output directory")
    parser.add_argument("--exclude_base_path", action="store_true", help="Exclude the base path file from search")
    parser.add_argument("--dict_format", type=str, default="dict", help="Exclude the base path file from search")
    return parser

def main(args=None):
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(args))
    json_dict = append_json_files_in_collection_matching_base_path(**vars(args))
    save_dict_as_json(json_dict, args.base_path)

if __name__=="__main__":
    main()

