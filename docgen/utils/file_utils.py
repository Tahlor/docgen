from pathlib import Path
import json
import argparse
import shlex
import logging

logger = logging.getLogger(__name__)

def get_collection_files_matching_base_path(base_path):
    """
    """
    folder = Path(base_path).parent
    file_name = Path(base_path).stem
    matching_file_names = list(folder.glob(f"**/{file_name}*"))
    if matching_file_names:
        logger.info(f"Found files matching {base_path}\n{matching_file_names}")
    else:
        logger.info(f"No files found matching {base_path}")
    return matching_file_names

def get_last_file_in_collection_matching_base_path(base_path):
    """
    """
    matching_files = get_collection_files_matching_base_path(base_path)
    if len(matching_files) == 0:
        return None
    else:
        return sorted(matching_files)[-1]

def append_json_files_in_collection_matching_base_path(base_path):
    """
    """
    matching_json_files = get_collection_files_matching_base_path(base_path)
    output_json = {}
    for f in matching_json_files:
        logger.info(f"Adding {f} to master dict")
        with Path(f).open("r") as f:
            output_json.update(json.load(f))
    return output_json

def save_dict_as_json(dic, json_path):
    logger.info(f"Saving as {json_path}")
    with Path(json_path).open("w") as f:
        json.dump(dic, f)

def create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    # base_file name
    parser.add_argument("--base_file", type=str, help="Path to output directory")
    return parser

def main(args=None):
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(args))
    json_dict = append_json_files_in_collection_matching_base_path(args.base_file)
    save_dict_as_json(json_dict, args.base_file)


if __name__=="__main__":
    main()

