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

def append_json_files_in_collection_matching_base_path(base_path, exclude_base_path):
    """
    """
    matching_json_files = get_collection_files_matching_base_path(base_path, exclude_base_path)
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
    parser.add_argument("--exclude_base_file", action="store_true", help="Exclude the base path file from search")
    return parser

def main(args=None):
    parser = create_parser()
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(shlex.split(args))
    json_dict = append_json_files_in_collection_matching_base_path(args.base_file, args.exclude_base_file)
    save_dict_as_json(json_dict, args.base_file)


if __name__=="__main__":
    main()

