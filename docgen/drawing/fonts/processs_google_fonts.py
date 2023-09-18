from pathlib import Path
from typing import List, Dict, Any, Tuple
import languages_public_pb2
from tqdm import tqdm


def parse_pb_file(file_path: Path) -> Tuple[Any, List[Path]]:
    """
    Parses a .pb file based on the provided schema and returns its content.

    Args:
        file_path (Path): The path to the .pb file.

    Returns:
        Tuple[Any, List[Path]]: A Protocol Buffer object containing the parsed data
                                and a list of .ttf file paths in the same directory.
    """
    with file_path.open("rb") as file:
        pb_object = languages_public_pb2.LanguageProto()
        pb_object.ParseFromString(file.read())

        # Get all .ttf files in the same directory as the .pb file
        ttf_files = list(file_path.parent.glob("*.ttf"))
        return pb_object, ttf_files


def load_all_pb_files(directory: Path) -> List[Tuple[Any, List[Path]]]:
    """
    Loads all .pb files in a directory (and its subdirectories).

    Args:
        directory (Path): The directory to search in.

    Returns:
        List[Tuple[Any, List[Path]]]: A list of tuples, where each tuple contains
                                     a Protocol Buffer object and associated .ttf files.
    """
    pb_objects_and_files = []

    for file in tqdm(directory.rglob("*.pb"), desc="Parsing .pb files"):
        try:
            pb_object_and_files = parse_pb_file(file)
            pb_objects_and_files.append(pb_object_and_files)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    return pb_objects_and_files


def filter_languages(pb_objects: List[Any], criterion: str) -> List[Any]:
    """
    Filters parsed .pb objects based on a given criterion.

    Args:
        pb_objects (List[Any]): The parsed .pb objects.
        criterion (str): The filtering criterion.

    Returns:
        List[Any]: A list of .pb objects that match the given criterion.
    """
    return [obj for obj, _ in pb_objects if obj.name == criterion]


if __name__ == "__main__":
    directory_path = Path("G:/fonts/fonts-main/fonts-main")
    all_pb_objects_and_files = load_all_pb_files(directory_path)

    filtered_objects = filter_languages(all_pb_objects_and_files, "English")
    for obj, ttf_files in filtered_objects:
        print(obj)
        for ttf_file in ttf_files:
            print(f"\t- {ttf_file.absolute()}")
