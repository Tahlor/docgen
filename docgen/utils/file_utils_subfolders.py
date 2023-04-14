import json
from pathlib import Path

dir_path = "/media/data/1TB/datasets/synthetic"

def get_last_completed_index(folder_path: Path) -> int:
    subfolders = [int(f.name) for f in folder_path.iterdir() if f.is_dir()]
    if not subfolders:
        return -1
    return max(subfolders)


def move_files_to_folders(subfolder: Path):
    jpg_files = list(subfolder.glob("*.jpg"))
    json_files = list(subfolder.glob("*_*.json"))

    if not jpg_files:
        return

    for jpg_file in jpg_files:
        idx = (int(jpg_file.stem) // 50000 +1) * 50000
        new_folder = subfolder / f"{idx + 1:08}"
        new_folder.mkdir(exist_ok=True)
        jpg_file.rename(new_folder / jpg_file.name)

    for json_file in json_files:
        idx = int(json_file.stem.split("_")[0]) - 1
        new_folder = subfolder / f"{idx:08}"
        json_file.rename(new_folder / json_file.name)


def main():
    base_path = Path(dir_path)
    for subfolder in base_path.iterdir():
        if subfolder.is_dir():
            last_index = get_last_completed_index(subfolder)
            print(f"Last completed index in {subfolder}: {last_index}")
            move_files_to_folders(subfolder)


if __name__ == "__main__":
    main()
