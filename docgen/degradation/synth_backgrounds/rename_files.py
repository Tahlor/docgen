import argparse
from pathlib import Path
from tqdm import tqdm

def rename_files_numerically(directory: Path):
    for folder in directory.iterdir():
        if folder.is_dir():
            counter = 0
            print(f"Renaming files in folder: {folder}")
            for file in tqdm(sorted(folder.glob('**/*'))):
                if file.suffix.lower() not in [".jpg", ".jpeg", ".jfif", ".png", ".tiff", ".tif", ".jfif"]:
                    continue
                if file.is_file() and not file.stem.isdigit():
                    failed=True
                    while failed:
                        try:
                            new_name = f"{folder}/{counter:06d}{file.suffix}"
                            file.rename(new_name)
                            failed=False
                        except:
                            pass
                        counter += 1

def main(args=None):
    parser = argparse.ArgumentParser(description="Rename files numerically within each folder.")
    parser.add_argument("directory", type=str, help="Directory to start renaming files in.")
    if args is not None:
        import shlex
        args = parser.parse_args(shlex.split(args))
    else:
        args = parser.parse_args()

    root_dir = Path(args.directory)
    if not root_dir.exists() or not root_dir.is_dir():
        raise ValueError(f"The path specified ({root_dir}) does not exist or is not a directory.")

    rename_files_numerically(root_dir)

if __name__ == "__main__":
    #folder = "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/CROPPED/paper_only"
    folder = "B:/document_backgrounds/handwriting"
    main(folder)
