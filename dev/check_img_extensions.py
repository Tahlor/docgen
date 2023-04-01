from argparse import ArgumentParser
import imghdr
from pathlib import Path

def get_file_type(file_path):
    """Determine the actual file type of an image."""
    return imghdr.what(file_path)

def scan_directory(path):
    """Scan the directory and return files with incorrect extensions, excluding 'BACKUP' directory."""
    incorrect_files = []
    path = Path(path)
    for file_path in path.rglob('*'):
        if "BACKUP" in file_path.parts:
            continue
        if file_path.is_file():
            actual_type = get_file_type(file_path)
            if actual_type == 'jpeg':
                actual_type = 'jpg'
            if actual_type and file_path.suffix != f'.{actual_type}':
                incorrect_files.append((file_path, actual_type))
    return incorrect_files

def confirm_and_rename(files):
    """Propose changes and rename files upon user confirmation."""
    for file_path, actual_type in files:
        new_name = file_path.with_suffix(f'.{actual_type}')
        print(f'Rename: {file_path} -> {new_name}')

    if input('Proceed with these changes? (y/n): ').lower() == 'y':
        for file_path, actual_type in files:
            new_name = file_path.with_suffix(f'.{actual_type}')
            file_path.rename(new_name)
        print('Files have been renamed.')
    else:
        print('No changes made.')

def create_parser():
    """Create an argparse parser."""
    parser = ArgumentParser(description='Check and correct file extensions in a directory.')
    parser.add_argument('path', type=str, help='Path to the directory to scan')
    return parser

def main(args=None):
    parser = create_parser()
    parsed_args = parser.parse_args(args.split()) if args else parser.parse_args()
    incorrect_files = scan_directory(parsed_args.path)
    if incorrect_files:
        confirm_and_rename(incorrect_files)
    else:
        print('All file extensions are correct.')

if __name__ == '__main__':
    args = "C:/Users/tarchibald/Downloads/DIBCO"
    main(args)
