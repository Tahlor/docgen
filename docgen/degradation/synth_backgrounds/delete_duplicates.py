import hashlib
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tqdm
def hash_file(file_path):
    """Generate a hash for a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def display_images(images):
    """Display images side by side."""
    plt.figure(figsize=(10, 10))
    columns = 2  # Adjust based on how many images you want to display side by side
    for i, image_path in enumerate(images):
        plt.subplot(len(images) // columns + 1, columns, i + 1)
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'File: {os.path.basename(image_path)}')
    plt.show()

def find_and_delete_duplicates(directory_path):
    """Find duplicates in a directory and display them for deletion confirmation."""
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return

    hash_dict = defaultdict(list)

    # Hash all files and group by hash
    for file_path in Path(directory_path).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".jfif", ".png", ".tiff", ".tif", ".jfif"]:
            file_hash = hash_file(file_path)
            hash_dict[file_hash].append(file_path)

    total_files = sum([len(files) for files in hash_dict.values()])
    duplicates = []
    # Identify duplicates
    for file_list in hash_dict.values():
        if len(file_list) > 1:
            duplicates.extend(file_list[1:])  # Exclude the first file

    print(f"Found {len(duplicates)} duplicates out of {total_files} total files.")
    if not CONFIRM_EACH and len(duplicates) > 0:
        input("Press any key to delete")

    # Process duplicates
    for file_list in hash_dict.values():
        if len(file_list) > 1:
            if CONFIRM_EACH:
                print(f"Potential duplicates found: {[str(file) for file in file_list]}")
                display_images(file_list)

                # Ask for confirmation before deletion
                confirm = input("Do you want to delete the duplicates? (yes/no): ")
                if confirm.lower() == 'yes':
                    for file_path in file_list[1:]:  # Skip the first file
                        os.remove(file_path)
                        print(f"Deleted {file_path}")
                    print("Duplicates deleted.")
                else:
                    print("Deletion cancelled.")
            else:
                for file_path in file_list[1:]:  # Skip the first file
                    os.remove(file_path)


# Example usage
CONFIRM_EACH = False
directory = "B:/document_backgrounds/with_backgrounds"
directory = "G:/s3/synthetic_data/resources"
find_and_delete_duplicates(directory)
