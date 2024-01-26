import os
import re
import glob
import time
from pathlib import Path

# Path to the directory
directory = "C:/Users/tarchibald/github/document_embeddings/document_embeddings/segmentation/dataset/v5.3_100k"  # Assuming the directory structure is replicated here for the test

# Compiled regex for filtering directories and files with "input" in their names
input_pattern = re.compile(r".*input.*")
directory = Path(directory)

# Method 1: Using glob
start_time = time.time()
#glob_folders = glob.glob(os.path.join(directory, "*input*/"))
#glob_files = [file for folder in glob_folders for file in glob.glob(os.path.join(folder, "*")) if "input" in file]
glob_files = [file for file in directory.glob("*input*")]
glob_duration = time.time() - start_time

# Method 2: Using compiled regex
start_time = time.time()
regex_folders = [folder for folder in os.listdir(directory) if input_pattern.match(folder) and os.path.isdir(os.path.join(directory, folder))]
regex_files = []
for file in directory.glob("*input*"):
    if input_pattern.match(file.stem):
        regex_files.append(directory / file)
regex_duration = time.time() - start_time

print(glob_duration, regex_duration, len(glob_files), len(regex_files))  # Output durations and file counts

