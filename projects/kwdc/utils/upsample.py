from pathlib import Path
import cv2
import numpy as np

# Define the files using Path
files = [
    Path(r"C:\Users\tarchibald\github\docgen\projects\kwdc\kwdc-synthetic-data-forms\images\Guam_Arriola_SSS.j2k_page_2.j2k"),
    Path(r"C:\Users\tarchibald\github\docgen\projects\kwdc\kwdc-synthetic-data-forms\images\Guam_Arriola_SSS.j2k_page_1.j2k")
]

# Calculate ratio
ratio = 2175/870

for file_path in files:
    # Read the image
    img = cv2.imread(str(file_path))
    
    if img is None:
        print(f"Error reading {file_path}")
        continue
    
    # Calculate new dimensions
    height, width = img.shape[:2]
    new_height = int(height * ratio)
    new_width = int(width * ratio)
    
    # Resize image
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Create new filename using Path
    new_file_name = file_path.stem + "_UPSAMPLED.jpg"  # Changed to .jpg for better compatibility
    output_path = file_path.parent / new_file_name
    
    # Save the image
    cv2.imwrite(str(output_path), resized)
    print(f"Saved upsampled image to {output_path}")