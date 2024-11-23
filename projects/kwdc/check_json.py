import json
from collections import Counter
from pathlib import Path
import argparse

def handle_args(args, parser):
    if args is None:
        args = parser.parse_args()
    elif isinstance(args, list):
        args = parser.parse_args(args)
    elif isinstance(args, dict):
        # convert dict to namespace
        args = argparse.Namespace(**args)
    elif isinstance(args, str):
        import shlex
        args = parser.parse_args(shlex.split(args))
    return args

def check_json_uniqueness(data):
    """
    Check if all image IDs in the ground truth JSON are unique.
    Returns a tuple of (is_unique, duplicate_ids).
    """
    # Get all image IDs
    image_ids = [img['id'] for img in data['images']]
    
    # Count occurrences of each ID
    id_counts = Counter(image_ids)
    
    # Find duplicates (IDs that appear more than once)
    duplicates = {id: count for id, count in id_counts.items() if count > 1}
    
    return {
        'duplicate_count': len(duplicates),
        'duplicates': duplicates,
        'is_unique': len(duplicates) == 0,
        'count': len(image_ids)
    }

def check_json_image_correspondence(data, json_path):
    """
    Check if there's a 1:1 relationship between JSON image IDs and image files.
    Returns a dictionary with the analysis results.
    """
    # Get image folder path (same directory as JSON)
    image_folder = Path(json_path).parent
    
    # Get all image IDs from JSON
    json_ids = {img['id'] for img in data['images']}
    
    # Get all image files from folder (assuming they match the ID pattern)
    image_files = set()
    for ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
        image_files.update(f.stem for f in image_folder.glob(f'*{ext}'))
    
    # Check for mismatches
    missing_files = json_ids - image_files
    extra_files = image_files - json_ids
    
    return {
        'json_image_count': len(json_ids),
        'folder_image_count': len(image_files),
        'is_matching': len(missing_files) == 0 and len(extra_files) == 0,
        'missing_files': list(missing_files),
        'extra_files': list(extra_files)
    }

def main(args=None):
    parser = argparse.ArgumentParser(description="Check uniqueness of image IDs in ground truth JSON")
    parser.add_argument('json_path', type=str, help='Path to ground truth JSON file')
    args = handle_args(args, parser)    
    json_path = Path(args.json_path)
    
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return
    
    # Load JSON once
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Check JSON uniqueness
    result_dict = check_json_uniqueness(data)
    
    print(f"\nChecking JSON ID uniqueness:")
    print(f"Total number of images in JSON: {result_dict['count']}")
    if result_dict['is_unique']:
        print("✅ All image IDs are unique!")
    else:
        print("❌ Found duplicate image IDs:")
        for id, count in result_dict['duplicates'].items():
            print(f"  Image ID '{id}' appears {count} times")
    
    # Check JSON-folder correspondence
    print(f"\nChecking JSON-folder correspondence:")
    corr_dict = check_json_image_correspondence(data, json_path)
    
    if corr_dict['is_matching']:
        print(f"✅ Perfect 1:1 correspondence between JSON and image folder!")
        print(f"   Found {corr_dict['json_image_count']} images")
    else:
        if corr_dict['missing_files']:
            print(f"❌   Files in JSON but missing from folder ({len(corr_dict['missing_files'])}):")
            for file in corr_dict['missing_files']:
                print(f"    - {file}")
        else:
            print(f"✅   No files in JSON missing from folder")
        if corr_dict['extra_files']:
            print(f"❌   Files in folder but missing from JSON ({len(corr_dict['extra_files'])}):")
            for file in corr_dict['extra_files']:
                print(f"    - {file}")
        else:
            print(f"✅   No files in folder missing from JSON")

if __name__ == "__main__":
    args = {"json_path": "C:/Users/tarchibald/github/docgen/projects/kwdc/kwdc-synthetic-data-forms/output_new/ground_truth.json"}
    main(args)
