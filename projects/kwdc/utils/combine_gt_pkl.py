from pathlib import Path
import pickle
import json
import numpy as np
try:
    from .output_saver import OutputSaver  # For imports within the package
except ImportError:
    from output_saver import OutputSaver   # For imports from parent directory

def convert_numpy_int(obj):
    """Convert numpy int64 to native Python int"""
    if isinstance(obj, dict):
        return {k: convert_numpy_int(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_int(elem) for elem in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

def combine_ground_truth_pickles(output_dir: Path, cleanup: bool = True) -> None:
    """
    Combine all ground truth pickle files in a directory into a single JSON file.
    
    Args:
        output_dir: Directory containing the pickle files and where JSON will be saved
        cleanup: If True, delete pickle files after combining
    """
    combined_ground_truth = {'images': []}
    
    # Load and combine all pickle files
    for pickle_file in sorted(output_dir.glob("ground_truth_batch_*.pkl")):
        with open(pickle_file, "rb") as f:
            batch_data = pickle.load(f)
            combined_ground_truth['images'].extend(batch_data['images'])
        
        # Optionally delete pickle file after combining
        if cleanup:
            pickle_file.unlink()
    
    # Convert any numpy int64 to native Python int
    combined_ground_truth = convert_numpy_int(combined_ground_truth)

    # Save combined ground truth as JSON
    output_saver = OutputSaver(output_dir)
    output_saver.save_json(combined_ground_truth, "ground_truth.json")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combine ground truth pickle files into a single JSON.")
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory containing the pickle files'
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help='Do not delete pickle files after combining'
    )
    
    args = parser.parse_args()
    combine_ground_truth_pickles(Path(args.output_dir), cleanup=not args.no_cleanup)
