from pathlib import Path
import logging
import random
import argparse
from typing import Dict, Any
from multiprocessing import Pool, cpu_count
import pickle
import json
from tqdm import tqdm
import numpy as np

from utils.dictionary_loader import DictionaryLoader
from utils.image_loader import ImageLoader
from utils.json_parser import JSONParser
from utils.field_filler import FieldFiller
from utils.text_generator import TextGenerator
from utils.image_composer import ImageComposer
from utils.output_saver import OutputSaver
from textgen.fonts.font_sampler import FontSampler
from docgen.bbox import BBox
from utils.visualize_annotations import visualize_annotations
from utils.combine_gt_pkl import combine_ground_truth_pickles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

issued_warnings = set()

def generate_ground_truth(
    field: Dict[str, Any],
    text: str,
    encoding: str,
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate ground truth entry in the same format as input JSON"""
    bbox = field['bounding_box']
    bbox_obj = BBox("ul", bbox, format="XYXY")
    if 'actual_bbox' in metadata:
        actual_bbox = metadata['actual_bbox']
    else:
        actual_bbox = field['bounding_box']

    return {
        'id': field['id'],
        'bounding_box': actual_bbox.get_XYXY(),  # Ensure bounding_box is in XYXY
        'polygon': bbox_obj.box_4pt(bbox_obj.get_XYXY()),  # Generate polygon from BBox
        'labels': [
            {
                "name": field['labels'][0]['value'],
                "value": text,
                "encoding": encoding
            }
        ]
    }

def process_image(args):
    """
    Process a single image for form generation.
    """
    try:
        (base_image_id, fields, dict_dir, font_config, output_prefix) = args

        new_image_id = f"{int(output_prefix):06d}"
        logger.debug(f"Processing image {base_image_id} -> {new_image_id}")

        # Get cached RGBA image
        base_image = global_image_cache[base_image_id]
        result_image = base_image.copy()

        # Store ground truth for this image
        image_ground_truth = {'id': new_image_id, 'regions': []}

        # Process each field
        for field in fields:
            field_name = next((label['value'] for label in field['labels'] 
                               if label['name'] == 'field'), None)

            if not field_name.endswith('Text'):
                continue

            # Determine if field should be filled
            should_fill, placeholder = global_field_filler.should_fill(field_name)
            
            if should_fill:
                # Get value from dictionary
                value_encoding = global_field_filler.get_value(field_name)
                if value_encoding is None:
                    continue
                value, encoding = value_encoding
                           
                # Create BBox object
                bbox_xyxy = field['bounding_box']
                bbox_obj = BBox("ul", bbox_xyxy, format="XYXY")

                # Generate and paste text
                result_image, metadata = global_image_composer.paste_text(
                    result_image,
                    value,
                    bbox_obj,
                )
                
                # Only add to ground truth if text was successfully pasted
                if not metadata.get('skipped', False):
                    gt_entry = generate_ground_truth(field, value, encoding, metadata)
                    image_ground_truth['regions'].append(gt_entry)
                
            elif placeholder:
                # Add placeholder if specified
                bbox_xyxy = field['bounding_box']
                bbox_obj = BBox("ul", bbox_xyxy, format="XYXY")

                result_image, metadata = global_image_composer.paste_text(
                    result_image,
                    placeholder,
                    bbox_obj,
                )
                
                # Only add to ground truth if text was successfully pasted
                if not metadata.get('skipped', False):
                    gt_entry = generate_ground_truth(field, placeholder, placeholder, metadata)
                    image_ground_truth['regions'].append(gt_entry)

        return (new_image_id, result_image, image_ground_truth)
    except Exception as e:
        logger.error(f"Failed to process image {args[0]}: {str(e)}")
        return None

def save_batch_ground_truth(ground_truth, batch_num, output_dir):
    """Save a batch of ground truth data as a pickle file"""
    pickle_path = output_dir / f"ground_truth_batch_{batch_num}.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(ground_truth, f)

def init_worker(dictionary_paths, font_config, json_path, output_dir, image_dir):
    global global_dictionary_loader
    global global_field_filler
    global global_font_sampler
    global global_text_generator
    global global_image_composer
    global global_json_parser
    global global_output_saver
    global global_image_cache

    global_dictionary_loader = DictionaryLoader(dictionary_paths)
    global_field_filler = FieldFiller(global_dictionary_loader)
    global_font_sampler = FontSampler(**font_config)
    global_text_generator = TextGenerator(
        font_sampler=global_font_sampler,
        target_height_ratio=(0.5, 0.75)
    )
    global_image_composer = ImageComposer(global_text_generator)
    global_json_parser = JSONParser(json_path)
    global_output_saver = OutputSaver(output_dir)

    # Create image cache
    global_image_cache = {}
    image_loader = ImageLoader(image_dir)  # You'll need to pass image_dir to init_worker
    for img_id, img in image_loader.images.items():
        global_image_cache[img_id] = img.convert("RGBA")

def get_last_image_id(output_dir):
    """Find the highest image ID from existing batch files and final JSON"""
    last_id = None
    
    # Check pickle files
    for pickle_file in output_dir.glob("ground_truth_batch_*.pkl"):
        with open(pickle_file, "rb") as f:
            batch_data = pickle.load(f)
            if batch_data['images']:
                # Get the highest ID in this batch
                batch_max = max(int(img['id']) for img in batch_data['images'])
                last_id = batch_max if last_id is None else max(last_id, batch_max)
    
    # Check final JSON if it exists
    json_path = output_dir / "ground_truth.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            if json_data.get('images'):
                json_max = max(int(img['id']) for img in json_data['images'])
                last_id = json_max if last_id is None else max(last_id, json_max)
                
                # Save JSON data as batch_0.pkl if it doesn't exist
                batch_0_path = output_dir / "ground_truth_batch_0.pkl"
                if not batch_0_path.exists():
                    with open(batch_0_path, "wb") as f:
                        pickle.dump(json_data, f)
    
    return str(last_id).zfill(6) if last_id is not None else None

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic form images.")
    parser.add_argument(
        '--num_images',
        type=int,
        default=10,
        help='Number of images to generate.'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='000001',
        help='Prefix for output image filenames.'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=None,
        help='Number of processes to use. Defaults to CPU count - 2. Set to 1 to disable multiprocessing.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Number of images per batch for saving ground truth.'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from the last generated image ID.'
    )
    parser.add_argument(
        '--base_images',
        type=str,
        nargs='+',
        help='List of specific base image IDs to use (e.g., --base_images 000001 000002)'
    )
    parser.add_argument(
        '--exclude_images',
        type=str,
        nargs='+',
        help='List of base image IDs to exclude (e.g., --exclude_images 000001 000002)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="C:/Users/tarchibald/github/docgen/projects/kwdc/kwdc-synthetic-data-forms/output",
        help='Path to output directory for generated images and ground truth'
    )
    args = parser.parse_args()
    
    num_images = args.num_images
    output_prefix = args.output_prefix
    
    # Setup paths
    base_dir = Path("C:/Users/tarchibald/github/docgen/projects/kwdc/kwdc-synthetic-data-forms")
    image_dir = base_dir / "images"
    dict_dir = base_dir / "dictionaries"
    json_path = base_dir / "kwdc-templates-synthetic.json"
    output_dir = Path(args.output_dir)  # Use the provided output path
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare minimal data for tasks
    font_config = {
        'font_folder': Path("F:/s3/synthetic_data/resources/fonts/fonts"),
        'font_class_files': [
            "F:/s3/synthetic_data/resources/fonts/FONT_SAMPLES/handwriting_fonts.pkl",
            "F:/s3/synthetic_data/resources/fonts/FONT_SAMPLES/typewriter_fonts.pkl",
            "F:/s3/synthetic_data/resources/fonts/FONT_SAMPLES/typewriter_ish_fonts.pkl",
            "F:/s3/synthetic_data/resources/fonts/FONT_SAMPLES/APPROVED_fonts.pkl",
        ],
        'weights': [10, 20, 10, 4],
        'csv_file': Path("F:/s3/synthetic_data/resources/fonts/ALL_FONTS.csv")
    }


    # Initialize components
    dictionary_loader = DictionaryLoader(dict_dir)
    image_loader = ImageLoader(image_dir)
    json_parser = JSONParser(json_path)
    field_filler = FieldFiller(dictionary_loader)
    font_sampler = FontSampler(**font_config)
    text_generator = TextGenerator(
        font_sampler=font_sampler,
        target_height_ratio=(0.5, 0.9)
    )
    image_composer = ImageComposer(text_generator)
    output_saver = OutputSaver(output_dir)
    
    # Handle resume functionality
    if args.resume:
        last_id = get_last_image_id(output_dir)
        if last_id:
            output_prefix = str(int(last_id) + 1).zfill(6)
            logger.info(f"Resuming from image ID: {output_prefix}")
        else:
            logger.info("No existing images found. Starting from the beginning.")
    
    # Prepare tasks by looping through reference images
    tasks = []
    reference_image_ids = list(image_loader.images.keys())
    if not reference_image_ids:
        logger.error("No reference images found in the image directory.")
        return
    
    # Filter reference images based on include/exclude lists
    if args.base_images:
        valid_base_images = set(args.base_images)
        reference_image_ids = [img_id for img_id in reference_image_ids if img_id in valid_base_images]
        if not reference_image_ids:
            logger.error("None of the specified base images were found in the image directory.")
            return
        logger.info(f"Using {len(reference_image_ids)} specified base images: {', '.join(reference_image_ids)}")
    elif args.exclude_images:
        exclude_images = set(args.exclude_images)
        reference_image_ids = [img_id for img_id in reference_image_ids if img_id not in exclude_images]
        if not reference_image_ids:
            logger.error("All available images were excluded.")
            return
        logger.info(f"Excluded {len(args.exclude_images)} images. Using {len(reference_image_ids)} remaining images.")
    
    current_id = int(output_prefix)
    for i in range(current_id, num_images + 1):
        base_image_id = random.choice(reference_image_ids)
        
        base_image = image_loader.images[base_image_id]
        fields = json_parser.get_fields_for_image(base_image_id)
        
        # Pack only necessary data
        task_data = (
            base_image_id,
            fields,
            dict_dir,
            font_config,
            current_id
        )
        tasks.append(task_data)
        current_id += 1 
    
    ground_truth = {'images': []}
    batch_num = 0
    
    # Choose between multiprocessing and single-process execution
    if args.processes != 1:
        num_processes = args.processes if args.processes else cpu_count() - 2
        with Pool(
            processes=num_processes, 
            initializer=init_worker, 
            initargs=(dict_dir, font_config, json_path, output_dir, image_dir)
        ) as pool:
            # Wrap the imap_unordered with tqdm for progress bar
            for idx, result in enumerate(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Processing Images")):
                if result is None:
                    continue
                new_image_id, filled_image, image_ground_truth = result
                output_saver.save_image(filled_image, f"{new_image_id}.jpg")
                ground_truth['images'].append(image_ground_truth)
                
                # Save batch and clear memory when batch_size is reached
                if (idx + 1) % args.batch_size == 0:
                    save_batch_ground_truth(ground_truth, batch_num, output_dir)
                    ground_truth = {'images': []}
                    batch_num += 1
    else:
        init_worker(dict_dir, font_config, json_path, output_dir, image_dir)
        for idx, task in enumerate(tqdm(tasks, desc="Processing Images")):
            result = process_image(task)
            if result is None:
                continue
            new_image_id, filled_image, image_ground_truth = result
            
            output_saver.save_image(filled_image, f"{new_image_id}.jpg")
            ground_truth['images'].append(image_ground_truth)
            
            # Save batch and clear memory when batch_size is reached
            if (idx + 1) % args.batch_size == 0:
                save_batch_ground_truth(ground_truth, batch_num, output_dir)
                ground_truth = {'images': []}
                batch_num += 1
    
    # Save final batch if there are remaining images
    if ground_truth['images']:
        save_batch_ground_truth(ground_truth, batch_num, output_dir)
    
    # Combine all pickle files into final JSON (now using the imported function)
    combine_ground_truth_pickles(output_dir)
    
    # Visualize the first generated image
    first_image_id = args.output_prefix
    image_path = output_dir / f"{first_image_id}.jpg"
    json_path = output_dir / "ground_truth.json"
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Find the matching image data
    image_data = next(
        (img for img in json_data['images'] if img['id'] == first_image_id),
        None
    )
    
    if image_data is None:
        logger.warning(f"No annotation data found for image {first_image_id}")
        return
        
    output_path = output_dir / f"{first_image_id}_annotated.jpg"
    visualize_annotations(image_path, image_data, output_path)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()