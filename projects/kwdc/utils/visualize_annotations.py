from PIL import Image, ImageDraw, ImageFont
import json
from pathlib import Path
import random

def visualize_annotations(image_path, json_data, output_path):
    """
    Draw bounding boxes and labels on an image based on JSON annotations.
    
    Args:
        image_path: Path to the input image
        json_data: Dictionary containing annotations for a single image
        output_path: Path to save the visualized image
    """
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Generate random colors for different fields
    colors = {}
    
    # Process each region
    for region in json_data['regions']:
        # Get bounding box coordinates
        bbox = region['bounding_box']
        field_name = region['labels'][0]['name']
        value = region['labels'][0]['value']
        
        # Generate a random color for this field type if we haven't seen it before
        if field_name not in colors:
            colors[field_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
        
        # Draw rectangle
        draw.rectangle(bbox, outline=colors[field_name], width=2)
        
        # Draw label
        label_text = f"{field_name}: {value}"
        draw.text((bbox[0], bbox[1] - 10), label_text, fill=colors[field_name])
        
    # Save the annotated image
    image.save(output_path)

def main():
    # Setup paths
    base_dir = Path("C:/Users/tarchibald/github/docgen/projects/kwdc/kwdc-synthetic-data-forms")
    output_dir = base_dir / "output"
    
    image_id = "000001"
    image_path = output_dir / f"{image_id}.jpg"
    json_path = output_dir / "ground_truth.json"
    
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Find the matching image data
    image_data = next(
        (img for img in json_data['images'] if img['id'] == image_id),
        None
    )
    
    if image_data is None:
        raise ValueError(f"No annotation data found for image {image_id}")
    
    # Create visualization
    output_path = output_dir / f"{image_id}_annotated.jpg"
    visualize_annotations(image_path, image_data, output_path)

if __name__ == "__main__":
    main()
