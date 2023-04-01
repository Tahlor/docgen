from pathlib import Path
import json
from PIL import Image
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class OutputSaver:
    """Saves generated images and metadata"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize OutputSaver
        
        Args:
            output_dir (Path): Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_image(self, image: Image.Image, filename: str):
        """
        Save an image
        
        Args:
            image (Image.Image): Image to save
            filename (str): Output filename
        """
        try:
            output_path = self.output_dir / filename
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(output_path)
        except Exception as e:
            logger.error(f"Failed to save image {filename}: {e}")
            
    def save_json(self, data: Dict[str, Any], filename: str):
        """
        Save JSON data in the desired format
        
        Args:
            data (Dict): Data to save
            filename (str): Output filename
        """
        try:
            output_path = self.output_dir / filename
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save JSON {filename}: {e}")