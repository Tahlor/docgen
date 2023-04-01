from pathlib import Path
from PIL import Image
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ImageLoader:
    """Loads and manages form template images"""
    
    def __init__(self, image_dir: Path):
        """
        Initialize ImageLoader
        
        Args:
            image_dir (Path): Directory containing form template images
        """
        self.image_dir = Path(image_dir)
        self.images = self._load_images()
        
    def _load_images(self) -> Dict[str, Image.Image]:
        """Load all images from the directory"""
        images = {}
        for img_path in self.image_dir.glob("*.*"):
            try:
                images[img_path.name] = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
        return images
    
    def get_image(self, filename: str) -> Image.Image:
        """
        Get a specific image by filename
        
        Args:
            filename (str): Name of the image file
            
        Returns:
            Image.Image: The requested image
        """
        if filename not in self.images:
            raise KeyError(f"Image {filename} not found")
        return self.images[filename].copy() 