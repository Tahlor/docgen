from pathlib import Path
import json
from typing import Dict, List, Any
import logging
from docgen.bbox import BBox, BBoxNGon

logger = logging.getLogger(__name__)

class JSONParser:
    """Parses and manages form template JSON data"""
    
    def __init__(self, json_path: Path):
        """
        Initialize JSONParser
        
        Args:
            json_path (Path): Path to the JSON file containing form definitions
        """
        self.json_path = Path(json_path)
        self.data = self._load_json()
        
    def _load_json(self) -> Dict:
        """Load and parse the JSON file"""
        try:
            with open(self.json_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON {self.json_path}: {e}")
            raise
            
    def get_fields_for_image(self, image_id: str) -> List[Dict[str, Any]]:
        """
        Get all fields for a specific image
        
        Args:
            image_id (str): ID of the image
            
        Returns:
            List[Dict]: List of field definitions with BBox objects
        """
        for image in self.data['images']:
            if image['id'] == image_id or image['name'] == image_id:
                return [
                    {
                        **region,
                        'bounding_box': BBox("ul", region['bounding_box'], format="XYXY")
                    } for region in image['regions']
                    if any(label['name'] == 'field' for label in region['labels'])
                ]
        return []
    
    def get_field_value(self, field_id: str) -> str:
        """Get the value for a specific field"""
        for image in self.data['images']:
            for region in image['regions']:
                if region['id'] == field_id:
                    for label in region['labels']:
                        if label['name'] == 'field':
                            return label['value']
        return None 