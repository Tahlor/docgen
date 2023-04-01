from PIL import Image
from typing import Tuple, Dict, Any
import random
from utils.text_generator import TextGenerator
from textgen.fonts.font_sampler import FontSampler
from docgen.bbox import BBox

class ImageComposer:
    """Composes text onto form images"""
    
    def __init__(self, text_generator: TextGenerator):
        """
        Initialize ImageComposer
        
        Args:
            text_generator: TextGenerator instance to use
        """
        self.text_generator = text_generator
        
    def paste_text(
        self,
        base_image: Image.Image,
        text: str,
        bbox: BBox,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Generate and paste text onto base image
        
        Args:
            base_image: Base form image
            text: Text to add
            bbox: BBox object defining the area to paste text
        
        Returns:
            Tuple of (result image, metadata dict)
        """
        # Generate text image with transparency
        text_image, metadata = self.text_generator.generate_text_image(
            text=text,
            bbox=bbox,
        )
        
        # Validate text image dimensions
        if text_image is None or text_image.size[0] <= 0 or text_image.size[1] <= 0:
            return base_image.copy(), {'skipped': True}
        
        # Determine vertical position with randomness but ensure it doesn't exceed bbox height
        bbox_height = bbox.height
        text_height = metadata['text_height']
        max_vertical_offset = max(bbox_height - text_height, 0)
        y_offset = random.randint(0, max_vertical_offset)
        
        # Determine horizontal position with controlled randomness
        bbox_width = bbox.width
        text_width = metadata['text_width']
        available_width = bbox_width - text_width
        
        if available_width > 20:  # Arbitrary threshold for sufficient space
            # Introduce randomness up to 10% of the available space
            max_random_offset = int(available_width * 0.1)
            x_offset = random.randint(0, available_width) if available_width > 0 else 0
        else:
            # Center the text if not much space is available
            x_offset = available_width // 2 if available_width > 0 else 0
        
        paste_x = bbox.x1 + x_offset
        paste_y = bbox.y1 + y_offset
        
        # Update metadata with actual paste position
        actual_bbox = BBox("ul", [paste_x, paste_y, paste_x + text_width, paste_y + text_height], format="XYXY")
        metadata.update({
            'paste_x': paste_x,
            'paste_y': paste_y,
            'actual_bbox': actual_bbox
        })
        
        # Create a copy of the base image to avoid modifying the original
        result_image = base_image.copy()
        
        # Paste the text image using its alpha channel as mask
        result_image.paste(text_image, (paste_x, paste_y), text_image)
        
        # draw original bounding box, new bounding box, and show, import 
        # from PIL import ImageDraw
        # result_image = base_image.copy()
        # result_image = result_image.resize((int(result_image.width * 2.5), int(result_image.height * 2.5)), Image.NEAREST)

        # draw = ImageDraw.Draw(result_image)
        # draw.rectangle(bbox, outline="red")
        # draw.rectangle(actual_bbox, outline="green")
        # result_image.show()



        return result_image, metadata