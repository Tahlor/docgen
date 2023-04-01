import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
from typing import Tuple, Optional, List, Dict, Any
import logging
from textgen.fonts.font_sampler import FontSampler
from docgen.bbox import BBox
from textgen.rendertext.render_word import RenderWordFont

logger = logging.getLogger(__name__)

class TextGenerator:
    """Generates text images with proper font sizing and positioning"""
    
    def __init__(
        self,
        target_height_ratio: Tuple[float, float] = (0.5, 0.75),
        font_sampler: FontSampler = None,

    ):
        """
        Initialize TextGenerator
        
        Args:
            font_dir: Base directory containing fonts and metadata
            target_height_ratio: Tuple of (min, max) ratios for text height relative to bbox
        """
        self.target_height_ratio = target_height_ratio
        
        
        self.render_word = RenderWordFont(
            format="PIL",
            font_sampler=font_sampler
        )
        
    def generate_text_image(
        self,
        text: str,
        bbox: BBox,  # Changed to accept BBox object
        color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Tuple[Image.Image, Dict]:
        """
        Generate text image and placement metadata using RenderWordFont
        
        Args:
            text: Text to render
            bbox: BBox object defining the area to paste text
            color: Text color (ignored, using grayscale instead)
            
        Returns:
            Tuple of (text image, metadata dict)
        """
        text = str(text)
        bbox_height = bbox.height
        bbox_width = bbox.width
        
        # Generate the word using RenderWordFont
        render_result = self.render_word.render_word(
            word=text,
            font=None,
            size=int(bbox_height * random.uniform(*self.target_height_ratio)),
            color=color,
            retry=True,
            transparency_layer=True
        )
        
        if render_result is None:
            logger.warning(f"Failed to render text: {text}")
            return Image.new('RGBA', (bbox_width, bbox_height), (0, 0, 0, 0)), {}
        
        word_image = render_result["image"]
        font_size = render_result.get("size", 12)
        
        # Convert text image to grayscale skewed toward black
        word_image = self._convert_to_grayscale(word_image)
        
        # Calculate text size
        text_width, text_height = word_image.size
        
        # Ensure text fits within the bbox
        if text_width > bbox_width or text_height > bbox_height:
            scaling_factor = min(bbox_width / text_width, bbox_height / text_height)
            new_size = (int(text_width * scaling_factor), int(text_height * scaling_factor))
            word_image = word_image.resize(new_size, Image.Resampling.LANCZOS)
            text_width, text_height = word_image.size
        
        # Metadata
        metadata = {
            'text_width': text_width,
            'text_height': text_height,
            'font_size': font_size,
        }
        
        return word_image, metadata
    
    def _convert_to_grayscale(self, image: Image.Image) -> Image.Image:
        """
        Convert the image to grayscale with black/dark gray text and transparent background
        
        Args:
            image: PIL Image to convert
            
        Returns:
            PIL Image in RGBA mode with black text and transparent background
        """
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Get the alpha channel which defines the text shape
        alpha = image.split()[3]
        
        # Create new RGBA image
        rgba_image = Image.new("RGBA", image.size)
        
        # Create dark text color (random dark gray)
        text_color = random.randint(0, 60)  # Darker range for text
        
        # Fill the image with the text color where alpha channel is non-zero
        pixels = []
        alpha_data = alpha.getdata()
        for alpha_value in alpha_data:
            if alpha_value > 0:  # Where text exists
                pixels.append((text_color, text_color, text_color, alpha_value))
            else:  # Transparent background
                pixels.append((0, 0, 0, 0))
        
        rgba_image.putdata(pixels)
        return rgba_image