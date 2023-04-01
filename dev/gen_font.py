import cv2
import numpy as np
import PIL

def render_cv2(text: str, font_size: int = 30, thickness: int = 1) -> tuple:
    """
    Generate an image of text using OpenCV and return it along with the text.

    Args:
        text (str): The text to render.
        font_size (int): Desired font size, used to compute font_scale.
        thickness (int): Thickness of the lines used to draw the text.

    Returns:
        tuple: A tuple containing the image and the text.
    """
    # Define font and text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 0)  # Black color

    # Calculate font scale based on desired font size
    # Note: The relation between font_size and font_scale may need empirical calibration
    font_scale = font_size / 30.0  # Hypothetical mapping, adjust as needed

    # Get text size and baseline
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Initialize a white image with dimensions based on text
    image = np.ones((text_height + baseline, text_width, 3), dtype=np.uint8) * 255

    # Draw the text on the image
    cv2.putText(image, text, (0, text_height), font, font_scale, color, thickness)

    return image, text

# Example usage
if __name__ == "__main__":
    text_to_render = "Heyg"
    image, rendered_text = render_cv2(text_to_render)

    # Convert to np.uint8 for display using PIL
    image = image.astype(np.uint8)
    PIL.Image.fromarray(image).show()
