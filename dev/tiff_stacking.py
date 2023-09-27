from pathlib import Path
import numpy as np
from PIL import Image
from tifffile import imread, imsave

ROOT=Path("G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds")
IMG1=ROOT / "with_backgrounds/aerial view of 2 blank pages of old open book, full frame/_0094e018-fd1b-4ee6-aa54-97cd4571528b.jfif"
IMG2=ROOT / "paper_only/paper with many ink marks, crinkles, wrinkles, and imperfections and variable lighting/_002ab850-e445-4e51-872d-4d5961020cb8.jfif"
OUTPUT = ROOT / "stacked_image.tiff"

def load_and_convert_image(image_path: Path) -> np.array:
    """Same as previous definition."""
    image = Image.open(image_path)
    return np.array(image)

def stack_images_to_channels(images: list) -> np.array:
    """Same as previous definition."""
    return np.concatenate(images, axis=-1)

def save_tiff_image(image: np.array, output_path: Path):
    """Same as previous definition."""
    imsave(output_path, image, compress=9)

def encode_channels_to_colors(image: np.array, colors: list) -> np.array:
    """
    Encode each channel to a different color and composite to a single image.
    Same as previous definition.
    """
    if colors is None:
        colors = [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1]
        ]

    output_img = np.zeros((image.shape[0], image.shape[1], 3))

    for i in range(image.shape[2]):
        channel = image[:,:,i]
        channel_normalized = channel / 255.0
        color = np.array(colors[i])
        colored_channel = channel_normalized[:, :, np.newaxis] * color
        output_img = np.maximum(output_img, colored_channel)

    output_img = (output_img * 255).astype('uint8')
    return output_img

def main():
    img1_path = Path(IMG1)
    img2_path = Path(IMG2)

    img1 = load_and_convert_image(img1_path)
    img2 = load_and_convert_image(img2_path)

    img6_channel = stack_images_to_channels([img1, img2])

    save_path = OUTPUT
    save_tiff_image(img6_channel, save_path)

    loaded_tiff = imread(save_path)

    Image.fromarray(loaded_tiff[:, :, :3]).show()
    Image.fromarray(loaded_tiff[:, :, 3:]).show()

    color_encoded_image = encode_channels_to_colors(loaded_tiff, colors=None)
    Image.fromarray(color_encoded_image).show()

if __name__ == "__main__":
    main()
