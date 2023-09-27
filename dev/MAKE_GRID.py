from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import itertools


def plot_images_in_grid(image_folder: Path, grid_shape: tuple = (3, 4)):
    """Plot images in a 3x4 grid until no more pictures.

    Args:
        image_folder (Path): The folder containing the images.
        grid_shape (tuple): Shape of the grid (rows, columns).

    """
    image_paths = sorted(image_folder.glob("*.jpg"))  # Assuming images are in jpg format
    n_rows, n_cols = grid_shape
    total_images = n_rows * n_cols

    for i in range(0, len(image_paths), total_images):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

        for ax, image_path in itertools.zip_longest(axes.flatten(), image_paths[i:i + total_images]):
            ax.axis("off")  # Turn off axis
            if image_path:
                image = Image.open(image_path)
                ax.imshow(image)
                ax.set_title(image_path.stem)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    image_folder = Path(r"G:\s3\sample_forms\augraphy_output\2wwii_1148632-3302")
    plot_images_in_grid(image_folder)
