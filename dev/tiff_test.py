import warnings
from pathlib import Path
import numpy as np
from PIL import Image
from tifffile import imread, imsave, TiffFile
import json

def read_img(path):
    path = Path(path)
    # if TIFF, open as TIFF
    if path.suffix == ".tiff":
        with TiffFile(path) as tif:
            label_chw = tif.asarray()
            metadata = parse_metadata(tif, keys_to_find="ActiveChannels")
        label_active_channels = metadata.get("ActiveChannels")
        label = np.transpose(label_chw, (1, 2, 0))
    else:
        label = Image.open(path)
        label_active_channels = metadata = None
    return {
        "label": label,
        "metadata": metadata,
        "label_active_channels": label_active_channels
    }


def parse_metadata(tif_binary, keys_to_find="ActiveChannels"):
    """
    Parse the image description metadata from a TIFF file.

    Args:
        tiff_path (str): The path to the TIFF file.

    Returns:
        dict: A dictionary containing the parsed metadata.
    """
    # Attempt to retrieve the image description metadata
    image_description = tif_binary.pages[0].tags.get('ImageDescription', None)
    if image_description:
        metadata_json = image_description.value
        # Deserialize the JSON string back into a Python dictionary
        metadata = json.loads(metadata_json)
        img_description = metadata.get("image_description")
        if img_description:
            img_description_dict = parse_image_description(img_description, keys_to_find)
            if isinstance(img_description_dict, dict):
                metadata.update(img_description_dict)
        return metadata

def parse_image_description(img_description, keys_to_find):
    try:
        metadata = json.loads(img_description)
        keys_to_find = [keys_to_find] if isinstance(keys_to_find, str) else keys_to_find

        for key in keys_to_find:
            if key in metadata:
                metadata[key] = metadata[key]
        return metadata
    except:
        warnings.warn(f"Could not parse metadata: {img_description}")
        return img_description


x = read_img(r"C:\Users\tarchibald\github\document_embeddings\document_embeddings\segmentation\dataset\v6_100k\0000022_label.tiff")
print(x)