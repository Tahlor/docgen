from pathlib import Path
import pickle
from tqdm import tqdm
from itertools import chain

def get_bounding_box(points):
    """
    Returns the bounding box for a given sequence of points in the format (x1, y1, x2, y2).

    Parameters:
    points (list of tuples): A list of (x, y) tuples representing points.

    Returns:
    tuple: A tuple (x1, y1, x2, y2) representing the bounding box,
           where (x1, y1) is the top-left and (x2, y2) is the bottom-right.
    """
    if not points:
        raise ValueError("The list of points is empty")

    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]

    return min_x, min_y, max_x, max_y


pickle_file = "G:/s3/synthetic_data/resources/backgrounds/synthetic_backgrounds/dalle/document_backgrounds/with_backgrounds/bounding_boxes.pkl"
pickle_file = Path(pickle_file)
# load
with open(pickle_file, 'rb') as f:
    data = pickle.load(f)

output = {}
for k, v in tqdm(data.items()):
    file_name = Path(k).stem
    polygon = list(chain.from_iterable(x["polygon"] for x in v[1]))
    bbox = get_bounding_box(polygon)
    # calculate area
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    if area < 10000:
        bbox = None
    output[file_name] = bbox

# dump
with open(pickle_file.with_name("bounding_boxes_FILESTEM_BOUNDINGBOX.pkl"), 'wb') as f:
    pickle.dump(output, f)