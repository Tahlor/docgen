from docgen.render_doc import BoxFiller
from docgen.bbox import BBox
from hwgen.data.utils import show,display

seed = 1
import random
import numpy as np
import torch

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def filler():
    box_filler = BoxFiller()
    bbox = BBox(origin='ul',bbox=(0, 0, 600, 600))
    bbox.font_size = 40
    box_dict = box_filler.fill_box(bbox)
    img = box_dict['img']
    localization = box_dict["bbox_list"]
    styles = box_dict["styles"]

    display(img)

if __name__ == '__main__':
    filler()