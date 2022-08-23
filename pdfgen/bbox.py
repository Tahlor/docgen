from typing import Literal
from PIL import ImageDraw, Image
from cv2 import rectangle as np_rectangle
import numpy as np
from matplotlib.colors import to_rgb
import cv2

class BBox:
    def __init__(self,
                 origin: Literal['ul', 'll'],
                 bbox,
                 line_number=None,
                 line_word_index=None,
                 text=None,
                 parent_obj_bbox=None,
                 paragraph_index=None):
        """

        Args:
            origin: ul = upper left, ll = lower left; lower left not IMPLEMENTED
            bbox: ALWAYS x1,y1,x2,y2; assume origin is top left, x1,y1 is top left and smaller than x2,y2
            line_number (int): if generated in a box, the index of the word within the line
            line_word_index (int): if generated in a box, the line number
            text (str): the word (optional)
            parent_obj_bbox (x1,y1,x2,y2):
        """
        #self.origin = self.set_origin(origin)
        if origin=="ll":
            raise NotImplementedError

        self.bbox = list(bbox)
        self.line_number = line_number
        self.line_word_index = line_word_index
        self.text = text
        self.parent_bbox = parent_obj_bbox
        self.paragraph_index = paragraph_index

    def change_origin(self, origin: Literal['ul', 'll'], height):
        """ If the self.parent_bbox is defined, user can use it to compute height.
            Not automatic, since it's not clear what origin was used to define the parent_bbox.
            Obviously won't adjust the parent bbox, since it would need the height of it's parent.

        Args:
            origin:
            height:

        Returns:

        """
        if self.origin!=origin:
            self.origin=origin
            self.invert_y_axis(height)

    def invert_y_axis(self, height):
            self.bbox = self.bbox[0], height-self.bbox[3], self.bbox[2], height-self.bbox[1]

    def swap_bbox_axes(self, bbox):
        return bbox[1], bbox[0], bbox[3], bbox[2],

    def offset_origin(self, offset_x=0, offset_y=0):
        self.bbox = self._offset_origin(self.bbox, offset_x=offset_x, offset_y=offset_y)
        return self

    @staticmethod
    def _offset_origin(bbox, offset_x=0, offset_y=0):
        new_bbox = bbox.copy()
        new_bbox[0], new_bbox[2] = int(new_bbox[0] + offset_x), int(new_bbox[2] + offset_x)
        new_bbox[1], new_bbox[3] = int(new_bbox[1] + offset_y), int(new_bbox[3] + offset_y)
        return new_bbox

    def get_dim(self):
        return self._get_dim(self.bbox)

    @staticmethod
    def _get_dim(bbox):
        return bbox[2]-bbox[0], bbox[3]-bbox[1]

    @staticmethod
    def _get_height(bbox):
        return bbox[3]-bbox[1]

    @staticmethod
    def _get_width(bbox):
        return bbox[2]-bbox[0]

    @property
    def size(self):
        return self.get_dim()

    @property
    def height(self):
        return self._get_height(self.bbox)

    @property
    def width(self):
        return self._get_width(self.bbox)

    @staticmethod
    def draw_box_numpy(bbox, img, color="red", thickness=None, fill=None):
        color = (np.array(to_rgb(color)).astype(int) * 255).tolist()
        if img.ndim <3:
            img = img[:,:,None]

        np_rectangle(img, bbox[0:2], bbox[2:], color=color, thickness=thickness)
        return img

    @staticmethod
    def draw_box_pil(bbox, img, color, fill=None):
        #img1 = ImageDraw.Draw(img) if not isinstance(img, ImageDraw.ImageDraw) else img
        img1 = ImageDraw.Draw(img)
        img1.rectangle(bbox, outline=color, fill=fill)
        return img

    def draw_box(self, img, color="red"):
        BBox._draw_box(self.bbox, img, color)

    @staticmethod
    def _draw_box(bbox, img, color="red"):
        if isinstance(img, np.ndarray):
            return BBox.draw_box_numpy(bbox, img, color)
        elif isinstance(img, (Image.Image, ImageDraw.ImageDraw)):
            return BBox.draw_box_pil(bbox, img, color)
        else:
            raise NotImplementedError

    @staticmethod
    def bbox_norm(bbox, image_w, image_h):
        return bbox[0]/image_w, bbox[1]/image_h, bbox[2]/image_w, bbox[3]/image_h

    @staticmethod
    def bbox_norm_from_image(image, bbox):
        """ given a (background) image and a bbox, normalize it to (0,1)

        Args:
            image:
            bbox:

        Returns:

        """
        if isinstance(image, Image.Image):
            return BBox.bbox_norm(bbox, image.size[0], image.size[1])
        elif isinstance(image, np.ndarray):
            return BBox.bbox_norm(bbox, image.shape[1], image.shape[0])
        else:
            raise NotImplementedError

    @staticmethod
    def img_and_pos_to_bbox(img, pos):
        """

        Args:
            img: Numpy or PIL, the small/local image being pasted at pos in larger image
            pos: x,y - ALWAYS x,y

        Returns:

        """
        if isinstance(img, Image.Image):
            return [*pos, pos[0] + img.size[0], pos[1] + img.size[1]]
        elif isinstance(img, np.ndarray):
            return [*pos, pos[0]+img.shape[1], pos[1]+img.shape[0]]
        else:
            raise NotImplementedError

    @staticmethod
    def _draw_center(bbox, img, color):
        width, height = BBox._get_dim(bbox)
        x_center = (bbox[0]+bbox[2])/2
        y_center = (bbox[1] + bbox[3]) / 2
        bbox = (width * -.1 + x_center,
               height * -.1 + y_center,
                width * .1 + x_center,
                height * .1 + y_center,
            )

        if isinstance(img, np.ndarray):
            return BBox.draw_box_numpy(bbox, img, color)
        elif isinstance(img, (Image.Image, ImageDraw.ImageDraw)):
            return BBox.draw_box_pil(bbox, img, color, fill="red")
        else:
            raise NotImplementedError

    def __iter__(self, i):
        return self.bbox[i]

    def __repr__(self):
        return str(self.bbox)

    @staticmethod
    def get_maximal_box(list_of_bboxes):
        """
        list_of_bboxes: [x1,y1,x2,y2,xx1,yy1,xx2,yy2,...]
        Returns:
            The union of all bboxes

        """
        boxes = np.array(list_of_bboxes).reshape(-1,4)
        #boxes = np.array(range(0,20))
        return [int(np.min(boxes[:, 0])),
        int(np.min(boxes[:, 1])),
        int(np.max(boxes[:, 2])),
        int(np.max(boxes[:, 3]))]