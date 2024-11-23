import random
import sys
import pickle
import warnings
from typing import Literal
from PIL import ImageDraw, Image
from cv2 import rectangle as np_rectangle
import numpy as np
from matplotlib.colors import to_rgb
import cv2
import json
from io import BytesIO

try:
    from scipy.spatial import ConvexHull
except:
    warnings.warn("Install scipy to enable calculation of ConvexHull for maximal BBox")

ravel = lambda sublist: sublist.ravel() if isinstance(sublist, np.ndarray) else sublist

class BBox(list):
    __slots__ = [
        'origin', '_bbox', 'format', 'force_int', 'img', 'height_ll',
        'line_number', 'line_word_index', 'text', 'parent_bbox',
        'paragraph_index', 'font_size', 'category'
    ]
    
    def __init__(self,
                 origin: Literal['ul', 'll'],
                 bbox,
                 line_index=None,
                 line_word_index=None,
                 text=None,
                 parent_obj_bbox=None,
                 paragraph_index=None,
                 force_int=True,
                 img=None,
                 font_size=None,
                 format: Literal['XYXY', 'XYWH'] = "XYXY",
                 height_ll=None,
                 category=None,
                 ):
        """
        Store BBox as UL XYXY TUPLE to prevent accidental modification
        The only functions that should modify in place are: update_bbox, enlarge, expand, offset_origin

        Args:
            origin: ul = upper left, ll = lower left; lower left not IMPLEMENTED
            bbox: ALWAYS x1,y1,x2,y2; assume origin is top left, x1,y1 is top left and smaller than x2,y2
            line_index (int): if generated in a box, the index of the word within the line
            line_word_index (int): if generated in a box, the line number
            text (str): the word (optional)
            parent_obj_bbox (x1,y1,x2,y2):
            paragraph_index (int): if multiple paragraphs, the index of the paragraph
            force_int (bool): if True, will force all values to be integers
            format:Literal['XYXY', 'XYWH']="XYXY"): by default, bbox is in X1,Y1,X2,Y2 format;
                            ALL INTERNAL OPERATIONS WORK ON XYXY
                            You can input/output to XYWH (COCO)
            height (int): if origin is ll, you must supply the height of the image; ONLY USED FOR LL calculation
        """
        super().__init__(bbox)

        self.origin = origin
        self.format = format
        self.force_int = self._force_int if force_int else lambda *x: x
        self.img = img
        self.height_ll = height_ll

        # BBox is always stored as XYXY, update if format is XYWH
        self.update_bbox(bbox, self.format)

        self.line_number = line_index
        self.line_word_index = line_word_index
        self.text = text
        self.parent_bbox = parent_obj_bbox
        self.paragraph_index = paragraph_index
        self.font_size = font_size
        self.category = category

    def __getstate__(self):
        state = {slot: getattr(self, slot) for slot in self.__slots__}
        if getattr(self, 'img', None) is not None:
            warnings.warn("Attribute 'img' is present but will not be pickled for security reasons.")
            state['img'] = None  # Remove 'img' from the state to ignore it
        return state

    def __setstate__(self, state):
        if state.get('img') is not None:
            warnings.warn("Attribute 'img' was found during unpickling but is being ignored.")
            state['img'] = None  # Ignore 'img' by setting it to None
        for slot, value in state.items():
            setattr(self, slot, value)

    def update_bbox(self, bbox, format:Literal['XYXY', 'XYWH']="XYXY", origin="ul"):
        if isinstance(bbox, tuple):
            self._bbox = bbox # list(bbox)
        elif isinstance(bbox, list):
            self._bbox = tuple(bbox)
        elif isinstance(bbox, np.ndarray):
            self._bbox = bbox.tolist()
        elif isinstance(bbox, BBox):
            self._bbox = bbox.get_bbox()
        else:
            raise NotImplementedError("Unexpected type for BBox")
        if origin=="ll" and format=="XYXY":
            self._bbox = self._change_origin(self.height_ll)
        elif format == "XYWH":
            if origin=="ll":
                self._bbox = self._XYWH_ll_to_XYXY_ul(self._bbox, self.height_ll)
            else:
                self._bbox = tuple(self._XYWH_to_XYXY(self._bbox))

        self._bbox = self.force_int(self._bbox)
        self[:] = self._bbox

    def expand_rightward(self, pixels):
        self._bbox = self.force_int((self._bbox[0], self._bbox[1], self._bbox[2]+pixels, self._bbox[3]))
        return self

    def expand_downward(self, pixels):
        self._bbox = self.force_int((self._bbox[0], self._bbox[1], self._bbox[2], self._bbox[3]+pixels))
        return self

    def expand_horizontally(self, pixels):
        self._bbox = self.force_int((self._bbox[0]-pixels, self._bbox[1], self._bbox[2]+pixels, self._bbox[3]))
        return self

    def expand_vertically(self, pixels):
        self._bbox = self.force_int((self._bbox[0], self._bbox[1]-pixels, self._bbox[2], self._bbox[3]+pixels))
        return self

    def expand_random_amount(self, max_padding_horizontal, max_padding_vertical):
        horizontal_expansion = random.randint(0, max_padding_horizontal)
        vertical_expansion = random.randint(0, max_padding_vertical)
        self._bbox = self.force_int((self._bbox[0]-horizontal_expansion,
                                     self._bbox[1]-vertical_expansion,
                                     self._bbox[2]+horizontal_expansion,
                                     self._bbox[3]+vertical_expansion))
        return self

    def rescale(self, scale):
        self._rescale(self._bbox, scale)
        return self.bbox
    @staticmethod
    def _rescale(bbox, scale):
        if isinstance(scale, (tuple, list)):
            return bbox[0]*scale[0], bbox[1]*scale[1], bbox[2]*scale[0], bbox[3]*scale[1]
        else:
            return bbox[0]*scale, bbox[1]*scale, bbox[2]*scale, bbox[3]*scale

    @staticmethod
    def _XYWH_ll_to_XYXY_ul(bbox, height):
        """ This assumes are BBox's are defined with X1,Y1 being the lower left (LL) corner relative to the LL origin

        Args:
            bbox:
            height:

        Returns:

        """
        return bbox[0], height-bbox[1]-bbox[3], bbox[0]+bbox[2], height-bbox[1]

    @staticmethod
    def _XYXY_ul_to_XYWH_ll(bbox, height):
        return bbox[0], height-bbox[3], bbox[2]-bbox[0], bbox[3]-bbox[1]

    @staticmethod
    def _XYWH_to_XYXY(bbox):
        return bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]

    @staticmethod
    def _get_XYWH(box):
        return box[0], box[1], BBox._get_width(box), BBox._get_height(box)

    def get_XYWH(self):
        return self._get_XYWH(self._bbox)

    @staticmethod
    def _get_XYXY(box):
        return box[0], box[1], box[0] + box[2], box[1] + box[3]

    def get_XYXY(self):
        """ Returns the bbox in XYXY format - ALL internal operations are done in XYXY format"""
        return self._bbox

    def get_bbox(self):
        if self.format == "XYWH":
            if self.origin == "ll":
                return self._XYXY_ul_to_XYWH_ll(self.get_XYWH(), self.height)
            else:
                return self.get_XYWH()
        else:
            if self.origin == "ll":
                return self._change_origin(self._bbox, self.height_ll)
            else:
                return self._bbox

    def intersects(self, other_box):
        return self._intersects(self, other_box)

    @staticmethod
    def _intersects(a,b):
        """Check if two bounding boxes intersect.

        Args:
            other (BBox): another bounding box to check for intersection

        Returns:
            bool: True if the bounding boxes intersect, False otherwise
        """
        a = a._bbox
        b = b._bbox

        # a is to the right of b
        if a[0] > b[2]:
            return False

        # a is to the left of b
        if a[2] < b[0]:
            return False

        # a is above b
        if a[1] > b[3]:
            return False

        # a is below b
        if a[3] < b[1]:
            return False

        # bounding boxes intersect
        return True

    def random_subbox(self, min_size_x=None, min_size_y=None, max_size_x=None, max_size_y=None):
        min_size_x, min_size_y, max_size_x, max_size_y = \
            self.validate_sizes(min_size_x, min_size_y, max_size_x, max_size_y)

        # Get the random sizes of the subbox within min and max size limit
        random_subbox_width = random.randint(min_size_x, max_size_x)
        random_subbox_height = random.randint(min_size_y, max_size_y)

        # Get the random start point (upper left corner of the subbox)
        # such that the whole subbox fits into the original bbox
        x1 = random.randint(self._bbox[0], self._bbox[2] - random_subbox_width)
        y1 = random.randint(self._bbox[1], self._bbox[3] - random_subbox_height)

        # Calculate the lower right corner of the subbox
        x2 = x1 + random_subbox_width
        y2 = y1 + random_subbox_height

        # Create and return the new subbox as a BBox object
        return BBox(self.origin, (x1, y1, x2, y2), format=self.format)

    def validate_sizes(self, min_size_x, min_size_y, max_size_x, max_size_y):
        width, height = self._get_width(self._bbox), self._get_height(self._bbox)
        if min_size_x is None:
            min_size_x = 0
        if min_size_y is None:
            min_size_y = 0
        if max_size_x is None:
            max_size_x = width
        if max_size_y is None:
            max_size_y = height
        if max_size_x > width:
            warnings.warn("The max_size_x is bigger than the width of the original bounding box.")
            max_size_x = width

        if max_size_y > height:
            warnings.warn("The max_size_y is bigger than the height of the original bounding box.")
            max_size_y = height

        if min_size_x > max_size_x:
            warnings.warn(
                "The min_size_x is greater than the max_size_x. Adjusting the min_size_x to match the max_size_x.")
            min_size_x = max_size_x

        if min_size_y > max_size_y:
            warnings.warn(
                "The min_size_y is greater than the max_size_y. Adjusting the min_size_y to match the max_size_y.")
            min_size_y = max_size_y
        return min_size_x, min_size_y, max_size_x, max_size_y

    def get_bbox_as_list(self):
        """
        Returns the bounding box coordinates as a list of integers.
        """
        return list(self._bbox)

    def __json__(self):
        """
        Method to customize JSON serialization.
        """
        return self.get_bbox_as_list()

    def toJSON(self):
        """
        Serialize the object to JSON format.
        """
        return json.dumps(self.__json__())

    @property
    def __dict__(self):
        """ Customize the dictionary representation of the object for serialization. """
        # This dictionary format is directly serializable by json.dumps.
        return {
            'bbox': self.get_bbox_as_list(),
            # 'origin': self.origin,
            # 'format': self.format,
            # 'force_int': self.force_int,
            # 'line_index': self.line_index,
            # 'line_word_index': self.line_word_index,
            # 'text': self.text,
            # 'paragraph_index': self.paragraph_index,
            # 'category': self.category
        }

    @property
    def bbox(self):
        return self.get_bbox()

    @property
    def bbox_xyxy(self):
        return self._bbox

    @staticmethod
    def _force_int(bbox):
        if isinstance(bbox,BBox):
            bbox._bbox = BBox._force_int(bbox._bbox)
            return bbox._bbox
        else:
            return tuple(int(x) for x in bbox)

    @staticmethod
    def _enlarge(bbox, pixels, format:Literal['XYXY', 'XYWH']="XYXY"):
        if format == "XYWH":
            bbox = BBox._XYWH_to_XYXY(bbox)
        bbox = bbox[0]-pixels, bbox[1]-pixels, bbox[2]+pixels, bbox[3]+pixels
        if format == "XYWH":
            bbox = BBox._get_XYWH(bbox)
        return bbox
    def enlarge_box(self, pixels):
        """ Enlarge the box by the number of pixels in each direction
            This is useful for drawing boxes around text
        Args:
            pixels:

        Returns:

        """
        self._bbox = self.force_int(self._enlarge(self._bbox, pixels, format="XYXY"))
        return self.bbox

    @staticmethod
    def _change_origin(bbox, height):
        """ Change from UL (upper-left) to LL (lower-left) or LL to UL coordinate origin
            Must know the height of the coordinate-plane naturally

            If the self.parent_bbox is defined, user can use it to compute height.
            Not automatic, since it's not clear what origin was used to define the parent_bbox.
        """

        bbox = BBox._invert_y_axis(bbox, height)

        # Swap y-coords; only works on boxes
        return bbox[0], bbox[3], bbox[2], bbox[1]


    def invert_y_axis(self, height=None):
        height = self.height_ll if height is None else height
        return self.force_int(self._invert_y_axis(self._bbox, height=height))

    @staticmethod
    def _invert_y_axis(_bbox, height):
        """ This only inverts the y scale
            Now the larger y-coord will be the first point, use change origin to swap y-coords
        Args:
            height:

        Returns:

        """
        return _bbox[0], height-_bbox[1], _bbox[2], height-_bbox[3]

    @staticmethod
    def _swap_bbox_axes(bbox):
        return bbox[1], bbox[0], bbox[3], bbox[2],

    def swap_bbox_axes(self):
        return self._swap_bbox_axes(self._bbox)

    def offset_origin(self, offset_x=0, offset_y=0):
        self._bbox = self.force_int(
            self._offset_origin(self._bbox, offset_x=offset_x, offset_y=offset_y)
        )
        return self

    @staticmethod
    def _offset_origin(bbox, offset_x=0, offset_y=0):
        new_bbox = list(bbox).copy()
        new_bbox[0], new_bbox[2] = new_bbox[0] + offset_x, new_bbox[2] + offset_x
        new_bbox[1], new_bbox[3] = new_bbox[1] + offset_y, new_bbox[3] + offset_y
        return new_bbox

    def get_dim(self):
        return self._get_dim(self._bbox)

    def __getitem__(self, idx):
        return self._bbox[idx]

    def __len__(self):
        return len(self._bbox)

    def __copy__(self):
        return BBox(
            self.origin,
            list(self),  # use list(self) to get the list contents
            line_index=self.line_number,
            line_word_index=self.line_word_index,
            text=self.text,
            parent_obj_bbox=self.parent_bbox,
            paragraph_index=self.paragraph_index,
            force_int=self.force_int is not None,
            img=self.img,
            font_size=self.font_size,
            format=self.format,
            height_ll=self.height_ll,
            category=self.category
        )

    def copy(self):
        return self.__copy__()

    @property
    def x1(self):
        return self._bbox[0]

    @property
    def y1(self):
        return self._bbox[1]

    @property
    def x2(self):
        return self._bbox[2]

    @property
    def y2(self):
        return self._bbox[3]

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
        return self._get_height(self._bbox)

    @property
    def width(self):
        return self._get_width(self._bbox)

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

    @staticmethod
    def draw_segmentation_numpy(segmentation, img, *args, **kwargs):
        img = Image.fromarray(img)
        img = BBox.draw_segmentation_pil(segmentation, img, *args, **kwargs)
        return np.array(img)

    @staticmethod
    def draw_segmentation(segmentation, img, buffer=5, *args, **kwargs):
        """
        Args:
            seg: List of points, x1,y1,x2,y2,...
            img:
            color:

        Returns:

        """

        seg2d = np.array(segmentation).reshape(-1, 2)
        if seg2d.shape[0] > 2: # add an end point to close polygon if more than 2 points
            seg2d = np.concatenate([seg2d, seg2d[0:1]], axis=0)

        if isinstance(img, np.ndarray):
            img = BBox.draw_segmentation_numpy(seg2d, img, *args, **kwargs)
        else:
            if img is None:
                background_size = (np.max(seg2d, axis=0) + buffer).astype(int).to_list()
                img = Image.new("RGB", background_size, color="white")
            img = BBox.draw_segmentation_pil(seg2d, img, *args, **kwargs)

        return img
    @staticmethod
    def draw_segmentation_pil(seg2d, img, color="green", *args, **kwargs):
        if color=="random":
            color = tuple(np.random.randint(0, 255, 3).tolist())

        xy = seg2d.reshape(-1).tolist()

        if len(xy)<=4: # segmentations should always be in XYXY format; if len=4, then it's just x1,y1,x2,y2, so draw a box
            BBox._draw_box(xy[:4], img, color, *args, **kwargs)
        else:
            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, outline=color, *args, **kwargs)
        return img

    def draw_box(self, img, color="red"):
        BBox._draw_box(self._bbox, img, color)

    @staticmethod
    def _draw_box(bbox, img, color="red", *args, **kwargs):
        if color=="random":
            color = tuple(np.random.randint(0, 255, 3).tolist())

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
    def box_4pt(bbox):
        return [bbox[0], bbox[1], bbox[2], bbox[1], bbox[2], bbox[3], bbox[0], bbox[3]]

    @staticmethod
    def img_and_pos_to_bbox(img, pos):
        """ Return bbox tuple given an image and a position to paste it

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

    def __getitem__(self, i):
        return self.bbox[i]

    def __repr__(self):
        return self.bbox.__repr__()

    def __str__(self):
        return str(self.bbox)

    def toJSON(self):
        return self.bbox

    @staticmethod
    def get_maximal_box(list_of_bboxes):
        """
        list_of_bboxes: [x1,y1,x2,y2,xx1,yy1,xx2,yy2,...]
        Returns:
            The union of all bboxes

        """
        boxes = np.array(list_of_bboxes).reshape(-1,4)
        #boxes = np.array(range(0,20))
        bbox_list = [int(np.min(boxes[:, 0])),
        int(np.min(boxes[:, 1])),
        int(np.max(boxes[:, 2])),
        int(np.max(boxes[:, 3]))]
        return bbox_list



    @staticmethod
    def _top_left_format(bbox):
        return min(bbox[0], bbox[2]),min(bbox[1], bbox[3]),max(bbox[0], bbox[2]), max(bbox[1], bbox[3])

    @staticmethod
    def _top_left_format_vector(bbox_array):
        bbox_array = np.random.randint(0,10, [10,4])
        bbox_array = np.array(bbox_array).reshape(-1,4)
        output_array = np.zeros_like(bbox_array)
        output_array[:,[0,2]] = np.min(bbox_array[:,[0,2]], axis=1)
        output_array[:,[1,3]] = np.max(bbox_array[:,[1,3]], axis=1)
        return output_array


class BBoxNGon(BBox):
    __slots__ = BBox.__slots__  # Inherit slots from BBox

    def __init__(self,
                 origin: Literal['ul', 'll'],
                 bbox,
                 line_index=None,
                 line_word_index=None,
                 text=None,
                 parent_obj_bbox=None,
                 paragraph_index=None,
                 force_int=True,
                 img=None,
                 format: Literal['XYXY', 'XYWH'] = "XYXY",
                 height_ll=None
                 ):
        """
        Args:
            bbox_4gon [x1,y1,x2,y2,x3,y3,...]:

        Returns:

        """
        super().__init__(origin, bbox, line_index, line_word_index, text, parent_obj_bbox, paragraph_index, force_int, img, format, height_ll)
        self.format = "XYXY"
        self._bbox = np.asarray(bbox)

        # even number of points
        assert self._bbox.size % 2 == 0

    def y_coords(self):
        return self._bbox[1::2]

    @staticmethod
    def _y_coords(bbox_array):
        return bbox_array[1::2]

    def x_coords(self):
        return self._bbox[::2]

    @staticmethod
    def _x_coords(bbox_array):
        return bbox_array[::2]

    def invert_y_axis(self, height=None):
        height = self.height_ll if height is None else height
        coords = self._bbox.reshape(-1, 2)
        coords[:,1] = height - coords[:,1]
        self._bbox = coords.ravel()
        return self._bbox

    @staticmethod
    def _swap_bbox_axes(bbox):
        return np.asarray(bbox).reshape(-1,2)[:,::-1].ravel().tolist()

    @staticmethod
    def _offset_origin(bbox, offset_x=0, offset_y=0):
        bboxarray = np.asarray(bbox).copy()
        bboxarray[::2] += offset_x
        bboxarray[1::2] += offset_y
        return bboxarray

    def __getitem__(self, idx):
        return self._bbox[idx]

    def __len__(self):
        return len(self._bbox)

    @staticmethod
    def _get_dim(bbox):
        return BBoxNGon._get_width(bbox), BBoxNGon._get_height(bbox)

    @staticmethod
    def _get_height(bbox):
        ys = BBoxNGon._y_coords(bbox)
        return np.max(ys) - np.min(ys)

    @staticmethod
    def _get_width(bbox):
        xs = BBoxNGon._x_coords(bbox)
        return np.max(xs) - np.min(xs)

    @property
    def size(self):
        return self.get_dim()

    @property
    def height(self):
        return self._get_height(self._bbox)

    @property
    def width(self):
        return self._get_width(self._bbox)

    @staticmethod
    def draw_box_numpy(bbox, img, color="red", thickness=None, fill=None):
        raise NotImplementedError

    @staticmethod
    def draw_box_pil(bbox, img, color, fill=None):
        raise NotImplementedError

    @staticmethod
    def bbox_norm(bbox, image_w, image_h):
        return [coord/image_w if i%2==0 else coord/image_h for i,coord in enumerate(bbox)]

    @staticmethod
    def _draw_center(bbox, img, color):
        raise NotImplementedError

    @staticmethod
    def get_maximal_box_orthogonal(list_of_bboxes):
        """ JUST THE ORTHOGONAL BOX
        list_of_bboxes: [x1,y1,x2,y2,xx1,yy1,xx2,yy2,...]
        Returns:
            The union of the area of all bboxes

        """
        all_points = np.concatenate(list_of_bboxes)
        xcoords = BBoxNGon._x_coords(all_points)
        ycoords = BBoxNGon._y_coords(all_points)
        return np.array((np.min(xcoords), np.min(ycoords), np.max(xcoords), np.max(ycoords)))

    get_maximal_box = get_maximal_box_orthogonal


    @staticmethod
    def get_convex_hull(list_of_bboxes):
        """ Convex hull

        Args:
            list_of_bboxes: supposed to be a list of 4-tuples/list

        Returns:

        """
        points = np.array(flatten(list_of_bboxes)).reshape(-1,2)
        if len(points)>2:
            return points[ConvexHull(points).vertices].ravel()
        else:
            return points.ravel()

        # x = points[ConvexHull(points).vertices].reshape([-1,2])
        # plt.scatter(x[:,0],x[:,1]); plt.show()

    @staticmethod
    def concave_hull(bbox_xyxy_list):
        """ Returns the polygon of the bbox
        Get polygon from a list of XYXY bbox coordinates
            Assumes boxes are added from left to right


        Args:
            bbox:
            all_points:

        Returns:

        """
        coords = np.array(flatten(bbox_xyxy_list)).reshape(-1, 2)
        centroid = np.mean(coords, axis=0)
        sorted_coords = coords[np.argsort(np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])), :]
        return sorted_coords

    @staticmethod
    def print_categories_recursively(layout, level=0):
        print(f"{'   '*level}{layout.category} {layout.bbox}")
        for child in layout.children:
            BBox.print_categories_recursively(child)


def flatten(mylist):
    """Flatten a list of lists OR arrays

    Args:
        mylist:

    Returns:

    """
    return [item for sublist in mylist for item in ravel(sublist)]


def flatten2(list_of_bboxes):
    """ DEPRECATED

    Args:
        list_of_bboxes:

    Returns:

    """
    if isinstance(list_of_bboxes, np.ndarray):
        points = list_of_bboxes.ravel()
    elif isinstance(list_of_bboxes, list):
        if isinstance(list_of_bboxes[0], list):  # flatten nested list
            points = np.array([item for sublist in list_of_bboxes for item in sublist])
        elif not isinstance(list_of_bboxes[0], np.ndarray):  # list of ints
             points = np.array(list_of_bboxes).ravel()
        else:
            # List of ndarrays
            try:
                points = np.concatenate(list_of_bboxes)
            except:
                points = np.array([item for sublist in list_of_bboxes for item in ravel(sublist)])
    else:
        raise Exception("Unknown list type")
    return points

def flatten_test():
    # test flatten with list of lists and list of arrays
    a = [[1,2,3],[4,5]]
    b = [np.array([1,2]),np.array([3,4,5,6])]
    c = [[1,2,3,4],np.array([5,6])]
    d = [np.array([[1,2],[3,4]]),np.array([5,6]),[7,8,9]]
    e = [np.random.rand(100,5),np.random.rand(500).tolist()]

    for example in [a, b, c, d]:
        x = flatten(example)
        y = flatten2(example)
        assert np.all(x == y)
        print(x, y)

    # time which function is faster
    import timeit
    print(timeit.timeit("flatten(a); flatten(b); flatten(c); flatten(d); flatten(e)", setup="from __main__ import flatten, a,b,c,d,e", number=10000))
    print(timeit.timeit("flatten2(a); flatten2(b); flatten2(c); flatten2(d); flatten2(e)", setup="from __main__ import flatten2, a,b,c,d,e", number=10000))



if __name__=='__main__':
    pass
