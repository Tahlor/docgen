import sys
from docgen.bbox import BBox
from docgen.render_doc import composite_images_PIL
from PIL import Image

class Gen:

    def get(self):
        raise NotImplementedError()
    def __getitem__(self, item):
        return self.get()

    def __len__(self):
        return sys.maxsize

    def __iter__(self):
        return self

    def __next__(self):
        return self.get()

    def get_random_bbox(self, img_size=None, font_size=10):
        if img_size is None:
            img_size = self.img_size
        bbox = BBox("ul", [0,0,*self.img_size]).random_subbox(max_size_x=img_size[1],max_size_y=img_size[0], min_size_x=font_size*6, min_size_y=font_size)
        return bbox

    @staticmethod
    def composite_pil(img, bg, pos, offset):
        return composite_images_PIL(img, bg, pos, offset)

    @staticmethod
    def composite_pil_from_blank(img, bg_size, pos):
        """

        Args:
            img:
            bg_size: PIL size, (width, height)
            pos:

        Returns:

        """
        bg_img = Image.new("RGB", bg_size, (255,255,255))
        return composite_images_PIL(img, bg_img, pos, offset=(0,0))