from docgen.layoutgen.segmentation_dataset.hw_gen import *
class PrintedTextGenerator:

    def get(self, img_size=None):
        font_size = random.randint(self.min_font_size, self.max_font_size)
        bbox = self.get_random_bbox(img_size=img_size, font_size=font_size)

        box_dict = self.filler.randomly_fill_box_with_words(bbox,
                                                       max_words=random.randint(1, 10),
                                                       )
        return box_dict


if __name__=="__main__":
    hwgen = PrintedTextGenerator()
    for i in range(10):
        box_dict = hwgen.get()
        box_dict["img"].show()


