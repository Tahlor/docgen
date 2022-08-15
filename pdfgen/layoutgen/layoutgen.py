import random
from pdfgen.bbox import BBox

def flip(prob=.5):
    return random.random() < prob


class Page:
    def __init__(self):
        pass

class Document:
    def __init__(self):
        self.bboxes = []

class LayoutGenerator:

    def  __init__(self,
                  pages_per_image=(1,2),
                  width=3200,
                  height=2400,
                  random_factor=2,
                  two_page_probability=.5,
                  top_margin_min=0,
                  bottom_margin_min=0,
                  left_margin_min=0,
                  right_margin_min=0,
                  top_margin_max=.1,
                  bottom_margin_max=.1,
                  left_margin_max=.1,
                  right_margin_max=.1,
                  margin_notes_probability=.5,
                  margin_notes_width_min=.1,
                  margin_notes_width_max=.3,
                  paragraph_note_probability=.5
                  ):
        self.width = width
        self.height = height
        self.random_factor = random_factor
        if isinstance(pages_per_image, int):
            self.pages_per_image = [pages_per_image]
        else:
            self.pages_per_image = list(range(*pages_per_image))

        self.width_upper = int(self.width * self.random_factor)
        self.width_lower = int(self.width / self.random_factor)
        self.height_upper = int(self.height * self.random_factor)
        self.height_lower = int(self.height / self.random_factor)

    def generate_layout(self):
        width = random.randint(self.width_lower, self.width_upper)
        height = random.randint(self.height_lower, self.height_lower)
        pages = random.choice(self.pages_per_image)
        current_x = 0
        page_width = int(width / pages)

        for page in range(0,pages):
            self.generate_layout(starting_x=current_x, page_width=page_width, page_height=height)
            current_x += page_width


    def generate_page(self, starting_x, page_width, page_height):


    def paragraph_box(self):
        pass

    def margin_column(self):
        pass

    def header_box(self):
        pass

    def paragraph_note_box(self):
        pass

if __name__ == "__main__":
    lg = LayoutGenerator()


    
