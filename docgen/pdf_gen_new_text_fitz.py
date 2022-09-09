import fitz
from pdf_edit import create_new_textbox
from math import ceil

def add_some_textboxes_test(input_path, output_path):
    doc = fitz.open(input_path)
    page = doc[0]  # choose some page
    try:
        auto_create_new_text(page, rect=(20,20,300,21), font_size=8, size_constraint="box")
    except Exception as e:
        print(f"Exception: {e}")
    try:
        auto_create_new_text(page, rect=(20,20,300,28), font_size=8, size_constraint="both")
    except Exception as e:
        print(f"Exception: {e}")

    auto_create_new_text(page, rect=(20, 20, 300, 28), font_size=8, size_constraint="font")
    auto_create_new_text(page, rect=(20, 20, 300, 34), text="Message1", font_size=16, size_constraint="box")

    doc.save(output_path)

def auto_create_new_text(page,
                       rect,
                       text="THIS IS MY NEW TEXT",
                       font_name ="Times-Roman",
                       font_size = 14,
                       draw_box=True,
                       box_color=(.25,1,.25),
                       size_constraint="font",
                       margin = 2
                       ):
    """
        size_constraint (font, box): font: font_size is locked, make text box bigger
                                     box: box is locked, make font_size smaller
                                     both: both constraints are active

    Returns:

    """
    text_length = fitz.get_text_length(text,
                                   fontname=font_name,
                                   fontsize=font_size)
    box_width = rect[2]-rect[0]
    box_height = rect[3]-rect[1]
    too_big = text_length + margin > box_width or font_size + margin > box_height

    if too_big:
        if size_constraint == "box":
            box_width = rect[2]-rect[0]
            # *2 just because 1pt is too small for a char. It mantains a good ratio for rect's width with larger text, but behaviour is not assured.
            new_font_size = min(box_width / len(text) * 2, box_height - margin)

            if box_height < new_font_size + margin or new_font_size < 1:
                raise Exception(f"Box height {box_height} too small for font {font_size} with margin {margin} and constrained box")

            print(f"Font {font_size} too big for box {rect}, using {new_font_size}")
            font_size = new_font_size
        elif size_constraint == "font":
            rect_x2 = ceil(rect[0] + text_length + margin)  # needs margin
            rect_y2 = ceil(rect[1] + font_size + margin)  # needs margin
            new_rect = rect[0:2] + (rect_x2, rect_y2)
            print(f"Rectangle {rect} too small for font size {font_size}, new rectangle {new_rect}")
            rect  = new_rect
        elif size_constraint == "both":
            raise Exception(f"Box height {box_height} too small for font {font_size} with margin {margin}")
        else:
            raise Exception(f"Unexpected {size_constraint} size_constraint option")

    return create_new_textbox(page, rect, text, font_name, font_size, draw_box, box_color)
