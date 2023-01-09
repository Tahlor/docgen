from docgen.layoutgen.layoutgen import *
from docgen.utils import display

"""
Uses layoutgen to generate French BMD layouts

"""

page_margins = MarginGenerator()

page_title_margins = MarginGenerator(top_margin=(-.02, .02),
                                     bottom_margin=(-.02, .02),
                                     left_margin=(-.02, .5),
                                     right_margin=(-.02, .5))
paragraph_margins = MarginGenerator(top_margin=(-.1, .1),
                                    bottom_margin=(-.1, .1),
                                    left_margin=(-.1, .1),
                                    right_margin=(-.1, .1))
margin_margins = MarginGenerator(top_margin=(-.1, .5),
                                 bottom_margin=(-.1, .1),
                                 left_margin=(-.1, .1),
                                 right_margin=(-.1, .1))
paragraph_note_margins = MarginGenerator(top_margin=(-.05, .2),
                                         bottom_margin=(-.05, .2),
                                         left_margin=(-.05, .2),
                                         right_margin=(-.05, .2))
lg = LayoutGenerator(paragraph_margins=paragraph_margins,
                     page_margins=page_margins,
                     margin_margins=margin_margins,
                     page_title_margins=page_title_margins,
                     paragraph_note_margins=paragraph_note_margins,
                     margin_notes_probability=1,
                     pages_per_image=(1, 3)
                     )

layout = lg.generate_layout()
image = lg.draw_doc_boxes(layout)
display(image)
