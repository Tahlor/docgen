page_margins = MarginGenerator()
page_header_margins = MarginGenerator(top_margin=(-.02, .02),
                                      bottom_margin=(-.02, .02),
                                      left_margin=(-.02, .5),
                                      right_margin=(-.02, .5))
paragraph_margins = MarginGenerator(top_margin=(-.05, .05),
                                    bottom_margin=(-.05, .05),
                                    left_margin=(-.05, .02),
                                    right_margin=(-.05, .02))
margin_margins = MarginGenerator(top_margin=(-.1, .2),
                                 bottom_margin=(-.05, .3),
                                 left_margin=(-.05, .1),
                                 right_margin=(-.08, .1))

paragraph_note_margins = MarginGenerator(top_margin=(-.05, .2),
                                         bottom_margin=(-.05, .2),
                                         left_margin=(-.05, .2),
                                         right_margin=(-.05, .2))

lg = LayoutGenerator(paragraph_margins=paragraph_margins,
                     page_margins=page_margins,
                     margin_margins=margin_margins,
                     page_header_margins=page_header_margins,
                     paragraph_note_margins=paragraph_note_margins,
                     margin_notes_probability=.5,
                     page_header_prob=.5,
                     paragraph_note_probability=.5,
                     pages_per_image=(1, 3)
                     )