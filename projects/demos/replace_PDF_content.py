from docgen.pdf_edit import PDF
from textgen.unigram_dataset import Unigrams
from docgen.rendertext.render_word import RenderWordFont
from hwgen.data.saved_handwriting_dataset import SavedHandwriting
import numpy as np
from docgen import utils
from docgen.utils import display as disp

PATH= r"C:\Users\tahlor\github\handwriting\handwriting\data\datasets\synth_hw\style_298_samples_0.npy"
PDF_FILE = r"C:\Users\tahlor\github\docx_localization\temp\TEMPLATE.pdf"
PDF_FILE = r"C:\Users\tahlor\Downloads\EXAMPLES\french_census_0000.pdf"

#PDF_FILE = r"C:\Users\tahlor\github\docx_localization\temp\example.pdf"

UNIGRAMS = r"C:\Users\tahlor\github\textgen\textgen\datasets\unigram_freq.csv"

""" 
COMPOSITE IMAGES: Paste word images -- take max of background/paste image, i.e. don't paste over gridlines

prep a real-looking dataset
SCALE TO FIT! build the object
Spacing for multiple word images

# TABLE ALL "SMART" FIT STUFF - just truncate the text for now!!!

### CREATE SAMPLE FORMS
    # What is the target dataset?
    # Target task?
        # Here is the transcription, localize it
        # Here is a document, autoregressively predict localization and word ****
            # Start with IAM dataset
    # Generate documents as DOCX
        # Populate form with synth data as fonts
        # Loop through, replace synth fonted data
    
# Word level replacer...
    # This should work for things generated in DOCX
    # BUT not great for PDF? MAYBE FDF pipeline would work
    # Still manually create / edit from DOCX, and then convert to PDF etc.

# More features:
    ## NEED scaler feature to work
    ## FILL textbox with paragraph
    
# Options for document creation:
    # Make the PDF, localize it, this becomes the template for replacing it
    # this would work at a textbox level
# FDF
    # Define an FDF PDF form
    # Define FDF formats for each form
    # Define different ways the form can be augmented / shifted / arranged
    # Define wholly invented FDF
    
# WANT: Phrase level replacer?
    # Have textboxes, fill with text so that it fits
    # Reject if you run out of space or truncate

# TODO:: Find typable fields in PDF and replace these with HWR
# TODO:: localization object, convert to/from numpy/PIL, offset, etc.

Later:
GridWarp

"""

def main():
    words = Unigrams(csv_file=UNIGRAMS)
    renderer2 = RenderWordFont(format="PIL")
    renderer = SavedHandwriting(
                                format="PIL",
                                dataset_path=PATH,
                                random_ok=True,
                                conversion=None #lambda image: np.uint8(image*255)
                                )

    r2 = renderer2.render_word(word="this")["image"]
    r = renderer.render_word(word="that")["image"]

    pdf = PDF(renderer=renderer)
    for i in range(1):
        images = pdf.replace_text_with_images(PDF_FILE, words, resize_words="height_only")

    utils.display(images[0])
    images[0][0].save(r"C:\Users\tahlor\Downloads\EXAMPLES\example.png")
    pass

if __name__ == '__main__':
    main()