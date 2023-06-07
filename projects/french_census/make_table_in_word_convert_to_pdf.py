import random
import sys

import numpy.random
from docgen import docx2pdf
from textgen.table_from_faker import TableDataFromFaker
from docgen.docx_tools.docx_tools import *
from docgen.utils.utils import *
from docgen.pdf_edit import PDF
from textgen.rendertext.render_word import RenderWordFont
from hwgen.data.saved_handwriting_dataset import SavedHandwriting
from docgen.bbox import BBox
from docgen.dataset_utils import coco_dataset
from torch.utils.data import Dataset, DataLoader

if True:
    OFFICE_PATH=r"C:\Program Files\LibreOffice\program\swriter.exe"
    WORD = None
else:
    OFFICE_PATH=None
    import win32com.client
    WORD = win32com.client.Dispatch("Word.Application")


MARK=chr(96)
providers = [{"provider": "faker.providers.address", "locale": "fr_FR"},
                                {"provider": "faker.providers.person", "locale": "fr_FR"},
                                {"provider": "faker.providers.ssn", "locale": "fr_FR"},
                                {"provider": "faker.providers.phone_number", "locale": "fr_FR"},
             ]

def flip(prob=.5):
    return random.random() < prob


def make_docx(output_folder,
              idx=None,
              replace_selected_fields=True):
    # Get content generator
    french_census_title = "French Census"

    # Randomize headers
    functions = ["address", "relationship", "job", "date", "height", "phone_number", "ssn"]
    random.shuffle(functions)
    functions = ["name"] + functions[:5]
    mark_for_replacement_fields = functions[2:] if replace_selected_fields else []

    generator = TableDataFromFaker(functions=functions,
                                   include_row_number=True,
                                   provider_dict_list=providers,
                                   mark_for_replacement_fields=mark_for_replacement_fields,
                                   replacement_char=MARK
                                   )
    rows = random.randint(30,60)
    column_widths = numpy.random.random(len(generator)) * 3

    n_columns = len(generator)
    n_rows = rows + 1
    #column_widths = numpy.random.random(len(generator))[None,:].repeat(rows+1, axis=0) * 2
    #column_widths = [5]+[.25 for i in range(len(generator)-1)]
    #column_widths = numpy.random.random(n_columns*n_rows).reshape(n_rows, n_columns)
    #column_widths = None

    default_string = "french_census.docx"
    if idx is None:
        out = file_incrementer(output_folder / default_string, digits=4)
    else:
        _out = Path(output_folder) / default_string
        out = _out.parent / (_out.stem + f"_{idx:0{4}d}" + _out.suffix)

    doc = create_document_base(heading=french_census_title,
                         table_data=[generator.header_names]+list(generator.gen_content(rows)),
                         output=out,
                         widths=column_widths,
                         expand_row_height=True,
                         column_autofit=True,
                         grid=flip())
    return out


def StartWord(i):
    import win32com.client
    return win32com.client.Dispatch("Word.Application")

WORKERS = 4
def threading():
    import win32com.client
    import pythoncom

    pythoncom.CoInitialize()
    # words = [win32com.client.Dispatch("Word.Application") for i in range(WORKERS)]
    # word_ids = [pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, word) for i in range(WORKERS)]

    word = win32com.client.Dispatch("Word.Application")
    word_id = pythoncom.CoMarshalInterThreadInterfaceInStream(pythoncom.IID_IDispatch, word)

    def StartWord(i):
        #pythoncom.CoInitialize()
        return win32com.client.Dispatch(
                pythoncom.CoGetInterfaceAndReleaseStream(word_id, pythoncom.IID_IDispatch)
        )

    globals().update(locals())


class DocxGen(Dataset):
    def __init__(self,
                 root_dir="./temp/french_census",
                 pdf=None,
                 office_path=OFFICE_PATH):
        self.root_dir = file_incrementer(root_dir, create_dir=True)
        (self.root_dir / "images").mkdir(exist_ok=True, parents=True)
        self.pdf = pdf
        self.word_instance = StartWord
        self.office_path = office_path

    def __len__(self):
        return sys.maxsize

    def __getitem__(self,idx):
        failed = True
        while failed:
            if True:
                out = make_docx(self.root_dir, idx=idx)
                pdf_dict = self.docx_to_img(out, word_instance=WORD)
                failed = False
            #except Exception as e:
            #    print(e)
        return pdf_dict

    def docx_to_img(self, docx_file, clear_text=True, word_instance=None):
        # Convert to Image

        # Tempfile not working
        # pdf_path = tempfile.TemporaryFile(suffix=".pdf").name
        pdf_path = docx_file.with_suffix(".pdf")

        docx2pdf.convert(docx_file, pdf_path, word=word_instance, office_path=self.office_path)

        # Localize
        # localization = localize.generate_localization_from_file(pdf_path,
        #                   first_page_only=True,
        #                   mark_for_replacement=self.pdf.mark_for_replacement)

        # Only get first page
        images, new_localization = self.pdf.replace_text_with_images(pdf_path,
                                             request_same_word=True,
                                             localization=None,
                                             resize_words="width_only",
                                             first_page_only=True)
        image = images[0]
        new_localization = new_localization[0]

        img_path = docx_file.parent / "images" / docx_file.with_suffix(".png").name
        image.save(img_path)

        # Draw bboxes
        if False:
            for word_dict in new_localization[0]["localization_word"]:
                # next(iter(localization[0]["localization_word"]))["norm_bbox"]
                BBox.draw_box_pil(word_dict["bbox"], image, color="red")

        return {"image_path":str(img_path),
                "localization":new_localization,
                "image":image,
                "height":image.size[1],
                "width":image.size[0]
                }

def collate_fn(list_of_dicts):
    return list_of_dicts

def main_no_dataloader(doctor_gen: DocxGen):
    dicts = []
    for i in range(100):
        dicts.append(doctor_gen[i])

    out = doctor_gen.root_dir / "french_census_coco.json"
    coco_dataset(dicts, out)

def main(doctor_gen: DocxGen):
    doctor_loader = DataLoader(doctor_gen,
                               num_workers=int(WORKERS/2),
                               batch_size=1,
                               shuffle=False,
                               collate_fn=collate_fn)
    dicts = []
    for i,d in enumerate(doctor_loader):
        print(i)
        dicts.append(d)
        if i > 10:
            break

    out = doctor_gen.root_dir / "french_census_coco.json"
    coco_dataset(dicts, out)

def small_test(doctor_gen):
    #make_docx(output_folder=doctor_gen.root_dir)
    doctor_gen[0]

if __name__ == '__main__':
    renderer = RenderWordFont(format="numpy")
    pdf = PDF(renderer=renderer, mark_for_replacement_char=MARK)
    root = Path("../../temp/french_census")
    doctor_gen = DocxGen(root_dir=root,
                         pdf=pdf)

    small_test(doctor_gen)



""" HIGH-LEVEL STRATEGIES:
# Define word document table / form
    # Use autofit where possible
    # Vary font/size of text written fields
# Define variable fields WITH some keyword that tells it what should be filled in
# Use my fill in with images (HW or text)
    # skip_bad, fill until full
        # with skip_bad, make sure it ends early when out of vertical space
    # don't offset down/up if that would put it outside of the bounding box

### OPTION 2:
    # don't autofill in WORD; lock boxes in certain size
    # PDF generates bounding boxes
    # Use FITZ to get only the words in the bounding box
    
### OPTION 3:
    # allow word to auto fill and generate a typed version of the document
    # replace stuff at the word level
    # 

THIS OPTION:
    # FILL OUT FORM WITH A SPECIAL MARK FOR TEXTBOXES THAT SHOULD BE REPLACED
        # PROBLEM: textboxes aren't perfect in PDF -- mostly when some text isn't visible (not wrapped into cells)

# Paradigm 0:
    Do everything in PIL / Python
    Pros: Fast, flexible
    Cons: not easy to generate specific templates
          formatting features must be painstakingly coded / vetted
    
# Paradigm 1:
    # Generate entire form word
    # Paste HW images into word?
        # UNSOLVED: 
            # How to get localization of images? Still need to render document and convert to image
            # Need to preserve labels from images; steganogroaphy needed???
     
# Paradigm 2: ** WIP **
    # Generate form in word
    # Convert to PDF to get localization 
    # Replace words in PDF with HW images appropriately sized
        # Have some extra character to designate which textboxes should be replaced
        # Editing textboxes in PyPDF doesn't work great, so if character appears, you blank out the whole textbox and replace
        
# Paradigm 3: ** TABLED, but easy to come back to **
    # Generate form in word, only as a template, with keywords in the appropriate fields like *date, *name, etc.
    # Convert to PDF
    # "fill in" textboxes with appropriate data
    
# Paradigm 4:
    # Start with PDF form OR have Acrobat convert an image to a form
    # Let ADOBE automatically find the fields
    # Populate fields with stuff
        # FDF
    
UNSOLVED DOCX PROBLEMS:
    # AUTOFIT COLUMNS
        # Start with small columns, use autofit to expand
            # Fill in text with nonbreaking spaces or hyphens
        # Use canvas that is plenty big
            
    # gridspan: different sized columns
    # small textboxes within columns, i.e., form cell title vs form cell value
    # move away from tables and use textboxes?
        
POSSIBLE FEATURES:
    # Auto randomize columns to a size that will fit
        # Get initial character width, look at generated data, size them accordingly
    # checkboxes

##### NEED TO DO:
    # Generate with handwriting - dataloader approach
    # Localization: option to invert y initially WHEN CREATING PDF BOUNDING BOXES
                    (before replacing them)
    # Replace word option: replace text in textbox with specific kind of data
                           resize to fit line
                           fill line

    # Better font specification for WordFontGen other than random
    # OOV Font symbols
    # Multithreading
    # Localization levels: right now it's actually a "phrase" localization, optionally want
             line localization, 
             word localization,
             cell localization
    # Cropping
        # Remove header line
        # just generate so it's plenty big, then crop
    # 
    # Degradations
    # Categories - date, name, address, phone, etc.
    
ASAP:
    ## REQUIREMENTS
    ## PUSH GIT

LIBRE SUCKS!!!!!
DON'T DELETE IF NOT REPLACE!!!
"""