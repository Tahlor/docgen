from PyPDF2 import PdfReader, PdfWriter
from pathlib import Path
from PyPDF2.generic import TextStringObject, NameObject
from docgen.pdf_edit_new import delete_all_text_content, keep_text_only, customize_pdf_content
from io import BytesIO
from docgen.utils.utils import file_incrementer
from tqdm import tqdm
from pathlib import Path
from io import BytesIO
import logging
from docgen.scripts.pdf.get_images import save_images_from_pdfs_in_dir

logger = logging.getLogger(__name__)

"""
TODO:
* Some form elements still have images, some text (top and bottom of page)
* Some text has _______ in it, which looks like a form element
* MANY pictures are just images of text (or form elements)
"""

def strip_pdf(input_pdf_path: Path, output_folder: Path, overwrite=False, **kwargs) -> None:
    """
        https://www.irs.gov/forms-instructions-and-publications
    Strips text from a PDF located at input_pdf_path and saves the stripped PDF at output_pdf_path.

    Args:
        input_pdf_path (Path): The path to the input PDF.
        output_folder (Path): The path to save the output PDF.

    Returns:
        None
    """
    # Read input PDF into a BytesIO object
    with open(input_pdf_path, 'rb') as f:
        input_bytes_io = BytesIO(f.read())

    suffix=""
    save_images_path = False
    for property in kwargs:
        if kwargs[property] and property.startswith("keep_"):
            suffix += ("_" if suffix else "") + property.replace("keep_", "")

    folder = Path(output_folder / suffix) / "PDF"
    folder.mkdir(exist_ok=True, parents=True)
    output_path = (folder / input_pdf_path.name)
    if output_path.exists() and not overwrite:
        logger.info(f"Skipping {input_pdf_path.name} because it already exists.")
        return
    output_bytes = customize_pdf_content(input_bytes_io,
                                         output_path=output_path,
                                         first_page_only=False,
                                         **kwargs)



def do_one_file(input_file, output_folder, overwrite=False):
    input_file = Path(input_file)
    output_folder = Path(output_folder)
    strip_pdf(input_file, output_folder, keep_other_elements=True, overwrite=overwrite)
    strip_pdf(input_file, output_folder, keep_text=True, overwrite=overwrite)
    save_images_from_pdfs_in_dir(input_file, output_dir=output_folder / "raw_images")


def do_sample():
    do_one_file("G:/s3/forms/PDF/IRS/fw4.pdf","G:/s3/forms/PDF/IRS", overwrite=True)


if __name__ == "__main__":
    do_sample()

    #input_pdf_path = Path("C:/Users/tarchibald/Documents/fw4.pdf")
    #input_pdf_path = Path("C:/Users/tarchibald/Downloads/metadc1757662_m1/full_pdf_data/1.ForRepo/2C5RKIBUAWOITURJRXP244UPY2QHUQXM/2C5RKIBUAWOITURJRXP244UPY2QHUQXM.pdf")
    # PDF_repo = r"G:/s3/forms/PDF/IRS"
    # output_pdf_path = "G:/s3/forms/PDF/IRS"
    #PDF_repos = ["G:/s3/forms/PDF/OPM", "G:/s3/forms/PDF/SSA"]
    PDF_repos = ["G:/s3/forms/PDF/IRS"]
    output_pdf_paths = PDF_repos

    for PDF_repo, output_pdf_path in zip(PDF_repos, output_pdf_paths):
        for file in tqdm(Path(PDF_repo).glob("*.pdf")):
            try:
                do_one_file(file, output_pdf_path)
            except Exception as e:
                logger.exception(f"Error processing {file}")
                logger.exception(e)

    # Convert to images, only keep if some minimum number of pixels is not blank
    # Use original filename plus page number
