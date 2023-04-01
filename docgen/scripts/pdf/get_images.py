from tqdm import tqdm
from docgen.pdf_edit_new import extract_images_from_pdf
from pathlib import Path
import argparse

def get_pdf_files(directory: Path, recursive: bool = True):
    """Retrieve all PDF files from the given directory."""
    if recursive:
        return list(directory.glob("**/*.pdf"))
    else:
        return list(directory.glob("*.pdf"))

def save_image(image_name: str, image_data: bytes, output_dir: Path):
    """Save the image data to the specified directory with the given name."""
    image_path = output_dir / image_name
    with open(image_path, "wb") as image_file:
        image_file.write(image_data)

def save_images_from_pdfs_in_dir(pdf_directory: Path,
                                 output_dir: Path = None,
                                 overwrite: bool = False,
                                 recursive: bool = True):
    """Main function to orchestrate the extraction process."""
    pdf_directory = Path(pdf_directory)
    if pdf_directory.is_file():
        pdf_files = [pdf_directory]
        pdf_directory = pdf_directory.parent
    else:
        pdf_files = get_pdf_files(pdf_directory, recursive=recursive)
    output_root_folder = pdf_directory / "images" / "raw_images"  if output_dir is None else Path(output_dir)

    for pdf_path in tqdm(pdf_files):
        images = extract_images_from_pdf(pdf_path)
        output_dir = output_root_folder / pdf_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_name, image_data in (images.items()):
            if overwrite or not (output_dir / image_name).exists():
                save_image(image_name, image_data, output_dir)

    print(f"Images extracted and saved in {output_dir}.")

def parser(args=None):
    parser = argparse.ArgumentParser(description="Extract images from PDFs in a directory.")
    parser.add_argument("pdf_directory", type=Path, help="The directory containing the PDF files.")

    if args is not None:
        import shlex
        return parser.parse_args(shlex.split(args))
    else:
        return parser.parse_args()

if __name__ == "__main__":
    args = """G:/s3/forms/PDF/IRS"""
    args = parser(args)
    main(args.pdf_directory)