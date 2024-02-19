import argparse
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import shlex
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
def is_blank(image: Image.Image, threshold: float = 0.99) -> bool:
    """Check if an image is almost completely blank."""
    grayscale = image.convert('L')
    histogram = grayscale.histogram()
    total_pixels = sum(histogram)
    white_pixels = histogram[-1]
    return white_pixels / total_pixels >= threshold

def crop_to_content(image: Image.Image, threshold: int = 245) -> Image.Image:
    """Crop an image to its content."""
    grayscale = image.convert('L')
    binary = grayscale.point(lambda p: p < threshold and 255)  # Convert to binary image
    bbox = binary.getbbox()  # Get bounding box of non-white region
    if bbox:
        return image.crop(bbox)
    return image  # Return original image if no content found

def extract_images_from_pdf(pdf_path: Path, output_dir: Path):
    """Extract images from a PDF and save them to the output directory."""
    images = convert_from_path(pdf_path)
    saved_image_count = 0
    for page_num, image in enumerate(images, start=1):
        cropped_image = crop_to_content(image)

        if not is_blank(image) and cropped_image.size[0] > 100 and cropped_image.size[1] > 100:
            cropped_image.save(output_dir / f"{pdf_path.stem}_{page_num}.png")
            saved_image_count += 1

    logger.info(f"Saved {saved_image_count} images out of {len(images)} pages from {pdf_path}.")


def get_processed_pdfs(output_dir: Path) -> set:
    """
    Return a set of PDF stems that have already been processed.

    Args:
        output_dir (Path): The directory containing the extracted images.

    Returns:
        set: A set of PDF stems.
    """
    processed_pdfs = set()
    for image_path in output_dir.glob("*.png"):
        stem = image_path.stem
        pdf_stem = "_".join(stem.split("_")[:-1])  # Remove the _XX.png part
        processed_pdfs.add(pdf_stem)
    return processed_pdfs

def parser(args=None):
    parser = argparse.ArgumentParser(description="Extract non-blank images from PDFs in specified directories.")
    parser.add_argument("root", type=Path, help="Root directory containing the PDF folders.")
    parser.add_argument("--output_dir", type=Path, help="Where to save IMGs", default=None)

    if args is not None:
        args = parser.parse_args(shlex.split(args))
    else:
        args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.root.parent / "images"
    args.root = Path(args.root)
    return args

def main(args=None):
    args = parser(args)

    pdf_dir = args.root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs_already_processed = get_processed_pdfs(output_dir)

    for i, pdf_path in tqdm(enumerate(pdf_dir.glob("*.pdf"))):
        if pdf_path.stem in pdfs_already_processed:
            logger.info(f"Skipping {pdf_path} as it has already been processed.")
            continue

        logger.info(f"Extracting images from {pdf_path}")
        try:
            extract_images_from_pdf(pdf_path, output_dir)
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {e}")

def loop_through_pdf_repos():
    PDF_repos = ["G:/s3/forms/PDF/OPM", "G:/s3/forms/PDF/SSA"]
    PDF_repos = ["G:/s3/forms/PDF/IRS"]
    variants = ["other_elements", "text"]
    variants = ["text"]
    for variant in variants:
        for folder in PDF_repos:
            folder = Path(folder)
            input_folder = f"{(folder / variant) / 'PDF'}".replace("\\","/")
            output_folder = f"{(folder / variant) / 'images'}".replace("\\","/")
            main(f"{input_folder} --output_dir {output_folder}")


if __name__ == "__main__":
    # args = "G:/s3/forms/PDF/IRS/other_elements/PDF --output_dir G:/s3/forms/PDF/IRS/other_elements/images"
    # args = None
    # main(args=args)
    loop_through_pdf_repos()
