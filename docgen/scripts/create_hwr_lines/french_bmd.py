from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
import csv
import pickle
from PIL import Image, ImageDraw

"""
For J2K support on Windows, download pillow from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow

"""
class ImageCropper:
    def __init__(self, image_dir: Path):
        self.image_dir = image_dir

    def crop_polygon(self, image_name: str, coords: list) -> Image.Image:
        image_path = self.find_image_path(image_name)
        with Image.open(image_path) as img:
            mask = Image.new('L', img.size, 0)
            ImageDraw.Draw(mask).polygon(coords, outline=255, fill=255)
            result = img.convert("RGBA")
            datas = result.getdata()
            newData = []
            for item, mask_value in zip(datas, mask.getdata()):
                if mask_value == 255:
                    newData.append(item)
                else:
                    newData.append((255, 255, 255, 0))
            result.putdata(newData)
            bbox = mask.getbbox()
            return result.crop(bbox)

    def find_image_path(self, image_name: str) -> Path:
        """Find the image path given its name, regardless of its extension."""
        if self.image_dir.joinpath(image_name).exists():
            return self.image_dir.joinpath(image_name)

        for ext in ['.j2k', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:  # Add other extensions if needed
            potential_path = self.image_dir / (image_name + ext)
            if potential_path.exists():
                return potential_path
        raise FileNotFoundError(f"No matching file found for {image_name} in {self.image_dir}")

    def process_transcription_file(self, transcription_file: Path, output_dir: Path, overwrite=False) -> dict:
        transcriptions = {}
        with open(transcription_file, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='|')
            for row in tqdm(reader):
                try:
                    result = self.process_one_line(row, output_dir, overwrite)
                    if result:
                        transcriptions.update(result)
                    else:
                        print(f"{row['ImageName']} already exists")
                except Exception as e:
                    print(f"Error processing {row['ImageName']}")
                    print(e)
        return transcriptions

    def process_one_line(self, row: dict, output_dir: Path, overwrite=False) -> dict:
        if 'Paragraph_PRect' in row:
            coords = self.convert_str_to_ints(row['Paragraph_PRect'])
        elif 'line_region' in row:
            coords = self.convert_str_to_ints(row['line_region'])
        else:
            return None

        new_image_name = f"{row['ImageName'].split('.')[0]}_{coords[0]}_{coords[1]}.png"
        transcription_dict = {new_image_name: row.get('noisy_transcription', row.get('line_orig', ''))}
        if not overwrite and (output_dir / new_image_name).exists():
            pass
        else:
            image_path = self.find_image_path(row['ImageName'])
            cropped_img = self.crop(image_path, coords)
            cropped_img.save(output_dir / new_image_name, "PNG")
        return transcription_dict

    def crop(self, image_path: Path, coords: list):
        if len(coords) > 4:
            return self.crop_polygon(image_path, coords)
        else:
            return self.crop_image(image_path, coords)

    @staticmethod
    def convert_str_to_ints(coord_str: str) -> tuple:
        return tuple(map(lambda x: int(float(x)), coord_str.split(',')))

    def crop_image(self, image_name: str, coords: tuple) -> Image.Image:
        image_path = self.image_dir / image_name
        with Image.open(image_path) as img:
            cropped_img = img.crop(coords)
        return cropped_img
def main():
    image_dir = Path("G:/s3/french_bmd/transcription_pairs/images")
    cropper = ImageCropper(image_dir)

    datasets = ["train", "valid", "test"]
    transcription_dicts = {}

    for dataset in datasets:
        transcription_file = Path(f"G:/s3/french_bmd/transcription_pairs/transcriptions/version3/frenchbmd_{dataset}_transcriptions.txt")
        output_dir = Path(f"G:/s3/french_bmd/transcription_pairs/transcriptions/version3/{dataset}_images")
        output_dir.mkdir(parents=True, exist_ok=True)

        transcriptions = cropper.process_transcription_file(transcription_file, output_dir)
        transcription_dicts[dataset] = transcriptions

    with open("transcriptions.pkl", "wb") as file:
        pickle.dump(transcription_dicts, file)

if __name__ == "__main__":
    main()
