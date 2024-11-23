import random
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfform
from reportlab.lib.colors import magenta, pink, blue, green
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from pathlib import Path
import os
import sys
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
from easydict import EasyDict as edict
from reportlab.pdfbase.pdfmetrics import stringWidth
from docgen.bbox import BBox
folder = Path(os.path.dirname(__file__))
from docgen.reportlab_tools.reportlab_generator import FormGenerator
from report_lab_tools import pdf_to_png
import json
from textgen.unigram_dataset import Unigrams, UnigramsData
from textgen.field_generator.format_form_fields import Field

def save_text(path: Path, text: str):
    with open(path, "w") as f:
        f.write(text)

def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, default=vars)


if __name__ == '__main__':
    from textgen.field_generator.table_from_csv import CSVDataSampler
    file_path = Path("C:/Users/tarchibald/github/trex_eric/trex_cache/s3/acom-datascience-trex/datasets/draft-cards/korean_war/release_1/raw_labels/kwdc_20240409_1002_val_small.txt")
    mapping = {
        "Name": [
            [
                {"generator": "Surname", "probability": 1.0},
                {"value": ", ", "probability": 1.0},
                {"generator": "GivenName", "probability": 1.0},
            ],
            [
                {"generator": "GivenName", "probability": 1.0},
                {"generator": "Surname", "probability": 1.0},
            ]
        ],
        "Place of residence": [
            {"generator": "ResidenceCity", "probability": 1.0},
            {"value": ", ", "probability": 1.0},
            {"generator": "ResidenceState", "probability": 1.0}
        ],
        "Mailing address": [
            {"generator": "ResidenceCity", "probability": 1.0},
            {"value": ", ", "probability": 0.5},
            {"generator": "ResidenceState", "probability": 1.0}
        ],
        "Name and address of person who will always know your address": [
            {"generator": "RelativeGivenName", "probability": 1.0},
            {"generator": "RelativeSurname", "probability": 1.0}
        ],
        "Date of birth": [
            {"generator": "BirthMonth", "probability": 1.0},
            {"generator": "BirthDay", "probability": 1.0},
            {"value": ", ", "probability": .5},
            {"generator": "BirthYear", "probability": 1.0}
        ],
        "Place of birth": [
            {"generator": "BirthCity", "probability": 1.0},
            {"value": ", ", "probability": 0.5},
            {"generator": "BirthState", "probability": 1.0}
        ],
        "Occupation": [
            {"generator": "Occupation", "probability": 1.0}
        ],
        "Local board with which registered": [
            {"generator": "RegistrationCity", "probability": 1.0}
        ],
        "Marital status": [
            {"generator": "RelationToHead", "probability": 1.0}
        ],
        "Color of eyes": [
            {"generator": "MilitaryEyeColor", "probability": 1.0}
        ],
        "Color of hair": [
            {"generator": "MilitaryHairColor", "probability": 1.0}
        ],
        "Height": [
            {"generator": "MilitaryHeight", "probability": 1.0}
        ],
        "Weight": [
            {"generator": "MilitaryWeight", "probability": 1.0}
        ],
        # "Other obvious physical characteristics": [
        #     {"generator": "MilitaryComplexion", "probability": 1.0}
        # ],
        # "Nature of business, service rendered, or chief product": [
        #     {"generator": "Occupation", "probability": 1.0}
        # ],
        "Registration Date": [
            {"generator": "RegistrationMonth", "probability": 1.0},
            {"generator": "RegistrationDay", "probability": 1.0},
            {"value": ", ", "probability": .5},
            {"generator": "RegistrationYear", "probability": 1.0}
        ],
        "Complexion": [
            {"generator": "MilitaryComplexion", "probability": 1.0}
        ],
        "Race": [
            {"generator": "Race", "probability": 1.0}
        ],
        # "Date of registration": [
        #     {"generator": "RegistrationMonth", "probability": 1.0},
        #     {"value": " ", "probability": 1.0},
        #     {"generator": "RegistrationDay", "probability": 1.0},
        #     {"value": ", ", "probability": 1.0},
        #     {"generator": "RegistrationYear", "probability": 1.0}
        # ]
    }

    mapping2 = {
        "Name": [
            [
                {"generator": "Surname", "probability": 1.0},
                {"value": ", ", "probability": 1.0},
                {"generator": "GivenName", "probability": 1.0},
            ],
            [
                {"generator": "GivenName", "probability": 1.0},
                {"generator": "Surname", "probability": 1.0},
            ]
        ],
        # "Place of residence": [
        #     {"generator": "ResidenceCity", "probability": 1.0},
        #     {"value": ", ", "probability": 1.0},
        #     {"generator": "ResidenceState", "probability": 1.0}
        # ],
    }

    generator = CSVDataSampler(file_path, mapping, num_fields_range=(10, 20), blank_probability=0.0)
    fg = FormGenerator(fill_fields_with_text=True, document_height=400)
    folder = Path("F:/synth_draft_cards")
    folder.mkdir(exist_ok=True)
    pdf_folder = folder / "pdfs"
    img_folder = folder / "images"
    json_folder = folder / "jsons"

    for f in [pdf_folder, img_folder, json_folder]:
        f.mkdir(exist_ok=True)

    UNIGRAMS = r"C:\Users\tarchibald\github\textgen\textgen\datasets\unigram_freq.csv"
    word_gen = Unigrams(csv_file=UNIGRAMS)

    COUNT = 1000
    for i, row in enumerate(generator.gen_content(COUNT)):
        form_title = " ".join(word_gen.unweighted_sample(n=2)).title()
        pdf_path = folder / "pdfs" / f"{i}.pdf"
        # print(row.get_complete_dict())
        # print(row.data)
        #row = row.get_complete_dict()
        ocr = fg.create_new_form(row, form_name=str(pdf_path), form_title=form_title)
        save_json(folder/"jsons"/f'{i}.json', ocr)
        img_path = img_folder / f"{i}.png"
        pdf_to_png(pdf_path, img_path)

    # print the location
    print(f"Form saved to {folder}")