from docgen.reportlab_tools.reportlab_generator import FormGenerator, filter_to_one_row
from docgen.content.table_from_faker import TableDataFromFaker
from docgen.img_tools import convert_pdf_to_img_paths
from hwgen.data.saved_handwriting_dataset import SavedHandwriting, SavedHandwritingRandomAuthor
from PIL import Image
from docgen.pdf_edit import fill_area_with_words, composite_images2
from docgen.rendertext.render_word import RenderImageTextPair
from docgen.bbox import BBox

functions = ["address",
             "relationship",
             "job",
             "date",
             "height",
             "phone_number",
             "ssn",
             "aba",
             "bank_country",
             "license_plate",
             "bban",
             "iban",
             "swift8",
             "ean13",
             "company",
             "currency",
             "date_time_this_century",
             "password",
             "paragraph"]

fg = FormGenerator()
fields = [["Name:", "Taylor", 50],
          ["Phone:", "610-573-7638", 50],
          ["Address:", "1973 Montana Ave", 100]
          ]

generator = TableDataFromFaker(functions=functions,
                               include_row_number=False,
                               random_fields=27)

one_row = filter_to_one_row(list(generator.gen_content(1))[0])
pdf_path = './temp/simple_form.pdf'
img = './temp/simple_form.png'
HWR_SAVED_DATASET_PATH = r"C:\Users\tarchibald\github\handwriting\handwriting\data\datasets\synth_hw"

localization = fg.create_new_form(zip(generator.header_names + generator.extra_headers, one_row), form_name=pdf_path)
convert_pdf_to_img_paths(pdf_path, img)
background_img = Image.open(img)

renderer = SavedHandwritingRandomAuthor(
    format="PIL",
    dataset_root=HWR_SAVED_DATASET_PATH,
    random_ok=True,
    conversion=None,  # lambda image: np.uint8(image*255)
    font_size=32
)


"""
NOT WORKING
# Make sure handwriting is in PIL
# Only replace "field" category
"""
for section in localization["sections"]:
    for d in section["paragraphs"]:
        if d["category"] != "field":
            continue
        text = d["text"].split(" ")
        img_word_pairs = RenderImageTextPair(renderer, text)
        image, localization = fill_area_with_words(img_word_pairs, d["bbox"], error_handling="force", text_list=None)
        composite_images2(background_img, image, BBox._force_int(d["bbox"]))

background_img.show()

import torchvision

imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
