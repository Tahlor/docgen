import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import json
from tqdm import tqdm
from docgen.reportlab_tools.reportlab_generator import FormGenerator, filter_to_one_row
from docgen.content.table_from_faker import TableDataFromFaker
from docgen.img_tools import convert_pdf_to_img_paths
from hwgen.data.saved_handwriting_dataset import SavedHandwritingRandomAuthor
from PIL import Image
from docgen.pdf_edit import BoxFiller
from docgen.rendertext.render_word import RenderImageTextPair
from docgen.bbox import BBox
from copy import deepcopy
from docgen.utils import file_incrementer
from docgen.dataset_utils import JSONEncoder

class FormerFiller:
    def __init__(self):
        self.renderer = SavedHandwritingRandomAuthor(
            format="PIL",
            dataset_root="CVL",
            random_ok=True,
            conversion=None,  # lambda image: np.uint8(image*255)
            font_size=32
        )

        self.functions = ["address",
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

        self.fg = FormGenerator()
        fields = [["Name:", "Taylor", 50],
                  ["Phone:", "610-573-7638", 50],
                  ["Address:", "1973 Montana Ave", 100]
                  ]

        self.generator = TableDataFromFaker(functions=self.functions,
                                       include_row_number=False,
                                       random_fields=27)

        self.bf = BoxFiller(random_word_idx=True)

    def generate_form(self):

        one_row = filter_to_one_row(list(self.generator.gen_content(1))[0])
        pdf_path = './temp/simple_form.pdf'
        img = './temp/simple_form.png'

        ocr_dict = self.fg.create_new_form(zip(self.generator.header_names + self.generator.extra_headers, one_row), form_name=pdf_path)
        convert_pdf_to_img_paths(pdf_path, img)
        background_img = Image.open(img)
        scale_factors = [background_img.size[0] / self.fg.document_width, background_img.size[1] / self.fg.document_height]
        return ocr_dict, background_img, scale_factors

    def fill_form(self, ocr_dict, background_img, scale_factors):
        for i, section in enumerate(ocr_dict["sections"]):
            for j,d in enumerate(section["paragraphs"]):
                if d["category"] != "field":
                    continue
                text = d["text"].strip().split(" ")
                img_word_pairs = RenderImageTextPair(self.renderer, text)
                d["bbox"] = BBox._rescale(d["bbox"], scale_factors)
                background_img, bbox_list = self.bf.fill_box(bbox=d["bbox"], img=background_img, img_text_pair_gen=img_word_pairs, error_mode="expand")
                d["text"] = bbox_list[0].text
                d["bbox"] = bbox_list[0].bbox

                #image, localization = fill_area_with_words(img_word_pairs, d["bbox"], error_handling="force", text_list=None)
                #composite_images2(background_img, image, BBox._force_int(d["bbox"]))

        return background_img, ocr_dict

    def main(self, output_path="./output", number_of_form_types=10, copies_of_each_form=10, test_set_ration=0.1):
        output_path = file_incrementer(output_path, create_dir=True)
        master_output = {}
        for form_type in tqdm(range(number_of_form_types)):
            ocr_dict, background_img, scale_factors = self.generate_form()
            output = master_output[form_type] = {}
            for variation in range(copies_of_each_form):
                if variation==0:
                    out_img, out_ocr = background_img, ocr_dict
                else:
                    out_img, out_ocr = self.fill_form(deepcopy(ocr_dict), background_img.copy(), scale_factors)
                name = f"{form_type:03.0f}_{variation:03.0f}"

                # if variation > copies_of_each_form * (1-test_set_ration):
                #     output_path = output_path / "test"
                #     output_path.mkdir(exist_ok=True)
                # else:
                #     output_path = output_path / "train"
                #     output_path.mkdir(exist_ok=True)

                save_path = output_path / f"{name}.jpg"
                out_img.save(save_path)
                out_ocr.update({"name": name, "path": save_path})
                output[variation] = out_ocr
        json_path = output_path / "output.json"

        with open(json_path, "w") as f:
            json.dump(master_output, f, cls=JSONEncoder)

if __name__ == "__main__":
    ff = FormerFiller()
    ff.main(copies_of_each_form=1000, number_of_form_types=10)