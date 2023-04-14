from docgen.dataset_utils import *
OCR = r"C:\Users\tarchibald\github\data\synthetic\FRENCH_BMD_LAYOUTv0.0.0.1\OCR.json"

def ocr_to_coco():
    path=OCR
    ocr_dataset_to_coco(ocr_dict=path, data_set_name="Handwritten Pages")

if __name__ == '__main__':
    ocr_to_coco()
