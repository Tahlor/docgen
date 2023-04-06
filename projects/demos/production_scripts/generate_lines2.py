import torch
import os
DEVICE = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

END=2000000
from projects.demos.generate_lines import LineGenerator
from pathlib import Path
from docgen.utils.utils import timeout
import time
from projects.demos.production_scripts.generate_lines import check_if_done

galois_huggingface_cache = "/media/data/1TB/datasets/synthetic/huggingface/datasets"
# docker kill : 09cf725fba01
# docker rm : 09cf725fba01

# if host is galois
if os.path.exists("/HOST/home/taylor"):
    #  /home/taylor/.cache/huggingface/datasets/wikipedia/20230301.pl-21baa4c9bf4fe40f/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559
    DATASETS_PATH = Path("/HOST/media/data/1TB/datasets/synthetic/huggingface/datasets")
    WIKIPEDIA = DATASETS_PATH / "wikipedia"
    HUGGING_FACE_DATASETS_CACHE = Path("/HOST") / galois_huggingface_cache #"/HOST/home/taylor/.cache/huggingface/datasets"
    IMAGE_OUTPUT = Path("/HOST/media/data/1TB/datasets/synthetic")
    batch_size = 72 if DEVICE=="0" else 84
    print("On Galois Docker")
elif os.path.exists("/home/taylor"):
    #  /home/taylor/.cache/huggingface/datasets/wikipedia/20230301.pl-21baa4c9bf4fe40f/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559
    DATASETS_PATH = Path("/media/data/1TB/datasets/synthetic/huggingface/datasets")
    WIKIPEDIA = DATASETS_PATH / "wikipedia"
    HUGGING_FACE_DATASETS_CACHE = galois_huggingface_cache
    IMAGE_OUTPUT = Path("/media/data/1TB/datasets/synthetic")
    batch_size = 72 if DEVICE=="0" else 84
    print("On Galois")
# check if on ec2
elif os.path.exists("/HOST"): # /HOST/etc/hostname
    DATASETS_PATH = Path("/HOST/home/ec2-user/docker/resources/datasets/")
    WIKIPEDIA = DATASETS_PATH / "wikipedia"
    HUGGING_FACE_DATASETS_CACHE = Path("~/.cache/huggingface/datasets/")
    IMAGE_OUTPUT = Path("/HOST/home/ec2-user/docker/outputs")
    batch_size = 200
    print("On EC2")


preprocessed = ["en", "fr", "it", "de"]
languages = {
     "fr": "french",
     "en": "english",
     "la": "latin",
     "hu": "hungarian",
     "de": "german",
     "es": "spanish",
     "it": "italian",
     "pt": "portuguese",
     "nl": "dutch",
     "sv": "swedish",
     "da": "danish",
     "no": "norwegian",
     "fi": "finnish",
     "tr": "turkish",
     "pl": "polish",
}

# filter to processed
#languages = {k: v for k, v in languages.items() if k not in preprocessed}

def run(language, abbreviation):
    path = IMAGE_OUTPUT / language
    if check_if_done(path):
        print(f"{language} is done")
        return

    args = f"""
     --output_folder {str(path)} \
     --batch_size {batch_size}  \
     --save_frequency 50000 \
     --saved_handwriting_model IAM \
     --wikipedia 20220301.{abbreviation} \
     --cache_dir {str(DATASETS_PATH)} \
     --canvas_size 1152,64 \
     --min_chars 8 \
     --max_chars 200 \
     --max_lines 1 \
     --max_paragraphs 1 \
     --count {END} \
     --resume \
     --no_incrementer
     """

    try:
        lg = LineGenerator(args)
        lg.main()
        # create s3 folder if not exists on next line
        # launch_background_rsync_task(path, f"s3://datascience-computervision-l3apps/HWR/synthetic-data/languages/{language}-lines/v1/")
    except Exception as e:
        print(e)
        print(f"Failed to run {language}")

if __name__ == "__main__":
    for abbreviation, language in languages.items():
        run(language, abbreviation)

