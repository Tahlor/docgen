from projects.demos.generate_lines import LineGenerator
from pathlib import Path
from docgen.utils.utils import timeout
import time
import os
# docker kill : 09cf725fba01
# docker rm : 09cf725fba01

# check if on ec2
if os.path.exists("/HOST"):
    DATASETS_PATH = Path("/HOST/home/ec2-user/docker/resources/datasets/")
    HUGGING_FACE_DATASETS_CACHE = Path("~/.cache/huggingface/datasets/")
    IMAGE_OUTPUT = Path("/HOST/home/ec2-user/docker/outputs")
# if host is galois
elif os.path.exists("/home/taylor"):
    DATASETS_PATH = Path("/home/ec2-user/docker/resources/datasets")
    HUGGING_FACE_DATASETS_CACHE = Path("~/.cache/huggingface/datasets")
    IMAGE_OUTPUT = Path("/home/ec2-user/docker/outputs")

preprocessed = ["en", "fr", "it", "de"]
languages = {
     #"fr": "french",
     #"en": "english",
     "la": "latin",
     "de": "german",
     "hun": "hungarian",
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
languages = {k: v for k, v in languages.items() if k in preprocessed}

def make_sys_link_for_wikipedia_files():
    # make sure the wikipedia files are in the right place
    path = DATASETS_PATH
    path.mkdir(parents=True, exist_ok=True)
    # ln -s /HOST/home/ec2-user/docker/resources/wikipedia ~/.cache/huggingface/datasets
    # python command for symlink command above
    os.symlink(HUGGING_FACE_DATASETS_CACHE, path)


def check_if_done(path):
    # check if path has more than 100000 files
    return len(list(path.glob("*"))) >= 999999

def run(language, abbreviation):
    path = IMAGE_OUTPUT / language
    if check_if_done(path):
        print(f"{language} is done")
        return

    args = f""" 
     --output_folder {str(path)} \
     --batch_size 200  \
     --save_frequency 50000 \
     --saved_handwriting_model IAM \
     --wikipedia 20220301.{abbreviation} \
     --canvas_size 1152,64 \
     --min_chars 8 \
     --max_chars 200 \
     --max_lines 1 \
     --max_paragraphs 1 \
     --count 1000000 \
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


def launch_background_rsync_task(path, destination):
    """ This won't really worker running from docker, need to launch process on host

    Args:
        path:
        destination:

    Returns:

    """
    import subprocess
    # Path(f"/HOST/mnt/s3-ds/computervision-l3apps/HWR/synthetic-data/languages/{language}-lines/v1/").mkdir(parents=True, exist_ok=True)
    #  aws s3 sync /home/ec2-user/docker/outputs/FRENCH_LINES_1_MILLION/lines/ s3://datascience-computervision-l3apps/HWR/synthetic-data/french-lines/v1/sample --exclude "*" --include "099999*"
    command = f"rsync -avz --progress {str(path)} {str(destination)} && echo 'rsync successful' && rm -rf {str(path)}"
    if False:
        subprocess.Popen(command, shell=True)
    else: # launch process on host, not docker container
        subprocess.Popen(f"ssh -i /HOST/home/ec2-user/.ssh/id_rsa ")

def test_downloading_wikipedia_using_huggingface_dataset_for_each_language():
    from datasets import load_dataset
    for language, abbreviation in languages.items():
        try:
            with timeout(seconds=2):
                load_dataset("wikipedia", language, split="train")
        except TimeoutError:
            print(f"{language} successfully timed out")
        except Exception as e:
            print(e)
            print(f"Failed to download {language}")

if __name__ == "__main__":
    for abbreviation, language in languages.items():
        run(language, abbreviation)