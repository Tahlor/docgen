import torch
import os
DEVICE="0;1"

END=2000000
from pathlib import Path
from docgen.utils.utils import timeout
import time

galois_huggingface_cache = "/media/data/1TB/datasets/synthetic/huggingface/datasets"
# docker kill : 09cf725fba01
# docker rm : 09cf725fba01

def determine_host():
    host_docker = "/HOST/etc/hostname"
    host_normal = "/etc/hostname"
    docker = Path(host_docker).exists()
    if docker:
        # read  host
        with open(host_docker, "r") as f:
            host = f.read()
    else:
        with open(host_normal, "r") as f:
            host = f.read()
    return host.lower().strip(), docker

class Config:
    def __init__(self, end_idx=END, device=DEVICE):
        self.end_idx = end_idx
        os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

        # if host is galois
        host, docker = determine_host()
        if host == "galois":
            if docker:
                #  /home/taylor/.cache/huggingface/datasets/wikipedia/20230301.pl-21baa4c9bf4fe40f/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559
                self.DATASETS_PATH = Path("/HOST/media/data/1TB/datasets/synthetic/huggingface/datasets")
                self.WIKIPEDIA = self.DATASETS_PATH / "wikipedia"
                self.HUGGING_FACE_DATASETS_CACHE = Path("/HOST") / galois_huggingface_cache.rstrip("/") #"/HOST/home/taylor/.cache/huggingface/datasets"
                self.IMAGE_OUTPUT = Path("/HOST/media/data/1TB/datasets/synthetic")
                self.batch_size = 72 if device=="0" else 84
                print("On Galois Docker")
            else:
                #  /home/taylor/.cache/huggingface/datasets/wikipedia/20230301.pl-21baa4c9bf4fe40f/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559
                self.DATASETS_PATH = Path("/media/data/1TB/datasets/synthetic/huggingface/datasets")
                self.WIKIPEDIA = self.DATASETS_PATH / "wikipedia"
                self.HUGGING_FACE_DATASETS_CACHE = galois_huggingface_cache
                self.IMAGE_OUTPUT = Path("/media/data/1TB/datasets/synthetic")
                self.batch_size = 72 if device=="0" else 84
                print("On Galois")
        elif docker and "ec2" in host: # /HOST/etc/hostname
            self.DATASETS_PATH = Path("/HOST/home/ec2-user/docker/resources/datasets/")
            self.WIKIPEDIA = self.DATASETS_PATH / "wikipedia"
            self.HUGGING_FACE_DATASETS_CACHE = Path("~/.cache/huggingface/datasets/")
            self.IMAGE_OUTPUT = Path("/HOST/home/ec2-user/docker/outputs")
            self.batch_size = 200
            print("On EC2")
        else:
            raise Exception(f"Unknown docker/host combo: {host} Docker: {docker}")
    def make_sys_link_for_wikipedia_files(self):
        raise Exception("Not reliable, since we may or may not be on DOCKER, syslink might point to wrong place")
        # make sure the wikipedia files are in the right place
        path = self.DATASETS_PATH
        path.mkdir(parents=True, exist_ok=True)
        # ln -s /HOST/home/ec2-user/docker/resources/wikipedia ~/.cache/huggingface/datasets
        # python command for symlink command above
        if not HUGGING_FACE_DATASETS_CACHE is None and HUGGING_FACE_DATASETS_CACHE.exists():
            os.symlink(HUGGING_FACE_DATASETS_CACHE, path)

    def run(self, language, abbreviation):
        path = self.IMAGE_OUTPUT / language
        if check_if_done(path, count=self.end_idx):
            print(f"{language} is done")
            return

        args = f"""
         --output_folder {str(path)} \
         --batch_size {self.batch_size}  \
         --save_frequency 50000 \
         --saved_handwriting_model IAM \
         --wikipedia 20220301.{abbreviation} \
         --cache_dir {str(self.HUGGING_FACE_DATASETS_CACHE)} \
         --canvas_size 1152,64 \
         --min_chars 8 \
         --max_chars 200 \
         --max_lines 1 \
         --max_paragraphs 1 \
         --count {self.end_idx} \
         --resume \
         --no_incrementer
         """

        try:
            from projects.demos.generate_lines import LineGenerator
            lg = LineGenerator(args)
            lg.main()
            # create s3 folder if not exists on next line
            # launch_background_rsync_task(path, f"s3://datascience-computervision-l3apps/HWR/synthetic-data/languages/{language}-lines/v1/")
        except Exception as e:
            print(e)
            print(f"Failed to run {language}")

    def test_downloading_wikipedia_using_huggingface_dataset_for_each_language(self):
        """ Basically just ensures the language exists, does not preprocess it

        Returns:

        """
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


preprocessed = ["en", "fr", "it", "de"]
languages = {
     #"fr": "french",
     #"en": "english",
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


def check_if_done(path, count=1000000):
    # check if path has more than 100000 files
    return len(list(path.glob("*"))) >= count


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


if __name__ == "__main__":
    config = Config()
    for abbreviation, language in languages.items():
        config.run(language, abbreviation)

