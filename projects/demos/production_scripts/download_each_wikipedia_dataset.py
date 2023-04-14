from datasets import load_dataset
from pathlib import Path
import datetime
import argparse

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
preprocessed = ["en", "fr", "it", "de"]

def download(language_abbreviation, date="20220301"):
    if not language_abbreviation in preprocessed:
        current_year = str(datetime.datetime.now().year)
        date = date.replace("2022", current_year)
        dataset = load_dataset("wikipedia", date=date, language=language_abbreviation, beam_runner="DirectRunner",
                               cache_dir=cache_dir)
    else:
        dataset = load_dataset("wikipedia", f"{date}.{language_abbreviation}", cache_dir=cache_dir)

if __name__=='__main__':
    cache_dir = "/media/data/1TB/datasets/synthetic/huggingface/datasets"
    for abbreviation,language in languages.items():
        download(abbreviation)
