import torch
import os
from projects.demos.production_scripts.generate_lines import Config
DEVICE = "0;1"
END=10000000
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

languages = {
     "en": "english",
}

if __name__ == "__main__":
    config = Config()
    for abbreviation, language in languages.items():
        config.run(language, abbreviation)

