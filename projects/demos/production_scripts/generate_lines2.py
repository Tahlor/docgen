import os
DEVICE = "0;1"
END=10000000
#os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

import torch
from projects.demos.production_scripts.generate_lines import Config

languages = {
     "en": "english",
}

if __name__ == "__main__":
    config = Config(end_idx=END,
                    device=DEVICE,
                    output_override="/HOST/media/data/1TB/datasets/synthetic",
                    start_idx=6000000)
    for abbreviation, language in languages.items():
        config.run(language, abbreviation)

