import os
DEVICE = "0;1"
DEVICE = "0"
END=5000000
#os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE

import torch
from projects.demos.production_scripts.generate_line_manager import Config

languages = {
     "fr": "french",
     "de": "german",
     "la": "latin",
     "hu": "hungarian",
     #"es": "spanish",
}

if __name__ == "__main__":
    config = Config(end_idx=END,
                    device=DEVICE,
                    output_override="",
                    output_subfolder="NEW_VERSION",
                    start_idx=-1,
                    style_data_split="all"
                    )
    for abbreviation, language in languages.items():
        config.run(language, abbreviation)
