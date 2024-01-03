import socket
from projects.french_bmd.french_bmd_layoutgen import main, parser
import logging
from pathlib import Path
import socket
import yaml
import sys
import docgen

# get the docgen package directory from import
docgen_path = Path(docgen.__file__).parent.parent
CONFIG_FOLDER = docgen_path / "projects/french_bmd/"
print(f"docgen_path: {docgen_path}")

logger = logging.getLogger("root")
logger.setLevel(logging.INFO)

if __name__ == "__main__":

    default_config = CONFIG_FOLDER / "./config/default_2.4.4_leading paragraph marks.yaml"

    if socket.gethostname() == "PW01AYJG":
        args = """
          --config '{config}'
          --count 1000
          --renderer novel
          --output  G:/s3/synthetic_data/FRENCH_BMD/v0{char}
          --wikipedia 20220301.fr
          --saved_hw_model IAM
          --hw_batch_size 8
          --workers 0
          --daemon_buffer_size 500
          --render_gt_layout
          --special_char_to_begin_paragraph_generator {dataset_path}
          --special_char_to_begin_paragraph_generator_width_multiplier {multiplier}
        """
        #           --use_hw_and_font_generator

    elif socket.gethostname() == "Galois":
        args = """
          --config '{config}'
          --start 53000
          --count 100000
          --renderer novel
          --output /media/EVO970/data/synthetic/french_bmd/ 
          --saved_hw_model_folder /media/data/1TB/datasets/s3/HWR/synthetic-data/python-package-resources/handwriting-models 
          --wikipedia 20220301.fr
          --saved_hw_model IAM
          --hw_batch_size 80    
          --workers 0
        """
    else:
        args = None

    multiplier_3 = ["X","-","l"]

    for dataset_path_str in ["C:/Users/tarchibald/github/docgen/projects/demos/archive/output/Le",
                         "C:/Users/tarchibald/github/docgen/projects/demos/archive/output/X",
                         "C:/Users/tarchibald/github/docgen/projects/demos/archive/output/-",
                         "C:/Users/tarchibald/github/docgen/projects/demos/archive/output/l",
                         "C:/Users/tarchibald/github/docgen/projects/demos/archive/output/Lan",
                         "C:/Users/tarchibald/github/docgen/projects/demos/archive/output"][-2::-1]:
        dataset_path = Path(dataset_path_str)
        opts = parser(args.format(dataset_path=dataset_path_str, char=dataset_path.name, config=default_config
                                  , multiplier=3 if dataset_path.name in multiplier_3 else 1))
        logger.info(f"OUTPUT: {opts.output}")
        print(opts)
        main(opts)