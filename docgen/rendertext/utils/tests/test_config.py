from docgen.rendertext.utils.config import *

variation_dict = {"training_dataset": ["dataset1", "dataset2"],
                  "training_option": [True,False],
                  "advanced_options":[{"advanced_option1":"this"},{"advanced_option1":"or that"}]
                  }

baseline_dict = {"training_dataset": ["dataset1"],
                  "training_option":[True]
                 }

def test_config_generation():
    baseline_configs = "./test_config.yaml",
    for config in baseline_configs:
        main(config, variation_dict, baseline_dict)
    print(main)

if __name__=='__main__':
    test_config_generation()

