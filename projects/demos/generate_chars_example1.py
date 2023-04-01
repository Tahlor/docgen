from projects.demos.generate_lines import LineGenerator

if __name__ == "__main__":
    args = rf"""
    --batch_size 16 
    --saved_handwriting_model IAM
    --unigram_list '-'
    --max_lines 1
    --max_chars 1
    --min_chars 1
    --canvas_size 48,48
    --saved_handwriting_model "IAM"
    --autocrop
    --count 1000
    --save_frequency 1000
    --iterations_before_new_style 1
    """.replace("\n"," ")

    # --saved_hw_model_folder /media/data/1TB/datasets/s3/HWR/synthetic-data/python-package-resources/handwriting-models


    ocr_format = LineGenerator(args=args).main()
