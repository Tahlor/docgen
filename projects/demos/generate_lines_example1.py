from projects.demos.generate_lines import LineGenerator

if __name__ == "__main__":
    # ' --output_folder C:\\Users\\tarchibald\\github\\docgen\\projects\\demos\\output --batch_size 16  --freq 1  --saved_handwriting_model IAM --wikipedia 20220301.fr '

    # galois french lines
    # GENERATE NEW HANDWRITING ON THE FLY -- NEEDED FOR FRENCH
    # --output_folder {output}
    args = rf"""
    --batch_size 16 
    --saved_handwriting_model IAM
    --wikipedia 20220301.fr
    --max_lines 1
    --max_chars 10
    --min_chars 3
    --canvas_size 1152,48
    --saved_hw_model_folder /media/data/1TB/datasets/s3/HWR/synthetic-data/python-package-resources/handwriting-models 
    """.replace("\n"," ")

    ocr_format = LineGenerator(args=args).main()

    #testing()

# truncate words
# bad lining up
# last line cutoff
