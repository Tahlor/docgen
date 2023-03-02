from projects.demos.generate_lines import LineGenerator

args =  """ --output_folder ./outputs/output --batch_size 16  \
 --freq 1 \
 --saved_handwriting_model IAM \
  --wikipedia 20220301.fr \
 --canvas_size 1152,64 \
 --min_chars 8 \
 --max_chars 200 \
 --max_lines 1 \
 --max_paragraphs 1 \
 --count 100
 """

lg = LineGenerator(args)
lg.main()