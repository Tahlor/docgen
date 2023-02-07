# tried to repurpose paragraph generation to generate lines, needs improvement
python generate_lines.py --output_folder ./output --batch_size 16  \
--freq 1 \
--saved_handwriting_model IAM \
 --wikipedia 20220301.fr \
--canvas_size (1152, 64) \
--min_words 12 \
--max_words 16 \
--max_lines 1 \
--max_paragraphs 1