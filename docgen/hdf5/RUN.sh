#!/bin/bash

# array of latin, french, german, spanish, hugarian
declare -a arr=("latin" "french" "german" "spanish" "hungarian")
for i in "${arr[@]}"  # The quotes are necessary here
do
    echo "Processing $i"
    python add_jsons_to_hdf5.py --hdf5_path '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'.h5' --img_count 5000000 --overwrite
done

python add_jsons_to_hdf5.py --hdf5_path '/media/data/1TB/datasets/synthetic/training_styles/english' '/media/data/1TB/datasets/synthetic/training_styles/english.h5' --img_count 10000000 --overwrite