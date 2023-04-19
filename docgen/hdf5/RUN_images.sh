#!/bin/bash

# array of latin, french, german, spanish, hugarian
declare -a arr=("latin" "french" "german" "spanish" "hungarian")
for i in "${arr[@]}"
do
    echo "Processing $i"
    python ./add_images_to_hdf5.py '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'.h5' --img_count 5000000
done

python ./add_images_to_hdf5.py '/media/data/1TB/datasets/synthetic/training_styles/english' '/media/data/1TB/datasets/synthetic/training_styles/english.h5' --img_count 10000000