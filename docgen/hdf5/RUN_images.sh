#!/bin/bash

# array of latin, french, german, spanish, hugarian, english
#declare -a arr=( "french" "german" "spanish" "hungarian")
declare -a arr=( "german" )
for i in "${arr[@]}"
do
    echo "Processing $i"
    python ./add_images_to_hdf5.py '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'.h5' --one_dataset --img_count 5000000
done

python ./add_images_to_hdf5.py '/media/data/1TB/datasets/synthetic/NEW_VERSION/english' '/media/data/1TB/datasets/synthetic/NEW_VERSION/english.h5' --img_count 10000000 --one_dataset