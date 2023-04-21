#!/bin/bash

program="./add_jsons_to_hdf5.py"

# array of latin, french, german, spanish, hugarian
#declare -a arr=("latin" "french" "german" "spanish" "hungarian")
#declare -a arr=("french")
declare -a arr=("french" "spanish" "hungarian")
for i in "${arr[@]}"
do
    echo "Processing $i"
    python $program '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i \
                    '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'.h5'  \
                     --overwrite \
                     --npy_folder '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'_labels'
done

#python $program '/media/data/1TB/datasets/synthetic/NEW_VERSION/english' '/media/data/1TB/datasets/synthetic/NEW_VERSION/english.h5' --img_count 10000000 --overwrite