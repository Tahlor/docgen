#!/bin/bash

program="./add_jsons_to_hdf5.py"
program="./add_jsons_to_hdf5_from_npy.py"

# array of latin, french, german, spanish, hugarian
declare -a arr=("latin" "french" "german" "spanish" "hungarian")
for i in "${arr[@]}"
do
    echo "Processing $i"
    # if program contains "npy" in it
    if [[ $program == *"npy"* ]]; then
        python $program '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'_labels' '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'.h5' --overwrite
    else
        python $program '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'.h5' --img_count 5000000 --overwrite
    fi

done

#python $program '/media/data/1TB/datasets/synthetic/NEW_VERSION/english' '/media/data/1TB/datasets/synthetic/NEW_VERSION/english.h5' --img_count 10000000 --overwrite


