#!/bin/bash

program="./add_jsons_to_hdf5.py"

# array of latin, french, german, spanish, hugarian
#declare -a arr=("latin" "french" "german" "spanish" "hungarian")
declare -a arr=("german")
#declare -a arr=("french" "spanish" "hungarian")
for i in "${arr[@]}"
do
    echo "Processing $i"
    python $program '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i \
                    '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'.h5'  \
                     --overwrite \
                     --npy_folder '/media/data/1TB/datasets/synthetic/NEW_VERSION/'$i'_labels'
done

<<<<<<< HEAD
#python $program '/media/data/1TB/datasets/synthetic/NEW_VERSION/english' '/media/data/1TB/datasets/synthetic/NEW_VERSION/english.h5' --img_count 10000000 --overwrite
=======
python add_jsons_to_hdf5.py '/media/data/1TB/datasets/synthetic/training_styles/english' '/media/data/1TB/datasets/synthetic/training_styles/english.h5' \
--npy_folder '/media/data/1TB/datasets/synthetic/NEW_VERSION/english_labels/' --img_count 10000000
>>>>>>> 44ac9bd8b409d20746b7e556aadff41a9822ee8e
