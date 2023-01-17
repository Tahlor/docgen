#!/bin/bash

# This installed but had some kind of LIB error on EC2
#sudo amazon-linux-extras install python3.8

if [ -f /usr/bin/python3.8 ]; then
    /usr/bin/python3.8 -m venv venv
    source ./venv/bin/activate
else
    conda create --name docgen python=3.8
    conda activate docgen
fi


if [  "$(dirname $PWD)" -ne docgen ]; then
  cd docgen
fi

pip install --upgrade pip wheel setuptools
pip install -e .

# Test download resources
python ./projects/french_bmd/french_bmd_from_layoutgen.py
