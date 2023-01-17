#!/bin/bash

/usr/bin/python3.8 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -e .
#pip install git+ssh://git@github.ancestry.com/tarchibald/docgen@galois#egg=docgen


# Test download resources
python ./docgen/projects/french_bmd/french_bmd_from_layoutgen.py
