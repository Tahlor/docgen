#!/usr/bin/env python

from setuptools import setup
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

import os

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
print(requirements)

setup(name='pdfgen',
      version='0.0.9',
      description='pdfgen',
      long_description= "" if not os.path.isfile("README.md") else read_md('README.md'),
      author='Taylor Archibald',
      author_email='taylor.archibald@byu.edu',
      url='https://github.com/tahlor/pdfgen',
      setup_requires=['pytest-runner',],
      tests_require=['pytest','python-coveralls'],
      packages=['pdfgen'],
      install_requires=[
          *requirements,
      ],
     )
