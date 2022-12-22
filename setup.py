#!/usr/bin/env python
import os
from setuptools import setup

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

def get_requirements():
    """
    Return requirements as list.

    package1==1.0.3
    package2==0.0.5
    """
    with open('requirements.txt') as f:
        packages = []
        for line in f:
            line = line.strip()
            # let's also ignore empty lines and comments
            if not line or line.startswith('#'):
                continue
            if 'https://' in line:
                tail = line.rsplit('/', 1)[1]
                tail = tail.split('#')[0]
                line = tail.replace('@', '==').replace('.git', '')
            packages.append(line)
    return packages

setup(name='docgen',
      version='0.0.30',
      description='docgen',
      long_description= "" if not os.path.isfile("README.md") else read_md('README.md'),
      author='Taylor Archibald',
      author_email='taylor.archibald@byu.edu',
      url='https://github.ancestry.com/tarchibald/docgen',
      setup_requires=['pytest-runner',],
      tests_require=['pytest','python-coveralls'],
      packages=['docgen'],
      install_requires=[
          get_requirements(),
      ],
     )
