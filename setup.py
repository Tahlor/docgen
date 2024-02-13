#!/usr/bin/env python
import os
from setuptools import setup, find_packages #, develop
import warnings
import argparse
from pip._internal import main

# Using --local will look in ".." to see if a package folder is there, and install it from there
# BUT it won't install it with the `pip install -e ` command.
# https://stackoverflow.com/questions/18725137/how-to-obtain-arguments-passed-to-setup-py-from-pip-with-install-option

#print(find_packages())

try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

develop_packages = []
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--local", action="store_true", help="use local file")
args, unknown = parser.parse_known_args()
print(args)

def get_requirements(path="requirements.txt"):
    """
    Return requirements as list.

    package1==1.0.3
    package2==0.0.5
    """
    with open(path) as f:
        packages = []
        for line in f:
            develop_package=False
            package = line.strip()
            print("PARSING PACKAGE", package)
            # let's also ignore empty lines and comments
            if not package or package.startswith('#'):
                continue
            if package.startswith('-r'):
                # recursive requirements
                packages.extend(get_requirements(package.split(' ',1)[1]))
            elif package.startswith('-e'):
                # -e does not work in pip
                # Right now this just ignores the -e
                # editable requirements
                develop_package = True
                package = package.split(' ',1)[1].strip()
            elif package.startswith('--'):
                continue
            if package.startswith("https"):
                package = "synthetictextgen @ " + package


            if package.startswith('.') or package.startswith('/'):
                package_name = os.path.basename(package)
                package = f"{package_name} @ file://localhost" + os.path.abspath(package)

            if "file://." in package: # this won't work with pip -r
                pre, url = package.split("file://")
                url = os.path.abspath(url)
                package = pre + "file://localhost" + url


            if  args.local or develop_package:
                if "file://." in package: # it's already a local copy
                    develop.packages.append(package)
                elif "@" in package: # look for local copy based on the name before the @
                    pre, post = package.split("@",1)
                    local_path = "../" + pre.strip()
                    if os.path.exists(local_path):
                        warnings.warn(f"Using local version of {pre} ({local_path})")
                        package = local_path
                        develop_packages.append(package)
                        continue
                    else:
                        print(f"Not able to find a local version of {package}")

            packages.append(package)
    print(f"PACKAGES: {packages}")
    return packages

setup(name='docgen',
      version='0.1.131',
      description='docgen',
      long_description= "" if not os.path.isfile("README.md") else read_md('README.md'),
      author='Taylor Archibald',
      author_email='taylor.archibald@byu.edu',
      url='https://github.com/tahlor/docgen',
      setup_requires=['pytest-runner',],
      tests_require=['pytest','python-coveralls'],
      packages=[*find_packages()],
      install_requires=[
          get_requirements(),
      ],
     )

for d in develop_packages:
    print(f"Installing {d} in develop mode")
    main(['install', '-e', d])
