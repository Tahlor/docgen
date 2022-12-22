# pdfgen

If using Linux, install libreoffice:

    sudo apt install libreoffice-writer
    
Or:
    version="7.3.4"
    cd /usr/local
    wget https://mirror.cyberbits.asia/tdf/libreoffice/stable/${version}/rpm/x86_64/LibreOffice_${version}_Linux_x86-64_rpm.tar.gz
    tar -xvzf LibreOffice_${version}_Linux_x86-64_rpm.tar.gz
    cd LibreOffice*/RPMS
    yum localinstall *.rpm --skip-broken
    yum install cairo
    yum install cups
    export PATH="$PATH:/opt/libreoffice5.0/program"
    sudo yum install libXinerama.x86_64 cups-libs dbus-glib

## TODO LIST
* finalize degradation
* update output format
* fix signature section
* make a version that mimics what Siteng wants better

## Dependencies
Requires:

* [**hwgen**](https://github.ancestry.com/tarchibald/hwgen): Uses transformers to generate synthetic handwriting
  git+https://github.ancestry.com/tarchibald/hwgen
    
* [**textgen**](https://github.ancestry.com/tarchibald/textgen): Package with natural language text to sample from (e.g. Wikipedia)
  git+https://github.ancestry.com/tarchibald/textgen

* [**Faker**](https://github.ancestry.com/tarchibald/Faker): Package for generating fake user data (names, birthdates, addresses, etc.) for many different locales
  git+https://github.ancestry.com/tarchibald/Faker

* [**docdegrade**](https://github.ancestry.com/tarchibald/docdegrade): For artificially degrading synthetic documents
  git+https://github.ancestry.com/tarchibald/docdegrade

* [**download_resources**](https://github.ancestry.com/tarchibald/download_resources): For downloading databases required by packages
  git+https://github.ancestry.com/tarchibald/download_resources

# Install
    
    pip3 install git+ssh://git@github.ancestry.com/tarchibald/docgen --upgrade

# Usage

Example projects for `french_bmd`, `census`, etc. can be found under `docgen/projects`.

# Submodules
## [**content**](https://github.ancestry.com/Tahlor/docgen/tree/master/docgen/content)

Pulls data from Faker to generate fake tabular data.


## [**docx2pdf**](https://github.ancestry.com/Tahlor/docgen/tree/master/docgen/docx2pdf)
Docx can be created programmatically from Python, and then converted into a PDF with localization information preserved.

docx2pdf include in this module is based on the publicly available module, with small chanages to accommodate Linux / LibreOffice. 

The most important function is `fill_area_with_words`, which takes a bounding box, list of images, and a list of words and 
fills the bounding box with the word images.

## [**layoutgen**](https://github.ancestry.com/Tahlor/docgen/tree/master/docgen/layoutgen)

LayoutGen is pure Python to create something that resembles the French BMD documents (handwritten header, paragraphs, margins).

## [**rendertext**](https://github.ancestry.com/Tahlor/docgen/tree/master/docgen/rendertext)
rendertext is a module that renders OCR fonts as images.

## [**reportlab_tools**](https://github.ancestry.com/Tahlor/docgen/tree/master/docgen/reportlab_tools)

Reportlab is a module that is used to create PDF documents from Python.



# Rename Package 
    find . -type f -name "*" -exec sed -i "s@docgen@docgen2@g" {} \;
    find . -type f -name "*" -exec sed -i "s@tarchibald@tarchibald@g" {} \;
    find . -type f -name "*" -exec sed -i "s@github.ancestry.com@github.ancestry.com@g" {} \;
