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

## degradation
## docx2pdf
Docx can be created programmatically from Python, and then converted into a PDF with localization information preserved.

docx2pdf include in this module is based on the publicly available module, with small chanages to accommodate Linux / LibreOffice. 

## layoutgen

LayoutGen is pure Python to create something that resembles the French BMD documents (handwritten header, paragraphs, margins).

## rendertext


## reportlab_tools

Reportlab is a module that is used to create PDF documents from Python.

# Install
    
    pip3 install git+ssh://git@github.com/tahlor/pdfgen --upgrade

# Rename Package
    
    find . -type f -name "*" -exec sed -i "s@pdfgen@pdfgen2@g" {} \;
