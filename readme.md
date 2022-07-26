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


# Install
    
    pip3 install git+ssh://git@github.com/tahlor/pdfgen --upgrade




# Rename Package
    
    find . -type f -name "*" -exec sed -i "s@pdfgen@pdfgen2@g" {} \;
