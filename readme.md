# docx_localization

To create a new repo, copy this repo to a new folder with the name of the new repo and run `initialize.sh`.

# Install
pip3 install git+ssh://git@github.com/tahlor/docx_localization --upgrade




# Replace
find . -type f -name "*" -exec sed -i "s@docx_localization@docx_localization2@g" {} \;
