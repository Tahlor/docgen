#!/bin/bash
#
# ./create.sh PACKAGE_NAME GITHUB_USER TOKEN EMAIL

### repo not fully created?
### not pushing: git push --set-upstream origin master

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";
FILE="$(basename "${SCRIPT_DIR}")"
echo $FILE

PACKAGE_NAME=${1:-$FILE}
GITHUB_USER=${2:-tahlor}
TOKEN=${3:-$GITHUB_TOKEN}
EMAIL=${4:-taylor.archibald@byu.edu}
PRIVATE=${5:-true}

read -r -p "Package Name ($PACKAGE_NAME): " PACKAGE_NAME1
PACKAGE_NAME=${PACKAGE_NAME1:-$PACKAGE_NAME}
read -r -p "Github User ($GITHUB_USER): " GITHUB_USER1
GITHUB_USER=${GITHUB_USER1:-$GITHUB_USER}
read -r -p "GitHub Token ($TOKEN): " TOKEN1
TOKEN=${TOKEN1:-$TOKEN}
read -r -p "Email ($EMAIL): " EMAIL1
EMAIL=${EMAIL1:-$EMAIL}

PACKAGE_NAME=`echo "$PACKAGE_NAME" | sed -E -e 's/ /-/g'`
SAFE_PACKAGE_NAME=`echo "$PACKAGE_NAME" | sed -E -e 's/ |-/_/g'`


# Remove old git
rm .git -rf

# Rename files
#package_files=$(ls . |grep .mp4)

for filename in `find . -type 'f,d' -name docx_localization`; 
do
    new_file=`echo "$filename" | sed -e "s/docx_localization/$PACKAGE_NAME/g"`
    if [ "$filename" != "$new_file" ]; then
        mv "$filename" "$new_file"
    fi
done

# find and replace in the files
find . -type f -name "*" -exec sed -i "s@docx_localization@${PACKAGE_NAME}@g" {} \;
find . -type f -name "*" -exec sed -i "s|taylor.archibald@byu.edu|${EMAIL}|g" {} \;
find . -type f -name "*" -exec sed -i "s@tahlor@${GITHUB_USER}@g" {} \;
find . -type f -name "*" -exec sed -i "s@version='0.0.1'

echo $PACKAGE_NAME
echo $SAFE_PACKAGE_NAME
echo $GITHUB_USER
echo $EMAIL
echo $TOKEN


# initialize new git
curl -u $GITHUB_USER:$TOKEN https://api.github.com/user/repos -d "{\"name\":\"$SAFE_PACKAGE_NAME\", \"private\": $PRIVATE}"

git init
git add .
git commit -m "first commit"
git branch -M master
git remote add origin git@github.com:$GITHUB_USER/${SAFE_PACKAGE_NAME}.git
git push -u origin master
#git push --set-upstream origin master

