from docgen.rendertext.utils import img_f
import random
import numpy as np

# put work images into paragraph form (single image)
def word_imgs_to_paragraph(word_imgs, height, space_width, boundary_x):
    # word_imgs: list of (text,word_img)
    # height: the height to resize word images to
    # space_width: how much space between words
    # boundary_x: when to stop horizontally and wrap to new line

    full_text = ''
    newline = round(height * 0.1 + 0.9 * random.random() * height)
    max_x = 0
    max_y = 0
    cur_x = 0
    cur_y = 0
    resized_words = []

    text, img = word_imgs[0]
    width = max(1, round(img.shape[1] * height / img.shape[0]))
    img = img_f.resize(img, (height, width))
    resized_words.append((img, cur_x, cur_y))
    full_text += text
    cur_x += width
    max_x = max(max_x, cur_x)
    max_y = max(max_y, cur_y + height)

    for text, img in word_imgs[1:]:
        width = max(1, round(img.shape[1] * height / img.shape[0]))
        img = img_f.resize(img, (height, width))
        if cur_x + width < boundary_x:
            full_text += ' ' + text
        else:
            cur_x = 0
            cur_y += newline + height
            full_text += '\\' + text
        resized_words.append((img, cur_x, cur_y))
        cur_x += space_width + width
        max_x = max(max_x, cur_x)
        max_y = max(max_y, cur_y + height)

    full_img = np.zeros([max_y, max_x], dtype=np.uint8)
    for img, x, y in resized_words:
        full_img[y:y + img.shape[0], x:x + img.shape[1]] = img

    return full_img, full_text
