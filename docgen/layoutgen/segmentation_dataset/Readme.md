import cv2
import numpy as np

# Read the images
src = cv2.imread("source.jpg")  # Source image
dst = cv2.imread("destination.jpg")  # Destination image

# Create a mask of the same size as the source image
src_mask = 255 * np.ones(src.shape, src.dtype)

# The center of the source image will be placed at this location in the destination image.
center = (dst.shape[1]//2, dst.shape[0]//2) # (x, y) = (width/2, height/2)

# Perform seamless cloning
output = cv2.seamlessClone(src, dst, src_mask, center, cv2.NORMAL_CLONE)

# Save result
cv2.imwrite("output.jpg", output)

https://docs.opencv.org/3.4/df/da0/group__photo__clone.html
https://arxiv.org/pdf/2101.11674.pdf
https://drive.google.com/drive/folders/19Xt35IZx8It5rYJXfwevnici3ZxWm_8O