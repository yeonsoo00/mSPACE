

from skimage import data
from skimage import filters
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
# Sample Image of scikit-image package
from PIL import Image
import numpy
import cv2
# Save image in set directory
# Read RGB image
# im = Image.open('./08102023/MAX_PRG4_altsub_RB50.tif')
# # im.show()
# imarray = numpy.array(im)
from tqdm import tqdm 
import os
import pandas as pd

import cv2
import numpy as np
import math

# import cv2
import numpy as np
import tifffile

# Read the DAPI image in TIFF format
image_path = "./09012023/DAPI.tif"  # Replace with your image path
img = tifffile.imread(image_path)

low_img = (img >> 8).astype('uint8')

# cv2.imwrite("./09012023-output/DAPI_identified_threshold.tif",dapi_image)
# Convert the image to grayscale
# gray = cv2.cvtColor(low_img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale if the image is in RGB

# Apply Gaussian blur to reduce noise
# blur = cv2.GaussianBlur(dapi_image, (5, 5), 0)

# Perform thresholding to obtain a binary image
thresh = 255-cv2.adaptiveThreshold(low_img, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 0)
cv2.imwrite("./09012023-output/DAPI_identified_threshold_3.tif",thresh)

# minx = 200
# maxx = 250
# miny = 55
# maxy = 56


# Count overlapping circles
single_area = 100
overlapping = 0

output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
originalmarkers = labels
for i in tqdm(range(0, numLabels)):
    area = stats[i, cv2.CC_STAT_AREA]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    if area<50 or area>500: 
        originalmarkers[labels==i] = 0
    else:
        maxarea = (math.pi*w+w*h-4*w)/(4*h)
        minarea = 1/3
        if (area/(w*h) ) > maxarea or (area/(w*h) ) < minarea:
            originalmarkers[labels==i] = 0


try_data = cv2.cvtColor(low_img, cv2.COLOR_GRAY2BGR)
markers = cv2.watershed(try_data,originalmarkers)
try_data[markers == -1] = [0,0,255]
cv2.imwrite("./09012023-output/DAPI_nuecial_identified_3.tif",try_data)


