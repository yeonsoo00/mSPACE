


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

import cv2
import numpy as np
import tifffile


image_path = "./09012023-output_wgene/DAPI_nuecial_identified.tif"  # Replace with your image path
img = cv2.imread(image_path)

df = pd.read_csv("./09012023-output/gene_identifided.csv")
genename = "COL2A1"
genegroupdf = df[df["gene"]==genename]

for index,row in tqdm(genegroupdf.iterrows()):
    cv2.circle(img, (row["x"],row["y"]), 1, (0, 255, 0), 1)

cv2.imwrite("./09012023-output_wgene/DAPI_nuecial_identifie_"+genename+".tif",img)



