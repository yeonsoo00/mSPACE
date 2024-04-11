

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









totaldf = pd.DataFrame()


for filename in tqdm(os.listdir('./09012023-output/')):
    if filename.endswith('.csv'):
        
        df = pd.read_csv("./09012023-output/"+filename)
        totaldf = pd.concat([totaldf,df])

totaldf.to_csv("./09012023-output/gene_identifided.csv",index=None)