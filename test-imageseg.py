

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
import tifffile
import pandas as pd
import math
for filename in os.listdir('./09012023-preprocessed/'):
    if filename.endswith('.tif'):
        prename = filename.replace(".tif","")
        print(prename)
        # filename = "COL1A2_Max_RB20"
        genename = prename.split("_")[0]
        # genename = prename


        image_path = "./09012023-preprocessed/"+prename+".tif"  # Replace with your image path
        img = cv2.imread(image_path)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # low_img = (img >> 8).astype('uint8')


        # thresh = 255-cv2.adaptiveThreshold(low_img, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 0)
        # cv2.imwrite("./09012023-output/"+prename+"_threshold.tif",thresh)





        # img = cv2.imread("./09012023/"+prename+".tif")

        # inBlack  = np.array([0, 0, 0], dtype=np.float32)
        # inWhite  = np.array([2, 2, 2], dtype=np.float32)
        # inGamma  = np.array([10, 10.0, 10.0], dtype=np.float32)
        # outBlack = np.array([0, 0, 0], dtype=np.float32)
        # outWhite = np.array([255, 255, 255], dtype=np.float32)

        # img = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 255 )                            
        # img = ( img ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
        # img = np.clip( img, 0, 255).astype(np.uint8)
        # cv2.imwrite("./output/"+prename+"_auto_color_adjust.tif",img)
        # output = img.copy()

        # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        df = {
            "Cellid":[],
            "x":[],
            "y":[],
            "gene":[],
            "W":[],
            "H":[],
            "cX":[],
            "cY":[],
            "Aera":[]
        }

        
        output = cv2.connectedComponentsWithStats(gray_image, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        originalmarkers = labels
        for i in tqdm(range(0, numLabels)):           
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            maxarea = (math.pi*w+w*h-4*w)/(4*h)
            minarea = 1/3
            if area <10 or area > 1000:
                originalmarkers[labels==i] = 0
            elif (area/(w*h) ) > maxarea or (area/(w*h)) < minarea:
                    originalmarkers[labels==i] = 0
            else:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                
                (cX, cY) = centroids[i]
                area = stats[i, cv2.CC_STAT_AREA]
                df["Cellid"].append(i)
                df["x"].append(x)
                df["y"].append(y)
                df["gene"].append(genename)
                df["W"].append(w)
                df["H"].append(h)
                df["cX"].append(cX)
                df["cY"].append(cY)
                df["Aera"].append(area)        


        markers = cv2.watershed(img,originalmarkers)
        img[markers == -1] = [0,0,255]
        cv2.imwrite("./09012023-output/"+prename+"_identified.tif",img)
        pd.DataFrame(df).to_csv("./09012023-output/"+prename+"_identifided.csv",index=None)

totaldf = pd.DataFrame()


for filename in tqdm(os.listdir('./09012023-output/')):
    if filename.endswith('.csv'):
        
        df = pd.read_csv("./09012023-output/"+filename)
        totaldf = pd.concat([totaldf,df])

totaldf.to_csv("./09012023-output/gene_identifided.csv",index=None)


# os.system("shutdown /s")