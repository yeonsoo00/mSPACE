

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

import pickle




def findclosedcell(geneloation,celllocations):

    key, value = min(celllocations.items(), key=lambda kv : math.sqrt((kv[1][1] - geneloation[1])**2+(kv[1][0] - geneloation[0])**2))
    
    return key,math.sqrt((value[1] - geneloation[1])**2+(value[0] - geneloation[0])**2)
    


 




if __name__=="__main__":



    
    

    # groupgene = 
    # allgenename = list(groupgene.keys())
    step1 = False
    step2 = True
    if step1:
        outputdict = {
            "cellid":[],
            "cellx":[],
            "celly":[]
        }

        # genename = "COL2A1"
        # genegroupdf = df[df["gene"]==genename]

        # for index,row in tqdm(genegroupdf.iterrows()):
        #     cv2.circle(img, (row["x"],row["y"]), 1, (0, 255, 0), 1)

        # cv2.imwrite("./09012023-output_wgene/DAPI_nuecial_identifie_"+genename+".tif",img)




        celllocations = {}


        # Read the DAPI image in TIFF format
        image_path = "./09012023/DAPI.tif"  # Replace with your image path
        img = tifffile.imread(image_path)

        low_img = (img >> 8).astype('uint8')


        thresh = 255-cv2.adaptiveThreshold(low_img, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 0)
        cv2.imwrite("./09012023-output_wgene/DAPI_identified_threshold.tif",thresh)

        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        originalmarkers = labels
        cellid = 0
        for i in tqdm(range(0, numLabels)):
            area = stats[i, cv2.CC_STAT_AREA]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            (cX, cY) = centroids[i]
            maxarea = (math.pi*w+w*h-4*w)/(4*h)
            minarea = 1/3
            if area<50 or area>500: 
                originalmarkers[labels==i] = 0
            elif (area/(w*h) ) > maxarea or (area/(w*h) ) < minarea:
                    originalmarkers[labels==i] = 0
            else:
                celllocations[cellid] = (cX,cY)
                outputdict["cellid"].append("Cell "+str(cellid))
                outputdict["cellx"].append(cX)
                outputdict["celly"].append(cY)
                cellid+=1

        try_data = cv2.cvtColor(low_img, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(try_data,originalmarkers)
        try_data[markers == -1] = [0,0,255]
        cv2.imwrite("./09012023-output_wgene/DAPI_nuecial_identified.tif",try_data)
        
        # pd.DataFrame(outputdict).to_csv("./09012023-output_wgene/geneexpression.csv",index=None)

        with open("./09012023-output_wgene/tmp.pkl", 'wb' ) as pklfile:
            pickle.dump([outputdict,celllocations] , pklfile)

    

    if step2:
        with open("./09012023-output_wgene/tmp.pkl", 'rb' ) as pklfile:
            outputdict,celllocations = pickle.load(pklfile)
            


        df = pd.read_csv("./09012023-output/gene_identifided.csv")
        image_path = "./09012023-output_wgene/DAPI_nuecial_identified.tif"  # Replace with your image path
        try_data = cv2.imread(image_path)


        for genename, group in df.groupby("gene"):
            print(genename)
            img = try_data.copy()
            totalareas = [0]*len(celllocations)


            for index,row in tqdm(group.iterrows(), total=group.shape[0]):
                geneloation = (row["cX"],row["cY"])
                longestdist = max(row["W"],row["H"])
                
                clostkey,clostvalue = findclosedcell(geneloation,celllocations)
                if clostvalue<=3*longestdist:
                    totalareas[clostkey] +=row["Aera"]
                    cv2.circle(img, (int(row["x"]),int(row["y"])), 1, (0, 255, 0), 1)
            
            outputdict[genename] = totalareas    

            cv2.imwrite("./09012023-output_wgene/DAPI_nuecial_identifie_"+genename+".tif",img)

        pd.DataFrame(outputdict).to_csv("./09012023-output_wgene/geneexpression.csv",index=None)








    # genevalue = "./09012023-output/gene_identifided.csv"


    # for filename in tqdm(os.listdir('./09012023-preprocessed/')):
    #     if filename.endswith('.tif'):
    #         prename = filename.replace(".tif","")
    #         genename = prename.split("_")[0]
    #         img = cv2.imread("./09012023-preprocessed/"+prename+".tif")
    #         output = img.copy()
    #         gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         
    #         output = cv2.connectedComponentsWithStats(gray_image, 8, cv2.CV_32S)
    #         (numLabels, labels, stats, centroids) = output
    #         originalmarkers = labels
    #         for i in tqdm(range(0, numLabels)):            
    #             area = stats[i, cv2.CC_STAT_AREA]
    #             if area <50 or area > 10000:
    #                 originalmarkers[labels==i] = 0
    #             else:
    #                 x = stats[i, cv2.CC_STAT_LEFT]
    #                 y = stats[i, cv2.CC_STAT_TOP]
    #                 w = stats[i, cv2.CC_STAT_WIDTH]
    #                 h = stats[i, cv2.CC_STAT_HEIGHT]
    #                 (cX, cY) = centroids[i]
    #                 area = stats[i, cv2.CC_STAT_AREA]
    #                 df["Cellid"].append(i)
    #                 df["x"].append(x)
    #                 df["y"].append(y)
    #                 df["gene"].append(genename)
    #                 df["W"].append(w)
    #                 df["H"].append(h)
    #                 df["cX"].append(cX)
    #                 df["cY"].append(cY)
    #                 df["Aera"].append(area)
            
            
    #         markers = cv2.watershed(img,originalmarkers)
    #         img[markers == -1] = [0,0,255]
    #         cv2.imwrite("./09012023-output/"+prename+"_identified.tif",img)
    #         pd.DataFrame(df).to_csv("./09012023-output/"+prename+"_identifided.csv",index=None)