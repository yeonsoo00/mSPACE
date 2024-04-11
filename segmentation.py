

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

for filename in tqdm(os.listdir('./08102023/')):
    if filename.endswith('.tif'):
        prename = filename.replace(".tif","")
        # print(prename)
        # filename = "COL1A2_Max_RB20"
        img = cv2.imread("./08102023/"+prename+".tif")

        inBlack  = np.array([0, 0, 0], dtype=np.float32)
        inWhite  = np.array([2, 2, 2], dtype=np.float32)
        inGamma  = np.array([10, 10.0, 10.0], dtype=np.float32)
        outBlack = np.array([0, 0, 0], dtype=np.float32)
        outWhite = np.array([255, 255, 255], dtype=np.float32)

        img = np.clip( (img - inBlack) / (inWhite - inBlack), 0, 255 )                            
        img = ( img ** (1/inGamma) ) *  (outWhite - outBlack) + outBlack
        img = np.clip( img, 0, 255).astype(np.uint8)
        cv2.imwrite("./output/"+prename+"_auto_color_adjust.tif",img)
        output = img.copy()

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # ret, thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # noise removal
        # kernel = np.ones((3,3),np.uint8)
        # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        # # sure background area
        # sure_bg = cv2.dilate(opening,kernel,iterations=3)
        # # Finding sure foreground area
        # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        # ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
        # # Finding unknown region
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv2.subtract(sure_bg,sure_fg)

        # ret, markers = cv2.connectedComponents(gray_image)
        # markers = cv2.watershed(img,markers)
        # img[markers == -1] = [0,255,0]

        output = cv2.connectedComponentsWithStats(gray_image, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        originalmarkers = labels
        for i in range(0, numLabels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area <50 or area > 300:
                originalmarkers[labels==i] = 0
            

        markers = cv2.watershed(img,originalmarkers)
        img[markers == -1] = [0,0,255]
        cv2.imwrite("./output/"+prename+"_processed.tif",img)
# resize = cv2.resize(img, (0,0), fx=0.4, fy=0.4) 
# window_name = 'image'
# cv2.imshow(window_name,resize)
# cv2.waitKey(0) 
  

# cv2.destroyAllWindows() 

# output = img.copy()
# for i in range(0, numLabels):
#     # if this is the first component then we examine the
#     # *background* (typically we would just ignore this
#     # component in our loop)
#     if i == 0:
#         text = "examining component {}/{} (background)".format(
#             i + 1, numLabels)
#     # otherwise, we are examining an actual connected component
#     else:
#         text = "examining component {}/{}".format( i + 1, numLabels)
#     # print a status message update for the current connected
#     # component
#     print("[INFO] {}".format(text))
#     # extract the connected component statistics and centroid for
#     # the current label
#     x = stats[i, cv2.CC_STAT_LEFT]
#     y = stats[i, cv2.CC_STAT_TOP]
#     w = stats[i, cv2.CC_STAT_WIDTH]
#     h = stats[i, cv2.CC_STAT_HEIGHT]
#     area = stats[i, cv2.CC_STAT_AREA]
#     (cX, cY) = centroids[i]
    
#     # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
#     componentMask = (labels == i).astype("uint8") * 255
#     # show our output image and connected component mask
#     # cv2.imshow("Output", output)
#     resize = cv2.resize(componentMask, (0,0), fx=0.4, fy=0.4) 
#     cv2.imshow("Connected Component", resize)
#     cv2.waitKey(0)

# resize = cv2.resize(img, (0,0), fx=0.4, fy=0.4) 
# window_name = 'image'
# # cv2.imshow(window_name,resize)
# cv2.imwrite('./08102023/COL1A2_Max_RB20_processed.tif',img)

# cv2.waitKey(0) 
  

# cv2.destroyAllWindows() 






# kernel = np.ones((5,5),np.uint8)
# opening = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
# detect circles in the image
# circles = cv2.HoughCircles(gray_image, cv2.HOUGH_GRADIENT, 1000, 100,minRadius=1,maxRadius=100)
# if circles is not None:
# # convert the (x, y) coordinates and radius of the circles to integers
#     circles = np.round(circles[0, :]).astype("int")
#     # loop over the (x, y) coordinates and radius of the circles
#     for (x, y, r) in circles:
#         # draw the circle in the output image, then draw a rectangle
#         # corresponding to the center of the circle
#         cv2.circle(output, (x, y), r, (0, 255, 0), 4)
#         # cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#     # show the output image
#     newimage = np.hstack([img, output])
#     newimage = cv2.resize(newimage,(0,0), fx=0.3, fy=0.3)
#     cv2.imshow("output", newimage)
#     cv2.waitKey(0)

# alpha = 3 # Contrast control (1.0-3.0)
# beta = 100 # Brightness control (0-100)

# manual_result = cv2.convertScaleAbs(opening, alpha=alpha, beta=beta)


# resize = cv2.resize(img, (0,0), fx=0.4, fy=0.4) 
# window_name = 'image'
# cv2.imshow(window_name,resize)

# cv2.waitKey(0) 
  
# # closing all open windows 
# cv2.destroyAllWindows() 

# a = list(gray_image.flatten())

# plt.hist(a , bins=1000)  # arguments are passed to np.histogram
# np.histogram(a,  density=True)
# plt.title("Histogram with 'auto' bins")
# Text(0.5, 1.0, "Histogram with 'auto' bins")
# plt.show()


# # gray_coffee = rgb2gray(imarray)
 
# # Setting the plot size to 15,15
# plt.figure(figsize=(15, 15))
 
# for i in range(10):
   
#   # Iterating different thresholds
#   binarized_gray = (imarray > i*0.1)*1
#   plt.subplot(5,2,i+1)
   
#   # Rounding of the threshold
#   # value to 1 decimal point
#   plt.title("Threshold: >"+str(round(i*0.1,1)))
   
#   # Displaying the binarized image
#   # of various thresholds
#   plt.imshow(binarized_gray, cmap = 'gray')
   
# plt.tight_layout()
# plt.show()