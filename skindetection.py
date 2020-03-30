from pyimagesearch import imutils
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import os.path
from PIL import Image

frame= cv2.imread("./image.png")
#plt.imshow(frame)
#plt.show()

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

frame = imutils.resize(frame, width = 400)
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
skinMask = cv2.inRange(converted, lower, upper)
skinMask = 255 - skinMask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
skinMask = cv2.erode(skinMask, kernel, iterations = 5)
skinMask = cv2.dilate(skinMask, kernel, iterations = 5)

frame[skinMask != 255] = 255

plt.figure(figsize=(15,20))
#plt.imshow(frame)

#plt.pause(3)

IMG = frame.tolist()
#print(IMG)
HEIGHT = len( IMG )
WIDTH = len( IMG[0] )

########################################################################
#LIST TO RGB_LIST
RGB_LIST=[]
for pixels in IMG :
    for RGB in pixels:
        #print(RGB)
        set = ( RGB[0] , RGB[1] , RGB[2] )
        RGB_LIST.append( set )

########################################################################
#LIST TO IMGAGE DATA CONVERSION
img = Image.new( 'RGB' , ( WIDTH , HEIGHT ) )
img.putdata( RGB_LIST )

########################################################################
"""
#IMAGE RESIZING
size = (  ,  )
img = img.resize( size )
"""
########################################################################
#FILE SAVING PART
filename = "image_1"
extension = ".png"
img.save( filename + '.png' )
print( "\nFILE_NUM : " , filename + '.png' , "\n" )

########################################################################
