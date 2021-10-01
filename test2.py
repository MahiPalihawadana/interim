# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 21:52:58 2021

@author: Dilki palihawadana
"""


import cv2 
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('o.jpg')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#applying median filter for Salt and pepper/impulse noise
filter1 = cv2.medianBlur(gray,5)

#applying gaussian blur to smoothen out the image edges
filter2 = cv2.GaussianBlur(filter1,(5,5),0)

#applying non-localized means for final Denoising of the image
dst = cv2.fastNlMeansDenoising(filter2,None,17,9,17)

#converting the image to binarized form using adaptive thresholding
th1 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

#ret3,th1= cv2.threshold(filter2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('check_median5_gaussian5_mean_adaptive.jpg', th1)