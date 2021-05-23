# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import cv2
import numpy as np

# 21811957 정보미
image = cv2.imread("./testPlate.tif")
Gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray image", Gray)
kernel = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])
p1 = cv2.filter2D(Gray, -1, kernel)
cv2.imshow("prac1_1 image", p1)
kernel2=np.array([[0,0,0],[0,2,0],[0,0,0]])-1/9*np.array([[1,1,1],[1,1,1],[1,1,1]])
p2= cv2.filter2D(Gray, -1, kernel2)
cv2.imshow("prac1_2 image",p2)
# ______________________________________________________________________
Ksize=(3,3)
#origin_sigmaX=0.5
sigmaX=100
blur =(cv2.GaussianBlur(Gray, Ksize, sigmaX))
alpha = 1
Gray = Gray + alpha*(Gray - blur)
cv2.imshow("prac2_sigma=100", Gray)
# ______________________________________________________________________
ret,dstG=cv2.threshold(Gray,127,255,cv2.THRESH_BINARY)
dstA=cv2.adaptiveThreshold(Gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
cv2.imshow("Binarized image_global",dstG)
cv2.imshow("Binarized image_adaptive",dstA)
cv2.waitKey(0)
cv2.destroyAllWindows()

np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])
