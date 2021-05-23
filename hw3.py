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

# +
# 21811957 정보미

# load an image
img=cv2.imread('pikachu.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow("Gray image",img)
# kernel
prewitt_x_kernel=np.array([[-1, 0, 1],[-1,0,1],[-1,0,1]]) 
sobel_y_kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
prewitt_dst=cv2.filter2D(img,-1,prewitt_x_kernel)
sobel_dst=cv2.filter2D(img,-1,sobel_y_kernel)
# show img
cv2.imshow("prewitt_x_kernel",prewitt_dst)
cv2.imshow("sobel_y_kernel",sobel_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# -

my_kernel=np.array([[-1,2,1],[0,0,0],[1,-2,-1]])
my_dst=cv2.filter2D(img,-1,my_kernel)
cv2.imshow("my_kernel",my_dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

my_kernel

gx=cv2.Sobel(img,-1,1,0,ksize=3)
gy=cv2.Sobel(img,-1,0,1,ksize=3)
gxf=cv2.Sobel(img,cv2.CV_32F,1,0,ksize=3)
gyf=cv2.Sobel(img,cv2.CV_32F,0,1,ksize=3)
# show img
cv2.imshow("default_gx",gx)
cv2.imshow("default_gy",gy)
cv2.imshow("float32_gx",gxf)
cv2.imshow("float32_gy",gyf)
cv2.waitKey(0)
cv2.destroyAllWindows()

# +
gx=np.abs(gx)
gxf=np.abs(gxf)
gx=cv2.normalize(gx,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
gxf=cv2.normalize(gxf,None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

cv2.imshow("normalize_default",gx)
cv2.imshow("normalize_float",gxf)
cv2.waitKey(0)
cv2.destroyAllWindows()

# +
# Gaussian Blur, smoothing effect
#sigma=0
#imgBlur0=cv2.GaussianBlur(img,(7,7),sigma)
#cv2.imshow("img_blur50",imgBlur0)

"""sigma=10
imgBlur10=cv2.GaussianBlur(img,(7,7),sigma)
cv2.imshow("img_blur10",imgBlur10)
sigma=50
cv2.imshow("img_blur50",imgBlur50)"""

# Canny edge detector, th1, th2 effect
th1, th2=100,200
imgCanny=cv2.Canny(img,th1,th2)
cv2.imshow("img_Canny100200",imgCanny)
th1, th2=100,250
imgCanny=cv2.Canny(img,th1,th2)
cv2.imshow("img_Canny100250",imgCanny)
th1, th2=50,200
imgCanny=cv2.Canny(img,th1,th2)
cv2.imshow("img_Canny50200",imgCanny)
th1, th2=150,200
imgCanny=cv2.Canny(img,th1,th2)
cv2.imshow("img_Canny150200",imgCanny)
th1, th2=200,250
imgCanny=cv2.Canny(img,th1,th2)
cv2.imshow("img_Canny200250",imgCanny)
cv2.waitKey(0)
cv2.destroyAllWindows()
