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

import numpy as np
import cv2
from matplotlib import pyplot as plt

print(cv2.__version__)

# +
#i=1
#name='me'+str(i)+'.jpg'
#print(name)
img=[]
img=cv2.imread(name,0)
#print(img)
#print('me'+str(i)+'.jpg')
img=[]
for i in range(4):
    name='me'+str(i)+'.jpg'
    img[i]=(cv2.imread(name,0))
    print(img[i])
    
img
# -

# ## library 불러오기

import numpy as np
import cv2
from matplotlib import pyplot as plt 

# ## main code

# +
img1=cv2.imread('peng1.jpg',0) #queryImg, 0-> gray
img2=cv2.imread('pengnothers.jpg',0) # train img
#cv2.imshow("Query img",img1)
#cv2.imshow("Reference img",img2)
'''21811957 정보미'''
# create SIFT feature extractor object
sift=cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)
# brute force matching
bf=cv2.BFMatcher()
matches=bf.knnMatch(des1,des2,k=2) #k=매칭할 근접 이웃 개수
i=1
good=[]
ratio=0.8

for m,n in matches:
    if m.distance < ratio*n.distance:
        good.append([m])
        print(i,'번째')
        print('m.distance:',m.distance)
        print('n.distance:',n.distance)
        print(str(ratio)+'*n.distance=',0.65*n.distance)
        print(good)
        i=i+1

img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv2.imshow("matching",img3)
plt.imshow(img3),plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# -

sift

bf

# +
#len(kp1)
#len(des1)
# -

matches

m.distance

(n.distance)*0.

img1=cv2.imread('peng1.jpg',0) #queryImg 0-> gray
img2=cv2.imread('peng2.jpg',0)

sift=cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)

bf=cv2.BFMatcher()
matches=bf.knnMatch(des1,des2,k=2)

good=[]
for m,n in matches:
    if m.distance < 0.1*n.distance:
        good.append([m])
        print(good)

img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

cv2.imshow("matching",img3)
cv2.waitKey(0)
cv2.destoryAllWindows()


