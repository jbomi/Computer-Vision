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

# +
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT=10

img1=cv2.imread('photo2.jpg') # query img
img2=cv2.imread('photo1.jpg') # train img

# initiate SIFT detector
sift=cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE=0
index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
search_params=dict(checks=50)
print('index_params: ',index_params,'search_params: ',search_params)

flann=cv2.FlannBasedMatcher(index_params, search_params)

matches=flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good=[]
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good.append(m)
        
if len(good)>MIN_MATCH_COUNT:
    # -1: auto size
    src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    M, mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,4.3)
    #M, mask=cv2. findHomography(src_pts,dst_pts)
    print('M: ',M)
    
    matchesMask=mask.ravel().tolist()
    
    h,w,c=img1.shape
    print('h: ',h,' w: ',w,' c :',c)
    pts=np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst=cv2.perspectiveTransform(pts,M)
    
    img2=cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)
    
else:
    print("Not enough matches are found - %d/%d "%(len(good),MIN_MATCH_COUNT))
    matchesMask=None
    
draw_params=dict(matchColor=(255,0,0),# draw matches in red color
                singlePointColor=None, 
                matchesMask=matchesMask, # draw only inliner
                flags=2)
img3=cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3,'gray'),plt.show()

width=(img2.shape[1]+img1.shape[1])
height=img2.shape[0]+img1.shape[0]

dst=cv2.warpPerspective(img1,M,(width,height))
print('Warping right to left')
#cv2.imshow("Warping right to left",dst)
plt.imshow(dst,'gray'),plt.show()

dst[0:img2.shape[0],0:img2.shape[1]]=img2

#cv2.namedWindow('Stiching',cv2.WINDOW_AUTOSIZE)
#cv2.resizeWindow('Stiching',750,750)

print('stiching')
cv2.imshow("Stiching",dst)
plt.imshow(dst,'gray'),plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()
# -


