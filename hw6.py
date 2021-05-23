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
import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread('road1.jpg',0) # read as a gray image
print(img)
cv2.imshow('input',img)

rows,cols=img.shape

# perspective transform

#pts1=np.float32([[321,655],[727,655],[141,981],[933,981]])
#pts1=np.float32([[390,655],[700,655],[200,980],[990,980]])
#pts1=np.float32([[390,655],[200,900],[700,655],[990,980]])
#pts1=np.float32([[390,655],[700,655],[140,980],[930,980]])
#pts1=np.float32([[320,655],[720,655],[140,980],[930,980]])
#pts1=np.float32([[200,800],[780,800],[200,1200],[800,1200]])
#pts1=np.float32([[321,655],[727,655],[141,981],[933,981]])
pts1=np.float32([[333,581],[823,581],[15,783],[1077,783]])
pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])

cv2.circle(img, (333,581), 20, (255,0,0),-1)
cv2.circle(img, (823,581), 20, (0,255,0),-1)
cv2.circle(img, (15,783), 20, (0,0,255),-1)
cv2.circle(img, (1077,783), 20, (0,0,0),-1)

M=cv2.getPerspectiveTransform(pts1,pts2)
dst=cv2.warpPerspective(img,M,(300,300))

plt.subplot(121),plt.imshow(img),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Output')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
# -


