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

# !git clone https://github.com/opencv/opencv.git

# +
import os

os.getcwd()

# +
import cv2
from matplotlib import pyplot as plt
#Load the cascade
face_cascade=cv2.CascadeClassifier('./opencv/data/haarcascades/haarcascade_frontalface_default.xml')
#print(face_cascade)

# Read the input img and Convert into grayscale
#img=cv2.imread('bnana.jpg')
img=cv2.imread('me.jpg')
#print('img',img)
print('origin img')
plt.imshow(img,'gray'),plt.show()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print('gray',gray)

a=1.4
b=0

# Detect faces
faces=face_cascade.detectMultiScale(gray,a,b)
print('a: ',a,' b: ',b)
print('faces',faces)
# Draw rectangle around the faces
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# Display the output
cv2.imshow('img',img)
plt.imshow(img,'gray'),plt.show()
cv2.waitKey()
cv2.destroyAllWindows()
# -

len(faces)

# +
import numpy as np
import cv2 
from matplotlib import pyplot as plt
#Load the cascade
face_cascade=cv2.CascadeClassifier('./opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('./opencv/data/haarcascades/haarcascade_eye.xml')
print(eye_cascade)

# Read the input img and Convert into grayscale
img=cv2.imread('me.jpg')
#print('img',img)
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#print('gray',gray)
print('origin img')
plt.imshow(img,'gray'),plt.show()
a=1.2 # a is scaleFactor
b=3 # b is minNeighbors
# Detect faces
faces=face_cascade.detectMultiScale(gray,a,b)
eyes=eye_cascade.detectMultiScale(gray,a,b)

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
# Draw rectangle around the faces
for (x,y,w,h) in eyes:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
print('a: ',a,' b: ',b)
print('faces: ',faces)
print('eyes: ',eyes)
print('x: ',x,' y: ',y,' x+w: ',x+w,' y+h: ',y+h)
plt.imshow(img,'gray'),plt.show()
# Display the output
#cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
# -

# ## GT

# +
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# 얼굴과 눈을 검출하기 위해 미리 학습시켜 놓은 XML 포맷으로 저장된 분류기를 로드합니다. 
face_cascade = cv.CascadeClassifier('./opencv/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('./opencv/data/haarcascades/haarcascade_eye.xml')


# 얼굴과 눈을 검출할 그레이스케일 이미지를 준비해놓습니다. 
img = cv.imread('me.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

a=1.1
b=4
# 이미지에서 얼굴을 검출합니다. 
faces = face_cascade.detectMultiScale(gray, a, b)


# 얼굴이 검출되었다면 얼굴 위치에 대한 좌표 정보를 리턴받습니다. 
for (x,y,w,h) in faces:

    # 원본 이미지에 얼굴의 위치를 표시합니다. 
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    # 눈 검출은 얼굴이 검출된 영역 내부에서만 진행하기 위해 ROI를 생성합니다. 
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    # 눈을 검출합니다. 
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # 눈이 검출되었다면 눈 위치에 대한 좌표 정보를 리턴받습니다. 
    for (ex,ey,ew,eh) in eyes:
        # 원본 이미지에 얼굴의 위치를 표시합니다. ROI에 표시하면 원본 이미지에도 표시됩니다. 
        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


# 얼굴과 눈 검출 결과를 화면에 보여줍니다.
#cv.imshow('img',img)
print('a: ',a,' b: ',b)
plt.imshow(img,'gray'),plt.show()
cv.waitKey(0)

cv.destroyAllWindows()
