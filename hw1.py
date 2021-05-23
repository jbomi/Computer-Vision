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

# 1. Load library

import cv2

# 2. show my info

print('21811957')
print('정보미')

# 3. Load an image and display

image=cv2.imread("./cat2.jpg")
cv2.imshow("test image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
