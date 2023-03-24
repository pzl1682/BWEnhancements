# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 19:54:05 2022

@author: riya3
"""
import cv2
from rmn import RMN
m = RMN()
m.video_demo()

# or
image = cv2.imread("C:/Users/riya3/Downloads/University of Rochester/Research/Facial Expression Recognition Task/FacialExprTest/TestImage1.jpg")
results = m.detect_emotion_for_single_frame(image)
print(results)
image = m.draw(image, results)
cv2.imwrite("output.png", image)