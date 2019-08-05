# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 13:45:33 2019

@author: Administrator
"""

import cv2

def imgBinarizal(img):
    img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return img