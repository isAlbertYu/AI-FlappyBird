# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 20:02:11 2019

@author: Administrator
"""
import cv2
import numpy as np
import tensorflow as tf

input_data = tf.placeholder("float", [None, 80, 80, 4])
shape = input_data.shape #? 80 80 4
print('shape = ', shape)
print('shape[-1] = ', shape[-1])
c = shape[-1]
print("c = ", c)
a = [80, 80, int(c), 32]
