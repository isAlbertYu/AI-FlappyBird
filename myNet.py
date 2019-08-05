# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:48:59 2019

@author: Administrator
"""

import tensorflow as tf

'''
该网络模型用于拟合近似状态值函数(拟合Q表格)
返回 output (1, output_size) 为 当状态input_data下，各个动作的回报值
'''
def DQ_net(input_data, output_size):
    shape = input_data.shape #? 80 80 4

    # network weights
    W_conv1 = weight_variable([8, 8, int(shape[-1]), 32])
    b_conv1 = bias_variable([32])
    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(input_data, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    print('h_pool3.shape = ', h_pool3.shape)
    h_pool3_flat = tf.reshape(h_pool3, [-1, 576])

    W_fc1 = weight_variable([576, 512])
    b_fc1 = bias_variable([512])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)


    W_fc2 = weight_variable([512, output_size])
    b_fc2 = bias_variable([output_size])
    output = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return output
    
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

# padding = ‘SAME’=> new_height = new_width = W / S （结果向上取整）
# padding = ‘VALID’=> new_height = new_width = (W – F + 1) / S （结果向上取整）
def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
