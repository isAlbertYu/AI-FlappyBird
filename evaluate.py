# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow as tf
import numpy as np
from myNet import DQ_net
import cv2, random
import game.wrapped_flappy_bird as game
from Experience import Experience
from image_process import imgBinarizal
from collections import deque
# 清除默认图的堆栈，并设置全局图为默认图 
tf.reset_default_graph()
MAX_STEP = 10000 # 训练轮数
ACTIONS_NUM = 2

x = tf.placeholder("float", [None, 80, 80, 4])
a = tf.placeholder("float", [None, ACTIONS_NUM])
y = tf.placeholder("float", [None])

y_ = DQ_net(x, output_size=ACTIONS_NUM)

saver = tf.train.Saver()
with tf.Session() as sess:
    
#    model_file = tf.train.latest_checkpoint('model/bird-dqn-2920000') 
    path = 'D:\\myProgram\\PyProject\\DeepLearningFlappyBird\\saved_networks\\bird-dqn-2920000'
    saver.restore(sess, path)
    
    # 开启游戏仿真器
    game_state = game.GameState()
    # 首先先不做任何工作，获取游戏的初始状态 and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS_NUM)
    do_nothing[0] = 1
    
    action_t = do_nothing
    next_img_stack = 0
    for t in range(MAX_STEP):
        
        img_t, _, tml = game_state.frame_step(action_t, showFrame=True)
        img_t = imgBinarizal(img_t)
        if t == 0:
            img_stack = np.stack((img_t,) * 4, axis=2)
        else:
            img_stack = next_img_stack
            
        
        # 预测结果（当前状态不同行为action的回报，其实也就 往上，往下 两种行为）
        y_pred = sess.run(y_, feed_dict={x : [img_stack]})[0]
        
        # 输出 使用深Q网络预测的回报最大的动作作为下一步的方向
        index = np.argmax(y_pred)
        
        tmp = np.zeros(ACTIONS_NUM)
        tmp[index] = 1
        action_t = tmp
        print('action_t = ', action_t)
        
        # 取之前状态的前3帧图片 + 当前得到的1帧图片
        # 每次输入都是4幅图像
        img_t = np.reshape(img_t, (80, 80, 1))
        next_img_stack = np.append(img_t, img_stack[:, :, :3], axis=2)
    


