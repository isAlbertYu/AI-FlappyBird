# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:05:15 2019

@author: Administrator
"""
import tensorflow as tf
import numpy as np
from myNet import DQ_net
import cv2, random
import game.wrapped_flappy_bird as game
from Experience import Experience
from image_process import imgBinarizal
from collections import deque

GAMMA = 0.99
FRAME_PER_ACTION = 1 # 每个动作持续几帧
MAX_STEP = 10000 # 训练轮数
ACTIONS_NUM = 2
epsilon = 0.1

BATCH = 32
EXPPOOL_MAX_SIZE = 500
OBSERVE = 1000. # timesteps to observe before training
EXPLORE = 3000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon 探索
INITIAL_EPSILON = 0.1 # starting value of epsilon

x = tf.placeholder("float", [None, 80, 80, 4])
y = tf.placeholder("float", [None])

y_ = DQ_net(x, output_size=ACTIONS_NUM)

# reduction_indices = axis  0 : 列  1: 行
# 因 y 是数值，而readout: 网络模型预测某个行为的回报 大小[1, 2] 需要将readout 转为数值，
# 所以有tf.reduce_mean(tf.multiply(readout, a), axis=1) 数组乘法运算，再求均值。
# 其实，这里readout_action = tf.reduce_mean(readout, axis=1) 直接求均值也是可以的。
readout_action = tf.reduce_mean(y_, axis=1)
cost = tf.reduce_mean(tf.square(y - readout_action))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # 开启游戏仿真器
    game_state = game.GameState()
    # 首先先不做任何工作，获取游戏的初始状态 and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS_NUM)
    do_nothing[0] = 1
    
    
    ExpPool = deque() # 经验池
    next_img_stack = 0
    for t in range(MAX_STEP):
        print('t = ', t)
        if t == 0:
            action_t = do_nothing
        '''
        game_state.frame_step
        返回 游戏画面图像 奖励 中止标志符
        '''

        img_t, reward_t, tml = game_state.frame_step(action_t, showFrame=True)
        
        img_t = imgBinarizal(img_t)
        
        if t == 0:
            img_stack = np.stack((img_t,) * 4, axis=2)
        else:
            img_stack = next_img_stack
            
        # 预测结果（当前状态不同行为action的回报，其实也就 往上，往下 两种行为）
        y_pred = sess.run(y_, feed_dict={x : [img_stack]})[0]
        
        action_t = np.zeros([ACTIONS_NUM])
        
        if t % FRAME_PER_ACTION == 0:
            # 且epsilon是随着模型稳定趋势衰减的，也就是模型越稳定，探索次数越少。
            if random.random() <= epsilon:
                # 在ACTIONS范围内随机选取一个作为当前状态的即时动作
                print("-Random Action-")
                index = random.randrange(ACTIONS_NUM)
                action_t[index] = 1
            else:
                # 输出 使用深Q网络预测的回报最大的动作作为下一步的方向
                index = np.argmax(y_pred)
                print('index = ', index)
                action_t[index] = 1
        else:
            action_t[0] = 1 # do nothing

        # 取之前状态的前3帧图片 + 当前得到的1帧图片
        # 每次输入都是4幅图像
        img_t = np.reshape(img_t, (80, 80, 1))
        next_img_stack = np.append(img_t, img_stack[:, :, :3], axis=2)
    
        Exp = Experience(img_stack, action_t, reward_t, next_img_stack, tml)

        # 以队列方式将经验保存入经验池中
        ExpPool.append(Exp)
        if len(ExpPool) > EXPPOOL_MAX_SIZE:
            ExpPool.popleft()

        '''
                # 观察完毕 开始训练
        '''
        if t > OBSERVE:
            # scale down epsilon 模型稳定，减少探索次数。
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
            
            # 从经验池中随机获取batch = 32个经验
            exp_batch = random.sample(ExpPool, BATCH)
            
            s_batch = [exp.currentState for exp in exp_batch]
            a_batch = [exp.action for exp in exp_batch]
            r_batch = [exp.reward for exp in exp_batch]
            next_s_batch = [exp.nextState for exp in exp_batch]
            
            y_batch = []
            y_pred_batch = sess.run(y_, feed_dict = {x : next_s_batch})#(32, 2)
            
            for i in range(len(exp_batch)):
                terminal = exp_batch[i].terminal
                # if terminal, only equals reward
                if terminal:  # 触障，终止
                    y_batch.append(r_batch[i])
                else: # 即时奖励 + 下一阶段回报
                    y_batch.append(r_batch[i] + GAMMA * np.max(y_pred_batch[i]))
                    
            # 根据cost -> 梯度 -> 反向传播 -> 更新参数
            # perform gradient step
            # 必须要3个参数，y, a, s 只是占位符，没有初始化
            # 在 train_step过程中，需要这3个参数作为变量传入
            sess.run(train_step, feed_dict = {x : s_batch, y : y_batch})

        # print info
        stage = ""
        if t <= OBSERVE:
            stage = "观察期"
        else:
            stage = "训练期"
        print("terminal", tml,
                "timestep", t, 
                "stage", stage,
                "ε", epsilon,
                "action", index,
                "reward", reward_t)
        
    saver.save(sess, 'model/ai_bird-dqn', global_step = t)








