# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:07:22 2019

@author: Administrator
"""
'''
currentState: 当前状态（80 * 80 * 4）
action: 即将行为 （1 * 2）
reward: 即时奖励
nextState: 下一状态
terminal: 当前行动的结果（是否碰到障碍物 True => 是 False =>否）
'''
class Experience:
    def __init__(self, currentState, action, reward, nextState, terminal):
        self.currentState = currentState
        self.action = action
        self.reward = reward
        self.nextState = nextState
        self.terminal = terminal
        