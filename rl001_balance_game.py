import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

env = gym.make('CartPole-v0').unwrapped

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        '''Transition 저장'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

'''
DQN 알고리즘
목표 : 할인된 누적보상(discounted cumulative reward)을 극대화하려는 정책(policy)학습
에이전트에게 먼 미래의 불확실한 보상보다 가까운 미래의 보상이 더 중요하게 판단시킨다
Q 함수는 Bellman 방정식을 준수

loss = 시간차오류(temporal difference error)

loss 함수는 Huber loss 를 사용한다 / 오류가 작을때는 mse 오류가 클때는 mae 와 유사하다
>> 이걸 쓰는 이유는 Q함수의 추정이 혼란스러울때 값에대한 확신을 주기때문

https://tutorials.pytorch.kr/intermediate/reinforcement_q_learning.html
'''

class DQN(nn.Module):
    def __init__(self, h, w, outputs)