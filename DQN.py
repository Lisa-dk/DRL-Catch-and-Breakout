import torch
import torch.nn as nn
import random
from tqdm import tqdm
import pickle 
import gym
import numpy as np
import collections 
import cv2
import copy
import pylab as pl
from collections import deque

class DQN_Network():
    def __init__(self, input_shape, n_actions, hidden_size, learn_rate=0.05):
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_shape, hidden_size),
            torch.nn.ReLu(),
            torch.nn.Linear(hidden_size, hidden_size*2),
            torch.nn.ReLu(),
            torch.nn.Linear(hidden_size*2, n_actions)
        )
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learn_rate)

    def update(self, y_pred, y_target):
        loss = self.loss(y_pred, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class DQN():
    def __init__(self, input_shape, n_actions, max_mem_size):
        self.online_network = DQN_Network(input_shape, n_actions, 64)
        self.target_network = copy.deepcopy(self.online_network)
        self.init_memory(max_mem_size)
        self.epsilon = 0.5
        self.epsilon_min = 0.001

    def init_memory(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    def get_action(self, state, action_space):
        if random.random() < self.epsilon:
            return action_space.sample()
        
        q_values = self.network(state)
        return torch.argmax(q_values).item()
    
    def epsilon_decay_update(self, decay_rate):
        new_epsilon = self.epsilon - self.epsilon * decay_rate
        self.epsilon = new_epsilon if new_epsilon > self.epsilon_min else self.epsilon_min
    
    def experience_replay(self, batch_size, gamma=0.9):
        if len(self.memory) >= batch_size:
            batch = random.sample(self.memory, batch_size)
            batch_t = list(map(list, zip(*batch)))

            states = torch.Tensor(np.array(batch_t[0]))
            actions = torch.Tensor(np.array(batch_t[1]))
            states_ = torch.Tensor(np.array(batch_t[2]))
            rewards = torch.Tensor(np.array(batch_t[3]))
            dones = torch.Tensor(np.array(batch_t[4]))

            q_values = self.online_network(states)
            next_q_values = self.target_network(states_)

            q_targets = rewards + gamma * torch.max(next_q_values, axis=1).values * dones

            return q_values, q_targets
            
    
    def learn(self, batch_size, target_net_update=0):
        q_values, q_targets = self.experience_replay(batch_size)
        self.online_network.update(q_values, q_targets)
        self.epsilon_decay_update(0.99)
        # Update target network
        if target_net_update:
            self.target_network.load_state_dict(self.online_network.state_dict())



            

        
    
