import torch
import torch.nn as nn
import random
import numpy as np
import copy

class DQN_Network(torch.nn.Module):
    #https://daiwk.github.io/assets/dqn.pdf
    def __init__(self, input_shape, n_actions, hidden_size):
        super(DQN_Network, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], hidden_size, 8, 4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_size, hidden_size*2, 4, 2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(hidden_size*2, hidden_size*2, 3, 1)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU()  
        self.fc3 = nn.Linear(512, n_actions)


    def forward(self, state):
        state = self.relu1(self.conv1(state))
        state = self.relu2(self.conv2(state))
        state = self.relu3(self.conv3(state))
        
        state = torch.flatten(state, 1)

        state = self.relu4(self.fc1(state))
        out = self.fc3(state)

        return out

class DQN():
    def __init__(self, input_shape, n_actions, learn_rate=0.00025):
        self.name = "DQN"
        self.online_network = DQN_Network(input_shape, n_actions, 32)
        self.target_network =  DQN_Network(input_shape, n_actions, 32)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.epsilon = 1.0
        self.epsilon_min = 0.1

        self.loss = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learn_rate)

    def act(self, state, action_space):
        if random.random() < self.epsilon:
            return random.randint(0, action_space - 1)
        with torch.no_grad():
            q_values = self.online_network(torch.Tensor(state))
        return torch.argmax(q_values).item()
    
    def update(self, y_pred, y_target):
        loss = self.loss(y_pred, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def epsilon_decay_update(self, decay_rate):
        new_epsilon = self.epsilon - self.epsilon * decay_rate
        self.epsilon = new_epsilon if new_epsilon > self.epsilon_min else self.epsilon_min
    
    def experience_replay(self, memory, batch_size, gamma=0.99):
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)

            batch = list(zip(*batch))
            batch = [list(category) for category in batch]

            states = torch.Tensor(np.array(batch[0]))
            actions = torch.Tensor(np.array(batch[1]))
            states_ = torch.Tensor(np.array(batch[2]))
            rewards = torch.Tensor(np.array(batch[3]))
            dones = torch.Tensor(np.array(batch[4]))

            q_values = self.online_network(states)
            q_values = q_values[np.arange(batch_size), actions.long()]

            with torch.no_grad():
                next_q_values = self.target_network(states_)

            q_targets = rewards + gamma * torch.max(next_q_values, axis=1).values * (1-dones)

            return q_values, q_targets
            
    
    def learn(self, memory, batch_size, target_net_update=False):
        if len(memory) < batch_size:
            return

        if target_net_update:
            self.target_network.load_state_dict(self.online_network.state_dict())

        q_values, q_targets = self.experience_replay(memory, batch_size)
        self.update(q_values, q_targets)
        self.epsilon_decay_update(0.0001)

        




            

        
    
