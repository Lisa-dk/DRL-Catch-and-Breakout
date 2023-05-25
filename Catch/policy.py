import numpy as np
import torch
import torch.nn as nn
import random

class PolicyNet(nn.Module):

    def __init__(self, input_shape, action_dim):
        super(PolicyNet, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.fc1 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU()

        self.prob_out = nn.Linear(512, action_dim)
        self.softmax = nn.Softmax(dim=1)
        
        self.value_out = nn.Linear(512, 1)

    def forward(self, state, value=False):
        out = self.relu1(self.conv1(state))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))

        out = torch.flatten(out, 1)

        out = self.relu4(self.fc1(out))

        if value:
            return self.value_out(out)

        return self.softmax(self.prob_out(out))
    

class Policy():

    def __init__(self, input_shape, action_dim):
        self.name = "policy"
        self.online_network = PolicyNet(input_shape, action_dim)
        self.target_network = PolicyNet(input_shape, action_dim)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optim = torch.optim.Adam(self.online_network.parameters(), lr=0.0001)
        self.value_loss = torch.nn.MSELoss()
    
    def predict(self, state):
        return self.online_network(state)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())
    
    def act(self, state, greedy=False):
        state = torch.Tensor(state)
        with torch.no_grad():
            probs = self.online_network(state)
            if greedy:
                action = torch.argmax(probs).item()
            else:
                m = torch.distributions.categorical.Categorical(probs)
                action = m.sample()
                action = action.item()
            return action
        
    def experience_replay(self, memory, batch_size, gamma=0.99):
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)

            batch = list(zip(*batch))
            batch = [list(category) for category in batch]

            states = torch.Tensor(np.array(batch[0]))
            rewards = torch.Tensor(np.array(batch[2]))
            rewards = rewards[:, None]
            states_ = torch.Tensor(np.array(batch[3]))
            dones = torch.Tensor(np.array(batch[4]))
            dones = dones[:, None]

            v_values = self.online_network(states, value=True)

            with torch.no_grad():
                next_values = self.target_network(states_, value=True)
            v_targets = rewards + gamma * next_values * (1-dones)

            return v_values, v_targets

    def learn_policy(self, action_probs, rewards, t):
        loss = 1 - (torch.sum(torch.log(action_probs) * rewards) / t)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()


    def learn_value(self, memory, batch_size):
        if len(memory) < batch_size:
            return

        v_values, v_targets = self.experience_replay(memory, batch_size)

        loss = self.value_loss(v_values, v_targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()