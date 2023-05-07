import torch
import torch.nn
import random
import numpy as np
import copy

class DQN_Network(torch.nn.Module):
    def __init__(self, input_shape, n_actions, hidden_size):
        super(DQN_Network, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_shape[0], hidden_size, 5)
        self.relu1 = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(6400, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size*2)
        self.relu3 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_size*2, n_actions)


    def forward(self, state):
        state = self.relu1(self.conv1(state))
        state = torch.flatten(state, 1)
        state = self.relu2(self.fc1(state))
        state = self.relu3(self.fc2(state))
        out = self.fc3(state)

        return out

class DQN():
    def __init__(self, input_shape, n_actions, learn_rate=0.05):
        self.online_network = DQN_Network(input_shape, n_actions, 64)
        self.target_network = copy.deepcopy(self.online_network)
        self.epsilon = 0.5
        self.epsilon_min = 0.001

        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learn_rate)

    def get_action(self, state, action_space):
        if random.random() < self.epsilon:
            return random.randint(0, action_space - 1)
        with torch.no_grad():
            q_values = self.online_network(torch.Tensor(state))
            print(q_values)
        return torch.argmax(q_values).item()
    
    def update(self, y_pred, y_target):
        loss = self.loss(y_pred, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def epsilon_decay_update(self, decay_rate):
        new_epsilon = self.epsilon - self.epsilon * decay_rate
        self.epsilon = new_epsilon if new_epsilon > self.epsilon_min else self.epsilon_min
    
    def experience_replay(self, memory, batch_size, gamma=0.9):
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            batch_t = list(map(list, zip(*batch)))

            states = torch.Tensor(np.array(batch_t[0]))
            actions = torch.Tensor(np.array(batch_t[1]))
            states_ = torch.Tensor(np.array(batch_t[2]))
            rewards = torch.Tensor(np.array(batch_t[3]))
            dones = torch.Tensor(np.array(batch_t[4]))

            q_values = self.online_network(states)
            with torch.no_grad():
                next_q_values = self.target_network(states_)

            q_targets = rewards + gamma * torch.max(next_q_values, axis=1).values * dones

            return q_values, q_targets
            
    
    def learn(self, memory, batch_size, target_net_update=0):
        if len(memory) < batch_size:
            return

        q_values, q_targets = self.experience_replay(memory, batch_size)
        self.online_network.update(q_values, q_targets)
        self.epsilon_decay_update(0.99)
        # Update target network
        if target_net_update:
            self.target_network.load_state_dict(self.online_network.state_dict())



            

        
    
