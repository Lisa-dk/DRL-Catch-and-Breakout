import numpy as np
import torch
import torch.nn as nn

class PolicyNet(nn.Module):

    def __init__(self, action_dim):
        super(PolicyNet, self).__init__()

        self.conv1 = nn.Conv2d(4, 8, kernel_size=5)
        self.relu1 = nn.ReLU()

        self.fc1 = nn.Linear(120, 16)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(16, action_dim)
        
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, state):
        out = self.relu1(self.conv1(state))
        out = torch.flatten(out, 1)
        out = self.relu2(self.fc1(out))

        return self.logsoftmax(self.out(out))
    

class Policy():

    def __init__(self, action_dim):
        self.model = PolicyNet(action_dim)
        self.optim = torch.optim.RMSprop(self.model.parameters())


    def loss(self, log_probs, rewards):
        vec_losses = -log_probs * rewards
        return 1 - torch.sum(vec_losses)
    
    def predict(self, state):
        return self.model(state)
    
    def act(self, state):
        with torch.no_grad():
            probs = self.model(state)
            action = torch.argmax(probs).item()
            return action

