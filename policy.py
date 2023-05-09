import numpy as np
import torch
import torch.nn as nn

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
        self.out = nn.Linear(512, action_dim)
        
        self.softmax = nn.Softmax()

    def forward(self, state):
        out = self.relu1(self.conv1(state))
        out = self.relu2(self.conv2(out))
        out = self.relu3(self.conv3(out))

        out = torch.flatten(out, 1)

        out = self.relu4(self.fc1(out))

        return self.softmax(self.out(out))
    

class Policy():

    def __init__(self, input_shape, action_dim):
        self.name = "policy"
        self.model = PolicyNet(input_shape, action_dim)
        self.optim = torch.optim.RMSprop(self.model.parameters(), lr=0.0001)


    def loss(self, log_probs, rewards):
        vec_losses = -log_probs * rewards
        return 1 - torch.sum(vec_losses)
    
    def predict(self, state):
        return self.model(state)
    
    def act(self, state, greedy=False):
        state = torch.Tensor(state)
        with torch.no_grad():
            probs = self.model(state)
            if greedy:
                action = torch.argmax(probs).item()
            else:
                m = torch.distributions.categorical.Categorical(probs)
                action = m.sample()
                action = action.item()
            return action

