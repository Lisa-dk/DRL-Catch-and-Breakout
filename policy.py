import numpy as np

class Policy():

    def __init__(self, state_dim, action_dim):
        self.weights = np.random.normal(size=(action_dim, state_dim))

    def forward(self, state):
        return np.softmax(np.dot(self.weights, state))