from catch import CatchEnv
from policy import Policy
from collections import deque
from DQN import DQN
import numpy as np
import copy
import sys

BUFFER_SIZE = 2048
BATCH_SIZE = 64

def train(env, model, episodes=500):
    rewards = []
    mem_buffer = deque(maxlen=BUFFER_SIZE)

    for episode in range(episodes):
        state, reward, done = env.reset()

        while not done:
            action = model.get_action(state)
            next_state, reward, done = env.step(action)
            rewards.append(reward)

            mem_buffer.append((state, action, next_state, reward, done))
            state = next_state

            model.learn(mem_buffer, BATCH_SIZE, target_net_update=True)

    return model, rewards






def main():
    env = CatchEnv()
    model = DQN(env.state_shape, env.get_num_actions())
    train(env)

if __name__ == '__main__':
    main()