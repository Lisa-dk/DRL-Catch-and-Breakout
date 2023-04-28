from catch import CatchEnv
from policy import Policy
from collections import deque
import numpy as np
import copy

BUFFER_SIZE = 2048

def train(env, model, episodes=500, use_prev_state=True):
    rewards = []
    mem_buffer = deque(maxlen=BUFFER_SIZE)

    for episode in range(episodes):
        state, reward, done = env.reset()
        # TODO: test if this is necessary
        if use_prev_state:
            prev_state = copy.deepcopy(state)

        while not done:
            action = model.policy(state)
            if use_prev_state:
                prev_state = copy.deepcopy(state)
            state, reward, done = env.step(action)

            mem_buffer.append((state, action, r))

    

def main():
    env = CatchEnv()
    train(env)

if __name__ == '__main__':
    main()