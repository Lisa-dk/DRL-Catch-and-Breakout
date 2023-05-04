from catch import CatchEnv
from policy import Policy
from collections import deque
from DQN import DQN
import numpy as np
import copy
import sys
import torch

BUFFER_SIZE = 512
BATCH_SIZE = 16

def train_value(env, model, episodes=500):
    rewards = []
    mem_buffer = deque(maxlen=BUFFER_SIZE)

    for episode in range(episodes):
        state = env.reset()
        state = np.transpose(state, [2, 0, 1])
        done = False
        cumulative_reward = 0

        while not done:
            action = model.get_action(state, env.get_num_actions())
            print(action)
            next_state, reward, done = env.step(action)
            next_state = np.transpose(next_state, [2, 0, 1])
            cumulative_reward += reward

            mem_buffer.append((state, action, next_state, reward, done))
            state = next_state

            model.learn(mem_buffer, BATCH_SIZE, target_net_update=True)

        print(f"Episode {episode}; reward: {cumulative_reward}")
        rewards.append(cumulative_reward)

    return model, rewards


def train_policy(env, model, episodes=500, max_episode_length=500):
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        t = 0
        history = []
        
        while not done and t < max_episode_length:
            action = model.act(state)
            next_state, reward, done = env.step(action)
            history.append((state, action, reward))
            state = next_state
            t += 1

        history = torch.Tensor(history)
        states = history[:, 0]
        actions = history[:, 1]
        rewards = history[:, 2]
        probs = model.predict(states)
        probs = probs[:, actions.long()]
        
        

        sum_probs = torch.sum(probs)
        G_t = torch.sum(rewards)
        
        


    return model, rewards



def main():
    env = CatchEnv()
    model = DQN(env.state_shape(), env.get_num_actions())
    train_value(env, model)

if __name__ == '__main__':
    main()