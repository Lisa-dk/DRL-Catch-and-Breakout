from catch import CatchEnv
from policy import Policy
from collections import deque
from DQN import DQN
import numpy as np
import torch

import matplotlib.pyplot as plt

BUFFER_SIZE = 512
BATCH_SIZE = 16

def plot_scores(scores):
    plt.plot(scores)
    plt.show()


def evaluate_model(env, model, eval_episodes=10):
    avg_reward = 0
    for _ in range(eval_episodes):
        state = env.reset()
        state = np.transpose(state, [2, 0, 1])
        total_reward = 0
        done = False
        
        while not done:
            state = np.expand_dims(state, axis=0)
            action = model.act(state, greedy=True)
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.transpose(next_state, [2, 0, 1])
            state = next_state
            
        avg_reward += total_reward

    return avg_reward / eval_episodes
        


def train_value(env, model, episodes=500):
    rewards = []
    mem_buffer = deque(maxlen=BUFFER_SIZE)

    for episode in range(episodes):
        state = env.reset()
        state = np.transpose(state, [2, 0, 1])
        state = np.expand_dims(state, axis=0)
        done = False
        cumulative_reward = 0

        while not done:
            action = model.get_action(state, env.get_num_actions())
            print(action)
            next_state, reward, done = env.step(action)
            next_state = np.transpose(next_state, [2, 0, 1])
            next_state = np.expand_dims(next_state, axis=0)
            cumulative_reward += reward

            mem_buffer.append((state, action, next_state, reward, done))
            state = next_state

            model.learn(mem_buffer, BATCH_SIZE, target_net_update=True)

        print(f"Episode {episode}; reward: {cumulative_reward}")
        rewards.append(cumulative_reward)

    return model, rewards


def train_policy(env, model, episodes=500, max_episode_length=1000, eval_period=10):
    eval_scores = []
    
    for ep in range(episodes):
        if ep % eval_period == 0:
            eval_reward = evaluate_model(env, model)
            eval_scores.append(eval_reward)
            # print(f"Episode: {ep}; Reward evaluation: {eval_reward}")
            
        state = env.reset()
        state = np.transpose(state, [2, 0, 1])
        
        done = False
        t = 0
        history = {
            'states': [],
            'actions': [],
            'rewards': []
        }
        total_reward = 0
        
        while not done and t < max_episode_length:
            history['states'].append(state)
            state = np.expand_dims(state, axis=0)

            action = model.act(state)
            history['actions'].append(action)

            next_state, reward, done = env.step(action)
            history['rewards'].append(reward)

            total_reward += reward
            
            next_state = np.transpose(next_state, [2, 0, 1])
            state = next_state
            t += 1

        states = torch.Tensor(np.asarray(history['states']))
        actions = torch.Tensor(np.asarray(history['actions']))
        rewards = torch.Tensor(np.asarray(history['rewards']))

        probs = model.predict(states)
        action_probs = probs[:, actions.long()]
        
        loss = - torch.sum(action_probs * rewards) / t

        model.optim.zero_grad()
        loss.backward()
        model.optim.step()

        # print(f"Episode: {ep}; Total reward: {total_reward}")

    return model, eval_scores



def main():
    env = CatchEnv()
    # model = DQN(env.state_shape(), env.get_num_actions())
    # train_value(env, model)
    model = Policy(env.get_num_actions())
    model, scores = train_policy(env, model)
    plot_scores(scores)

if __name__ == '__main__':
    main()