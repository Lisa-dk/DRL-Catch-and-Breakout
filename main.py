from catch import CatchEnv
from policy import Policy
from collections import deque
from DQN import DQN
import numpy as np
import torch
import random
import sys

import matplotlib.pyplot as plt

BUFFER_SIZE = 3000
BATCH_SIZE = 32

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

class RandomAgent():
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def act():
        return random.randint(self.num_actions)


def evaluate_model(env, model, eval_episodes=10):
    avg_reward = 0.0

    for _ in range(eval_episodes):
        state = env.reset()
        state = np.transpose(state, [2, 0, 1])
        total_reward = 0.0
        done = False
        
        while not done:
            state = np.expand_dims(state, axis=0)
            if model.name == "DQN":
                action = model.act(state, env.get_num_actions())
            elif model.name == "policy":
                action = model.act(state, greedy=True)
            next_state, reward, done = env.step(action)
            total_reward += reward
            next_state = np.transpose(next_state, [2, 0, 1])
            state = next_state
            
        avg_reward += total_reward

    return avg_reward / eval_episodes
        


def train_value(env, model, episodes=2500, eval_period=10):
    rewards = []
    mem_buffer = deque(maxlen=BUFFER_SIZE) 
    eval_scores = []
    t = 0

    for episode in range(episodes):
        if episode % eval_period == 0:
            eval_reward = evaluate_model(env, model)
            eval_scores.append(eval_reward)
            print(f"Episode {episode}; Evaluation reward: {eval_reward}; Epsilon: {model.epsilon}")

        state = env.reset()
        state = np.transpose(state, [2, 0, 1])
        done = False
        cumulative_reward = 0
        iter_copy  = 750

        while not done:
            action = model.act(np.expand_dims(state, axis=0), env.get_num_actions())
            
            next_state, reward, done = env.step(action)
            next_state = np.transpose(next_state, [2, 0, 1])
            mem_buffer.append((state, action, next_state, reward, done))

            if t % iter_copy == 0:
                print("update")
                model.learn(mem_buffer, BATCH_SIZE, target_net_update=True)
            else:
                model.learn(mem_buffer, BATCH_SIZE, target_net_update=False)

            state = next_state
            cumulative_reward += reward
            t += 1

        rewards.append(cumulative_reward)

    return model, eval_scores


def train_policy(env, model, episodes=3000, max_episode_length=1000, eval_period=10):
    eval_scores = []
    batch_k = 1
    memory = deque(maxlen=BUFFER_SIZE)
    total_iterations = 0
    
    for ep in range(episodes):
        if ep % eval_period == 0:
            eval_reward = evaluate_model(env, model)
            eval_scores.append(eval_reward)
            print(f"Episode: {ep}; Reward evaluation: {eval_reward}")
        
        iter_copy  = 750
        t = 0

        for k in range(batch_k):
            
            state = env.reset()
            state = np.transpose(state, [2, 0, 1])
            
            done = False
            
            while not done and t < max_episode_length:
                action  = model.act(np.expand_dims(state, axis=0))
            
                next_state, reward, done = env.step(action)
                
                next_state = np.transpose(next_state, [2, 0, 1])
                memory.append((state, action, reward, next_state, done))

                model.learn_value(memory, BATCH_SIZE)
                if total_iterations % iter_copy == 0:
                    model.update_target_network()

                state = next_state
                t += 1
                total_iterations += 1

        most_recent = list(memory)[-t:]

        states = torch.Tensor(np.array([most_recent[i][0] for i in range(t)]))
        actions = torch.Tensor(np.asarray([most_recent[i][1] for i in range(t)]))
        rewards = torch.Tensor(np.asarray([most_recent[i][2] for i in range(t)]))

        probs = model.predict(states)
        actions = actions[:, None].long()  
    
        action_probs = probs.gather(1, actions)
        
        model.learn_policy(action_probs, rewards, t)

    return model, eval_scores



def main():
    algorithm = sys.argv[1]
    print(algorithm)
    env = CatchEnv()
    
    if algorithm.lower() == "dqn":
        model = DQN(env.state_shape(), env.get_num_actions())
        model, scores = train_value(env, model)
    else:
        model = Policy(env.state_shape(), env.get_num_actions())
        model, scores = train_policy(env, model)
    plot_scores(scores)

if __name__ == '__main__':
    main()