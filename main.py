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
    mem_buffer = deque(maxlen=BUFFER_SIZE) # to numpy array and override from start when full
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
    batch_k = 8
    
    for ep in range(episodes):
        if ep % eval_period == 0:
            eval_reward = evaluate_model(env, model)
            eval_scores.append(eval_reward)
            print(f"Episode: {ep}; Reward evaluation: {eval_reward}")
        
        total_reward = 0
        t = 0
        
        for k in range(batch_k):
            
            state = env.reset()
            state = np.transpose(state, [2, 0, 1])
            
            done = False
            history = {
                'states': [],
                'actions': [],
                'rewards': []
            }
            
            while not done and t < max_episode_length:
                history['states'].append(state)
                state = np.expand_dims(state, axis=0)

                action  = model.act(state)

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
        actions = actions[:, None].long()  
        #m = torch.distributions.categorical.Categorical(probs)
        # action_probs = probs[:, actions.long()]
        action_probs = probs.gather(1, actions)
        
        # action_probs = action_probs.reshape(t)
        
        loss = 1 - (torch.sum(torch.log(action_probs) * rewards) / t)
        #loss = 1 - torch.sum(m.log_prob(actions) * rewards) / t

        model.optim.zero_grad()
        loss.backward()
        model.optim.step()

        # print(f"Episode: {ep}; Total reward: {total_reward}")

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