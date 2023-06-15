from catch import CatchEnv
from collections import deque
from DQN import DQN
import numpy as np
import random

import matplotlib.pyplot as plt

BUFFER_SIZE = 4096
BATCH_SIZE = 32
EPISODES = 2000

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

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
                action = model.act_greedy(state, env.get_num_actions())
            elif model.name == "policy":
                action = model.act(state, greedy=True)
            next_state, reward, done = env.step(action)
            
            next_state = np.transpose(next_state, [2, 0, 1])
            state = next_state
        total_reward += reward
            
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
                print("Update target network")
                model.learn(mem_buffer, BATCH_SIZE, target_net_update=True)
            else:
                model.learn(mem_buffer, BATCH_SIZE, target_net_update=False)

            state = next_state
            cumulative_reward += reward
            t += 1

        rewards.append(cumulative_reward)

    return model, eval_scores



def main():
    for run in range(1, 6):
        print("Run: ", run)
        env = CatchEnv()
        
        model = DQN(env.state_shape(), env.get_num_actions())
        model, scores = train_value(env, model, episodes=EPISODES)
        plot_scores(scores)

        np.save("group_07_catch_rewards_tets_" + str(run) + ".npy", scores)

    plot_scores(scores)

if __name__ == '__main__':
    main()