import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import os
import sys
from eval_policy import evaluate_policy

SEED = 42
EVAL_EPISODES = 100
TRAIN_STEPS = int(1e7)
N_ENVS = 16
N_STACK = 4
ENV = "BreakoutNoFrameskip-v4" # https://www.codeproject.com/Articles/5271947/Introduction-to-OpenAI-Gym-Atari-Breakout

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib notebook

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

def make_environment(env_name, n_env, seed, log_dir, n_stack):
    env = make_atari_env(env_name, n_envs=n_env, seed=seed, monitor_dir=log_dir)
    env = VecFrameStack(env, n_stack=n_stack)
    return env

def main():
    algorithm = sys.argv[1]
    lr_rate = sys.argv[2]
    lr_rate = float(lr_rate)
    eval = int(sys.argv[3])
    log_dir = "./logs/train/"
    os.makedirs(log_dir, exist_ok=True)

    # Preparing the environment
    env = make_environment(env_name=ENV, n_env=N_ENVS, seed=SEED, log_dir=log_dir, n_stack=N_STACK)
    
    print("Observation Space: ", env.observation_space)
    print("Action Space       ", env.action_space)
    print("Learning rate: ", lr_rate)
    
    if not eval:
        # Model selection
        if algorithm.lower() == "ppo":
            model = PPO("CnnPolicy", env, seed=SEED, tensorboard_log=log_dir, verbose=1, learning_rate=lr_rate)
        elif algorithm.lower() == "a2c":
            model = A2C("CnnPolicy", env, seed=SEED, tensorboard_log=log_dir, verbose=1, learning_rate=lr_rate)
        else:
            print("Enter a valid model (ppo or a2c)")
            exit()
        # Training and saving the agent
        model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=algorithm.lower() + '_' + str(lr_rate))
        model.save('trained_' + algorithm + '_model_10m_' + str(lr_rate))

    else:
        print("evaluating")
        # Preparing environment
        env = make_environment(env_name=ENV, n_env=N_ENVS, seed=SEED, log_dir=log_dir, n_stack=N_STACK)
        # Loading the model
        model = A2C.load('trained_' + algorithm + '_model_10m_' + str(lr_rate))
        # Evaluation
        random_eval_rewards = evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True, random_player=True)
        model_eval_rewards = evaluate_policy(model, env, n_eval_episodes=100, return_episode_rewards=True, random_player=False)
        
        print("Average model reward: ", np.mean(model_eval_rewards[0]), "SD: ", np.std(model_eval_rewards[0]))
        print("Average random reward: ", np.mean(random_eval_rewards[0]), "SD: ", np.std(random_eval_rewards[0]))

        np.save('./logs/eval_rewards_' +  algorithm.lower() + '_10m_' + str(lr_rate) + '.npy', model_eval_rewards[0])
        np.save('./logs/eval_rewards_random_10m_' + str(lr_rate) + '.npy', random_eval_rewards[0])



    

if __name__ == '__main__':
    main()