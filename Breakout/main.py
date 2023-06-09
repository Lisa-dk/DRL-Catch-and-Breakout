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

SEED = 42
EVAL_EPISODES = 50
TRAIN_STEPS = int(5e6)
N_ENVS = 16
ENV = "BreakoutNoFrameskip-v4" # https://www.codeproject.com/Articles/5271947/Introduction-to-OpenAI-Gym-Atari-Breakout

import matplotlib.pyplot as plt
import numpy as np
# %matplotlib notebook


class PlottingCallback(BaseCallback):
    """
    Callback for plotting the performance in realtime.

    :param verbose: (int)
    """
    def __init__(self, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self._plot = None
    
    def _on_step(self) -> bool:
        log_dir = "/tmp/bo/"
        # get the monitor's data
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if self._plot is None: # make the plot
            plt.ion()
            fig = plt.figure(figsize=(6,3))
            ax = fig.add_subplot(111)
            line, = ax.plot(x, y)
            self._plot = (line, ax, fig)
            plt.show()
        else: # update and rescale the plot
            self._plot[0].set_data(x, y)
            self._plot[-2].relim()
            self._plot[-2].set_xlim([self.locals["total_timesteps"] * -0.02, 
                                    self.locals["total_timesteps"] * 1.02])
            self._plot[-2].autoscale_view(True,True,True)
            self._plot[-1].canvas.draw()

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

def eval_random(env):
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        total_reward = 0.0
        obs = env.reset()
        dones = [False for _ in range(N_ENVS)]
        while not (True in dones):
            action = [env.action_space.sample() for x in range(N_ENVS)]
            obs, rewards, dones, info = env.step(action)
            total_reward += np.sum(rewards)
        eval_rewards.append(total_reward)
    return eval_rewards

def main():
    algorithm = sys.argv[1]
    lr_rate = sys.argv[2]
    lr_rate = float(lr_rate)
    print(lr_rate)

    env = gym.make("BreakoutNoFrameskip-v4", render_mode="human")
    print("Observation Space: ", env.observation_space)
    print("Action Space       ", env.action_space)
    log_dir = "./logs/train/"

    env = make_atari_env(ENV, n_envs=N_ENVS, seed=SEED, monitor_dir=log_dir)
    env = VecFrameStack(env, n_stack=4)

    os.makedirs(log_dir, exist_ok=True)

    if algorithm.lower() == "ppo":
        model = PPO("CnnPolicy", env, seed=SEED, tensorboard_log=log_dir, verbose=1, learning_rate=lr_rate)
    elif algorithm.lower() == "a2c":
        model = A2C("CnnPolicy", env, seed=SEED, tensorboard_log=log_dir, verbose=1, learning_rate=lr_rate)
    else:
        print("Enter a valid model (ppo or a2c)")
        exit()


    model.learn(total_timesteps=TRAIN_STEPS, tb_log_name=algorithm.lower() + '_' + str(lr_rate), reset_num_timesteps=True)

  
    eval_rewards = []
    for _ in range(EVAL_EPISODES):
        total_reward = 0.0
        obs = env.reset()
        dones = [False for _ in range(4)]
        while not (True in dones):
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            #env.render()
            total_reward += np.sum(rewards)
        eval_rewards.append(total_reward)

    # plot_scores(eval_rewards)
    # random_eval_rewards = eval_random(env)
    # plot_scores(eval_random(random_eval_rewards))

    model.save('trained_' + algorithm + '_model_' + str(lr_rate))
    np.save('./logs/eval_rewards_' +  algorithm.lower() + '_' + str(lr_rate) + '.npy', eval_rewards)
    # np.save('./logs/eval_rewards_random.npy', random_eval_rewards)

    #plot_scores(eval_rewards)


    

if __name__ == '__main__':
    main()