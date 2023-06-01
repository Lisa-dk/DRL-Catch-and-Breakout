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

EPISODES = 3000
LEARNING_RATE = 0.0005
SEED = 42
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

def main():
    algorithm = sys.argv[1]

    env = gym.make("BreakoutNoFrameskip-v4")
    print("Observation Space: ", env.observation_space)
    print("Action Space       ", env.action_space)
    log_dir = "./logs/train/"

    env = make_atari_env(ENV, n_envs=4, seed=SEED, monitor_dir=log_dir)
    env = VecFrameStack(env, n_stack=4)

    os.makedirs(log_dir, exist_ok=True)

    if algorithm.lower() == "ppo":
        model = PPO("CnnPolicy", env, verbose=1, seed=SEED, tensorboard_log=log_dir)
    elif algorithm.lower() == "a2c":
        model = A2C("CnnPolicy", env, verbose=1, seed=SEED, tensorboard_log=log_dir)
    else:
        print("Enter a valid model (ppo or a2c)")
        exit()

    eval_callback = StopTrainingOnMaxEpisodes(max_episodes=10, verbose=1)

    for ep in range(EPISODES/10):
        eval_rewards = []
        model.learn(total_timesteps=int(1e6), tb_log_name="A2C", callback=eval_callback, reset_num_timesteps=False)
        avg_rewards = 0.0
        num_iter = 0
        for _ in range(10):
            obs = env.reset()
            done = False
            while not done:
                action, _states = model.predict(obs)
                obs, rewards, done, info = env.step(action)
                env.render()
                avg_rewards += rewards
                num_iter += 1
        eval_rewards.append(avg_rewards/num_iter)
    np.save("./rewards_breakout_" + algorithm + ".npy", eval_rewards)
    plot_scores(avg_rewards)


    

if __name__ == '__main__':
    main()