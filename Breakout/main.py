import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import os


EPISODES = 3000
LEARNING_RATE = 0.0005
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

def eval_greedy(env, model, eval_runs=10):
    avg_reward = 0.0
    for run in range(eval_runs):
        reward_sum = 0
        done = False
        state = env.reset()

        while not done:
            # action = model.act(state) # make greedy
            action = env.action_space.sample()
            next_state, reward, done = env.step(action)

            reward_sum += reward
            state = next_state

        avg_reward += avg_reward

    avg_reward /= eval_runs
    return avg_reward
 
def train(env, model, eval_period=10):
    eval_reward_hist = np.zeros(int(EPISODES / eval_period))

    for ep in range(EPISODES):
        if ep % eval_period == 0:
            eval_reward = eval_greedy(env, model)
            eval_reward_hist[ep / eval_period] = eval_reward

        total_reward = 0
        done = False
        state = env.reset()

        while not done:
            action = model.act(state)
            next_state, reward, done = env.step(action)

            total_reward += reward
            state = next_state


    return eval_reward_hist


def main():
    env = gym.make("BreakoutNoFrameskip-v4")
    print("Observation Space: ", env.observation_space)
    print("Action Space       ", env.action_space)
    log_dir = "./logs/train/"

    env = make_atari_env(ENV, n_envs=4, seed=42, monitor_dir=log_dir)
    env = VecFrameStack(env, n_stack=4)
    os.makedirs(log_dir, exist_ok=True)
    # I don't really understand how this works yet
    eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=2000, n_eval_episodes=10,
                             deterministic=False, render=False)
    plotting_callback = PlottingCallback()

    model = PPO("CnnPolicy", env, verbose=1, seed=42, tensorboard_log=log_dir)
    model.learn(total_timesteps=int(1e6), tb_log_name="A2C")

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _state = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render("human")
    #     # VecEnv resets automatically
    #     # if done:
    #     #   obs = vec_env.reset()
    
    # Use deterministic actions for evaluation


    

if __name__ == '__main__':
    main()