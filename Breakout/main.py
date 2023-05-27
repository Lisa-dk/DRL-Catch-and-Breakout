import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


EPISODES = 3000
LEARNING_RATE = 0.0005
ENV = "BreakoutNoFrameskip-v4" # https://www.codeproject.com/Articles/5271947/Introduction-to-OpenAI-Gym-Atari-Breakout


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

    train_env = make_atari_env(ENV, n_envs=4, seed=42)
    train_env = VecFrameStack(train_env, n_stack=4)
    eval_env = make_atari_env(ENV, n_envs=4, seed=42)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    # I don't really understand how this works yet
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=2000, n_eval_episodes=10,
                             deterministic=False, render=False)

    model = A2C("CnnPolicy", train_env, verbose=1, seed=42)
    model.learn(total_timesteps=20_000, callback=eval_callback)

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