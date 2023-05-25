import gymnasium as gym
import numpy as np


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
    

if __name__ == '__main__':
    main()