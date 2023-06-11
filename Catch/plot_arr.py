import numpy as np
import matplotlib.pyplot as plt

FILE = "group_07_catch_rewards_"

def main():
    rewards = []
    for idx in range(1, 6):
        file = FILE + str(idx) + ".npy"
        arr = np.load(file)
        rewards.append(arr.tolist())
    
    mean_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)
    episodes = [ep for ep in range(0, len(mean_rewards) * 10, 10)]

    ax = plt.subplot()
    ax.plot(episodes, mean_rewards, '-', color='#1a9988')

    ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2, color='#105f55')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    ax.set_ylabel('Reward')
    ax.set_xlabel('Episode')
    plt.title('Mean Rewards across 5 Runs')

    plt.show()

if __name__ == "__main__":
    main()