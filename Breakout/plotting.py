import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

a2c_scores = np.load('./logs/eval_rewards_a2c_10m_0.0007.npy')
random_rewards = np.load('./logs/eval_rewards_random.npy')

print(np.std(a2c_scores), np.std(random_rewards))

ax = plt.subplot()

ax.plot(a2c_scores, color='#1a9988', label='A2C')

ax.plot(random_rewards, color='black', label='Random')

# Axis visibility
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
# ax.grid(axis = 'y')

# Axis labels and ticks
ax.set_ylabel('Total Reward')
ax.set_xlabel('Episodes')
plt.title('Total Reward Breakout Evaluation')

ax.legend(bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 11})
plt.tight_layout()

plt.show()
