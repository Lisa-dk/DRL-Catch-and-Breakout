import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

scores = np.load('./logs/eval_rewards_a2c_10m_0.0007.npy')
print("average reward: ", np.mean(scores))
plot_scores(scores)
