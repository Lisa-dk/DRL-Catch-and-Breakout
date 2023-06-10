import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

scores = np.load('./logs/eval_rewards_a2c_0.001.npy')
plot_scores(scores)
