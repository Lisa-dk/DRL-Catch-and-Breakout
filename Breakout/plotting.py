import matplotlib.pyplot as plt
import numpy as np

def plot_scores(scores):
    plt.plot(scores)
    plt.show()

scores = np.load('./logs/eval_rewards_random.npy')
print("average reward: ", np.mean(scores))
plot_scores(scores)
