import numpy as np
import matplotlib.pyplot as plt

FILE = "group_07_catch_rewards_tets_1.npy"

def main():
    arr = np.load(FILE)
    plt.plot(arr)
    plt.show()

if __name__ == "__main__":
    main()