import numpy as np
import matplotlib.pyplot as plt
import os, re

file_name = "csv_a2c_10m_0.0007.csv"

rewards = []
timesteps = []
with open(file_name, mode='r') as csv_file:
    results = [lines.split(',') for lines in csv_file.readlines()]
    results = results[1:]
    for res in results:
        rewards.append(float(res[-1].strip()))
        timesteps.append(int(res[1].strip())/1e6)


ax = plt.subplot()

ax.plot(timesteps,rewards, color='#1a9988')


# Axis visibility
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
# ax.grid(axis = 'y')

# Axis labels and ticks
ax.set_ylabel('Total Reward')
ax.set_xlabel('Timesteps (x 1e6)')
plt.title('Training Rewards Breakout')

ax.legend(bbox_to_anchor=(1, 1), loc='upper left', prop={'size': 11})
plt.tight_layout()

plt.show()
        
