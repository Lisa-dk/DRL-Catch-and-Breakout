The requirements are provided. To setup the packages, run:

pip install -r requirements.txt

Afterwards, to train the DQN agents, simply run:

python main.py

This will begin training 5 DQN agents. Each of the agents will be evaluated every 10 episodes by following a greedy policy for 10 more episodes, and the mean reward will be displayed. Furthermore, after each agent finishes training, the evaluation rewards will be saved in a npy file.
