import numpy as np

from homework2 import Hw2Env

N_ACTIONS = 8
env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
for episode in range(10):
    env.reset()
    done = False
    cum_reward = 0.0
    while not done:
        action = np.random.randint(N_ACTIONS)
        state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        cum_reward += reward
    print(f"Episode={episode}, reward={cum_reward}")