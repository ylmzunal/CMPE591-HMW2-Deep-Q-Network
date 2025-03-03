import torch
import numpy as np
import os
from dqn_model import DQN, select_action
from homework2 import Hw2Env  # Simülasyon ortamı


env = Hw2Env(n_actions=8, render_mode="gui")


sample_state = env.high_level_state()
input_dim = sample_state.shape[0]  # Should be 6
output_dim = env._n_actions  # Number of actions (8)

#
q_network = DQN(input_dim=input_dim, output_dim=output_dim)

# Check if model file exists
model_path = "trained_model.pt"
if os.path.exists(model_path):
    # Load with weights_only=True to avoid warning
    q_network.load_state_dict(torch.load(model_path, weights_only=True))
    q_network.eval()
    print(f"Loaded model from {model_path}")
else:
    print(f"Warning: Model file '{model_path}' not found!")
    print("Please run train.py first to train the model.")
    print("Running with untrained model for demonstration...")

num_test_episodes = 5  # Reduced for quick testing

for episode in range(num_test_episodes):
    state = env.reset()
    state = env.high_level_state()  # Use high-level state for consistency
    
    done = False
    cumulative_reward = 0.0

    while not done:
        action = select_action(state, epsilon=0.05, action_dim=output_dim, q_network=q_network)
        _, reward, terminal, truncated = env.step(action)
        next_state = env.high_level_state()
        
        done = terminal or truncated
        state = next_state
        cumulative_reward += reward

    print(f"Episode {episode}: Total Reward = {cumulative_reward:.4f}")