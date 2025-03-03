import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt  
from dqn_model import DQN, select_action
from replay_buffer import ReplayBuffer
from homework2 import Hw2Env  

# Ortamı başlat
env = Hw2Env(n_actions=8, render_mode="offscreen")

# For DQN, we'll use the high-level state which is more suitable than raw pixels
sample_state = env.high_level_state()
input_dim = sample_state.shape[0]  # Should be 6 (2D positions of end effector, object, and goal)
output_dim = env._n_actions  # Number of discrete actions (8)

q_network = DQN(input_dim=input_dim, output_dim=output_dim)
target_q_network = DQN(input_dim=input_dim, output_dim=output_dim)
target_q_network.load_state_dict(q_network.state_dict())  # Başlangıçta kopya al

optimizer = optim.Adam(q_network.parameters(), lr=0.0001)
replay_buffer = ReplayBuffer()
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.05
batch_size = 64
update_target_freq = 200
update_frequency = 10 
num_episodes = 10000  # More episodes for better learning


reward_history = []
rps_history = []

def train_dqn():
    global epsilon
    for episode in range(num_episodes):
        state = env.reset()
        state = env.high_level_state()  # Use high-level state (vectorized representation)
        
        done = False
        total_reward = 0  
        episode_steps = 0 

        while not done:
            action = select_action(state, epsilon, output_dim, q_network)
            _, reward, terminal, truncated = env.step(action)
            next_state = env.high_level_state()  # Get high-level state after step
            
            done = terminal or truncated  # Either terminal or truncated ends the episode
            
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward 
            episode_steps += 1 
            
            if replay_buffer.size() >= batch_size and episode_steps % update_frequency == 0:
                train_step()

        reward_history.append(total_reward)
        rps = total_reward / episode_steps if episode_steps > 0 else 0  
        rps_history.append(rps)
        print(f"Episode {episode}: Total Reward = {total_reward}, RPS = {rps:.4f}, Steps = {episode_steps}, Done = {done}")

        if episode % update_target_freq == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    
    plot_training_results()
    
    # Save the trained model
    torch.save(q_network.state_dict(), "trained_model.pt")
    print("Model saved to trained_model.pt")

def train_step():
    batch = replay_buffer.sample(batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

    q_values = q_network(states).gather(1, actions)

    with torch.no_grad():
        max_next_q_values = target_q_network(next_states).max(dim=1, keepdim=True)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)

    loss = torch.nn.MSELoss()(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def plot_training_results():
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label="Total Reward per Episode", color="blue")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress: Reward vs. Episode")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(rps_history, label="Reward per Step (RPS)", color="green")
    plt.xlabel("Episode")
    plt.ylabel("RPS")
    plt.title("Training Progress: Reward per Step vs. Episode")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("training_results.png")  

if __name__ == "__main__":
    train_dqn()