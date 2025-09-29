import gymnasium as gym
import torch
import argparse
import time

from agents.dqn_agent import DQNAgent
from configs.lunarlander_config import config


def demo(model_path, num_episodes=5, render=True):
    """Run and visualize a trained agent"""
    env_name = config['env_name']
    
    # Create environment with or without rendering
    if render:
        env = gym.make(env_name, render_mode='human')  # Shows GUI
    else:
        env = gym.make(env_name)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Create agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config['hidden_dims'],
        device=config['device']
    )
    
    # Load trained model
    agent.load(model_path)
    print(f"Loaded model from {model_path}")
    print(f"Running {num_episodes} episodes...\n")
    
    total_rewards = []
    
    # Run episodes
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Agent chooses action (no exploration)
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1
            
            # Slow down rendering for visibility
            if render:
                time.sleep(0.02)
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    env.close()
    
    # Print summary statistics
    print(f"\nAverage Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Max Reward: {max(total_rewards):.2f}")
    print(f"Min Reward: {min(total_rewards):.2f}")
    
    if sum(total_rewards) / len(total_rewards) >= 200:
        print("✅ SOLVED! Average reward >= 200")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Demo a trained DQN Lunar Lander agent')
    parser.add_argument('model_path', type=str, help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to run')
    parser.add_argument('--no-render', action='store_true', help='Disable rendering')
    
    args = parser.parse_args()
    
    # Run demo
    demo(args.model_path, num_episodes=args.episodes, render=not args.no_render)
