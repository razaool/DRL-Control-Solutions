import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import os

from agents.dqn_agent import DQNAgent
from configs.cartpole_config import config


def evaluate_agent(agent, env_name, num_episodes=5):
    env = gym.make(env_name)
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    env.close()
    return np.mean(total_rewards)


def train():
    env = gym.make(config['env_name'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=config['hidden_dims'],
        lr=config['lr'],
        gamma=config['gamma'],
        epsilon_start=config['epsilon_start'],
        epsilon_end=config['epsilon_end'],
        epsilon_decay=config['epsilon_decay'],
        buffer_capacity=config['buffer_capacity'],
        batch_size=config['batch_size'],
        target_update_freq=config['target_update_freq'],
        device=config['device']
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models/{config['env_name']}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    eval_rewards = []
    losses = []
    
    print(f"Training DQN on {config['env_name']}")
    print(f"Device: {config['device']}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print("-" * 60)
    
    for episode in tqdm(range(config['num_episodes']), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        episode_losses = []
        
        for step in range(config['max_steps']):
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        agent.decay_epsilon()
        
        episode_rewards.append(episode_reward)
        if episode_losses:
            losses.append(np.mean(episode_losses))
        
        if (episode + 1) % config['eval_freq'] == 0:
            eval_reward = evaluate_agent(agent, config['env_name'], config['eval_episodes'])
            eval_rewards.append(eval_reward)
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"\nEpisode {episode + 1}/{config['num_episodes']}")
            print(f"Avg Reward (last 10): {avg_reward:.2f}")
            print(f"Eval Reward: {eval_reward:.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            if losses:
                print(f"Avg Loss: {np.mean(losses[-10:]):.4f}")
        
        if (episode + 1) % config['save_freq'] == 0:
            agent.save(f"{save_dir}/checkpoint_ep{episode + 1}.pt")
    
    agent.save(f"{save_dir}/final_model.pt")
    env.close()
    
    plot_training_results(episode_rewards, eval_rewards, losses, save_dir)
    
    print(f"\nTraining complete!")
    print(f"Models saved to {save_dir}/")


def plot_training_results(episode_rewards, eval_rewards, losses, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(episode_rewards, alpha=0.6, label='Episode Reward')
    if len(episode_rewards) >= 10:
        smoothed = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        axes[0].plot(range(9, len(episode_rewards)), smoothed, label='Smoothed (10 ep)', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Rewards')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if eval_rewards:
        eval_episodes = [i * config['eval_freq'] for i in range(1, len(eval_rewards) + 1)]
        axes[1].plot(eval_episodes, eval_rewards, marker='o', linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Average Reward')
        axes[1].set_title('Evaluation Rewards')
        axes[1].grid(True, alpha=0.3)
    
    if losses:
        axes[2].plot(losses, alpha=0.6)
        if len(losses) >= 10:
            smoothed = np.convolve(losses, np.ones(10)/10, mode='valid')
            axes[2].plot(range(9, len(losses)), smoothed, linewidth=2)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Training Loss')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_results.png", dpi=150)
    print(f"Training plots saved to {save_dir}/training_results.png")


if __name__ == "__main__":
    train()
