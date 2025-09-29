config = {
    # Environment
    'env_name': 'LunarLander-v3',
    
    # Network architecture - larger for more complex problem
    'hidden_dims': [256, 256],  # Bigger network for 8D state space
    'lr': 5e-4,  # Lower learning rate for stability
    'gamma': 0.99,  # Discount factor for future rewards
    
    # Exploration strategy - slower decay for harder problem
    'epsilon_start': 1.0,  # Start with 100% random actions
    'epsilon_end': 0.01,  # End with 1% random actions
    'epsilon_decay': 0.998,  # Slower decay (was 0.995 for CartPole)
    
    # Experience replay - larger buffer for complex environment
    'buffer_capacity': 200000,  # More memories (was 100k for CartPole)
    'batch_size': 128,  # Larger batch size
    
    # Target network
    'target_update_freq': 100,  # Update less frequently for stability
    
    # Training - much longer for harder problem
    'num_episodes': 2000,  # 4x more episodes than CartPole
    'max_steps': 1000,  # Longer episodes possible
    
    # Logging and saving
    'save_freq': 100,  # Save less frequently
    'eval_freq': 20,  # Evaluate less frequently
    'eval_episodes': 5,  # Number of episodes per evaluation
    
    # Hardware
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
}
