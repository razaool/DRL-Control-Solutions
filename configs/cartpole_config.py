config = {
    # Environment
    'env_name': 'CartPole-v1',
    
    # Network architecture
    'hidden_dims': [128, 128],  # Two hidden layers with 128 neurons each
    'lr': 1e-3,  # Learning rate
    'gamma': 0.99,  # Discount factor for future rewards
    
    # Exploration strategy
    'epsilon_start': 1.0,  # Start with 100% random actions
    'epsilon_end': 0.01,  # End with 1% random actions
    'epsilon_decay': 0.995,  # Decay rate per episode
    
    # Experience replay
    'buffer_capacity': 100000,  # Max memories to store
    'batch_size': 64,  # Samples per training step
    
    # Target network
    'target_update_freq': 10,  # Update target network every N steps
    
    # Training
    'num_episodes': 500,  # Total episodes to train
    'max_steps': 500,  # Max steps per episode
    
    # Logging and saving
    'save_freq': 50,  # Save checkpoint every N episodes
    'eval_freq': 10,  # Evaluate every N episodes
    'eval_episodes': 5,  # Number of episodes per evaluation
    
    # Hardware
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
}
