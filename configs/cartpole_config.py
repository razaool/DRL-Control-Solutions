config = {
    'env_name': 'CartPole-v1',
    
    'hidden_dims': [128, 128],
    'lr': 1e-3,
    'gamma': 0.99,
    
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    
    'buffer_capacity': 100000,
    'batch_size': 64,
    
    'target_update_freq': 10,
    
    'num_episodes': 500,
    'max_steps': 500,
    
    'save_freq': 50,
    'eval_freq': 10,
    'eval_episodes': 5,
    
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
}
