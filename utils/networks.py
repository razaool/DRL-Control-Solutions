import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Neural network that estimates Q-values for each action"""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(QNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Build hidden layers dynamically
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))  # Fully connected layer
            layers.append(nn.ReLU())  # Activation function
            input_dim = hidden_dim
        
        # Output layer: one Q-value per action
        layers.append(nn.Linear(input_dim, action_dim))
        
        # Combine all layers into sequential network
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass: state -> Q-values"""
        return self.network(x)
