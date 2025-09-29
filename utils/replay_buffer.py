import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """Store and sample agent experiences for training"""
    
    def __init__(self, capacity):
        # Deque automatically removes oldest when full
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store one transition"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Randomly sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        # Unzip batch into separate arrays
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        """Current buffer size"""
        return len(self.buffer)
