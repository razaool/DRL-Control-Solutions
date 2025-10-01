# DRL-Control-Solutions

Deep Reinforcement Learning for Control Problems: A hands-on implementation of Deep Q-Networks (DQN) to solve classic control tasks like CartPole and Lunar Lander.

## Project Overview

This project implements a complete DQN agent with:
- Deep Q-Network (DQN) with target network
- Experience Replay Buffer for stable training
- Epsilon-Greedy Exploration with decay
- Modular Architecture for easy environment swapping
- Comprehensive Training Metrics and visualization

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train a DQN agent on CartPole:

```bash
python train.py
```

For Lunar Lander:

```bash
python train_lunarlander.py
```

Training will:
- Save checkpoints every 50 episodes to `models/`
- Generate training plots (rewards, loss, evaluation metrics)
- Print progress periodically

### Demonstration

Visualize a trained agent:

For CartPole:
```bash
python demo.py models/CartPole-v1_<timestamp>/final_model.pt --episodes 5
```

For Lunar Lander:
```bash
python demo_lunarlander.py models/LunarLander-v3_<timestamp>/final_model.pt --episodes 5
```

Add `--no-render` flag to run without visualization and see statistics only.

## Project Structure

```
DRL-Control-Solutions/
├── agents/
│   └── dqn_agent.py              # DQN agent with target network
├── utils/
│   ├── networks.py                # Neural network architectures
│   └── replay_buffer.py           # Experience replay buffer
├── configs/
│   ├── cartpole_config.py         # CartPole hyperparameters
│   └── lunarlander_config.py     # Lunar Lander hyperparameters
├── models/                        # Saved model checkpoints
├── logs/                          # Training logs
├── train.py                       # CartPole training script
├── train_lunarlander.py           # Lunar Lander training script
├── demo.py                        # CartPole visualization script
├── demo_lunarlander.py            # Lunar Lander visualization script
└── requirements.txt               # Dependencies
```

## Technical Details

### DQN Components

1. **Q-Network**: Multi-layer perceptron that maps states to Q-values for each action
2. **Target Network**: Stabilizes training by providing consistent target values
3. **Experience Replay**: Stores transitions and samples randomly to break correlations
4. **Epsilon-Greedy**: Balances exploration (random actions) vs exploitation (best known action)

### Algorithm Overview

The agent learns through the following process:
1. Observe current state from environment
2. Select action using epsilon-greedy strategy
3. Execute action and receive reward and next state
4. Store transition in replay buffer
5. Sample random batch from buffer
6. Compute loss using Bellman equation
7. Update Q-network via backpropagation
8. Periodically sync target network

### Key Hyperparameters

#### CartPole Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.995 | Epsilon decay per episode |
| `batch_size` | 64 | Replay buffer sample size |
| `buffer_capacity` | 100,000 | Max replay buffer size |
| `target_update_freq` | 10 | Steps between target network updates |
| `hidden_dims` | [128, 128] | Network architecture |

#### Lunar Lander Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 5e-4 | Learning rate (lower for stability) |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.998 | Epsilon decay per episode (slower) |
| `batch_size` | 128 | Replay buffer sample size (larger) |
| `buffer_capacity` | 200,000 | Max replay buffer size (larger) |
| `target_update_freq` | 100 | Steps between target network updates (less frequent) |
| `hidden_dims` | [256, 256] | Network architecture (larger) |

Edit configuration files in `configs/` to customize hyperparameters.

## Expected Results

### CartPole-v1
- **Solved Criterion**: Average reward ≥195 over 100 episodes
- **Typical Training Time**: 200-300 episodes
- **Expected Performance**: Consistent 500-step episodes (maximum possible)
- **Final Performance**: 500/500 average reward

### LunarLander-v3
- **Solved Criterion**: Average reward ≥200 over 100 episodes
- **Typical Training Time**: 1400-2000 episodes
- **Expected Performance**: 200-280 average reward
- **Final Performance**: 265-275 average reward

## Environments

### Currently Implemented

**CartPole-v1**: Balance a pole on a moving cart
- State Space: 4 dimensions (position, velocity, angle, angular velocity)
- Action Space: 2 discrete actions (left, right)
- Episode Termination: Pole angle > 15° or cart position > 2.4

**LunarLander-v3**: Land a spacecraft safely on a designated pad
- State Space: 8 dimensions (x, y, velocities, angle, angular velocity, leg contacts)
- Action Space: 4 discrete actions (nothing, left engine, main engine, right engine)
- Reward Structure: Complex multi-objective (landing position, velocity, angle, fuel efficiency)

## Potential Extensions

The following improvements can be implemented to enhance performance:
- **Double DQN**: Reduces overestimation bias in Q-value predictions
- **Dueling DQN**: Separates value and advantage streams for better value estimation
- **Prioritized Experience Replay**: Samples important transitions more frequently
- **Noisy Networks**: Implements learnable exploration parameters
- **Rainbow DQN**: Combines multiple DQN improvements for state-of-the-art performance

## Monitoring Training

Training metrics are automatically saved to `models/<env>_<timestamp>/`:
- `training_results.png`: Visualization of reward curves, evaluation performance, and loss
- `checkpoint_ep*.pt`: Periodic model checkpoints for recovery and analysis
- `final_model.pt`: Final trained model weights

### Training Visualizations

The training process generates comprehensive plots showing:
- **Training Rewards**: Episode rewards over time with moving average
- **Evaluation Rewards**: Performance without exploration (epsilon=0) 
- **Training Loss**: MSE loss between predicted and target Q-values
- **Epsilon Decay**: Exploration rate reduction over episodes

These visualizations help monitor learning progress and identify training issues. See the generated `training_results.png` files in your model directories for examples of successful training curves.

## Troubleshooting

### Agent Not Learning
- Verify epsilon decay schedule - agent may be exploring excessively or insufficiently
- Increase replay buffer capacity or extend training duration
- Adjust learning rate (try 5e-4 or 1e-4 for more stability)
- Ensure sufficient batch size for stable gradient estimates

### Training Instability
- Increase target network update frequency
- Reduce learning rate to prevent large weight updates
- Implement gradient clipping in `dqn_agent.py` (recommended: clip to [-1, 1])
- Increase batch size for smoother gradient estimates

### Poor Generalization
- Increase replay buffer size to improve sample diversity
- Extend exploration phase (slower epsilon decay)
- Verify network architecture is sufficiently large for state complexity

## Mathematical Foundation

The agent optimizes the action-value function Q(s,a) based on the Bellman optimality equation:

```
Q*(s, a) = E[r + γ · max Q*(s', a')]
                    a'
```

Where:
- s: current state
- a: action taken
- r: immediate reward
- γ: discount factor
- s': next state
- a': next action

The neural network approximates Q(s,a;θ) and is trained to minimize the temporal difference error:

```
L(θ) = E[(Q(s,a;θ) - y)²]
```

Where the target y is computed using the target network:

```
y = r + γ · max Q(s', a'; θ')
             a'
```

## References

- Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529-533.
- Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep reinforcement learning with double Q-learning." AAAI.
- Wang, Z., et al. (2016). "Dueling network architectures for deep reinforcement learning." ICML.
- Schaul, T., et al. (2015). "Prioritized experience replay." ICLR.

## Additional Resources

- [OpenAI Spinning Up - DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Sutton & Barto - Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html)

## License

MIT License - This code is provided for educational and research purposes.