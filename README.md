# DRL-Control-Solutions

Deep Reinforcement Learning for Control Problems: A hands-on implementation of Deep Q-Networks (DQN) to solve classic control tasks like CartPole and Lunar Lander.

## 🎯 Project Overview

This project implements a complete DQN agent with:
- **Deep Q-Network (DQN)** with target network
- **Experience Replay Buffer** for stable training
- **Epsilon-Greedy Exploration** with decay
- **Modular Architecture** for easy environment swapping
- **Comprehensive Training Metrics** and visualization

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train a DQN agent on CartPole:

```bash
python train.py
```

Training will:
- Save checkpoints every 50 episodes to `models/`
- Generate training plots (rewards, loss, evaluation metrics)
- Print progress every 10 episodes

### Demo

Visualize a trained agent:

```bash
python demo.py models/CartPole-v1_<timestamp>/final_model.pt --episodes 5
```

## 📁 Project Structure

```
DRL-Control-Solutions/
├── agents/
│   └── dqn_agent.py          # DQN agent with target network
├── utils/
│   ├── networks.py            # Neural network architectures
│   └── replay_buffer.py       # Experience replay buffer
├── configs/
│   └── cartpole_config.py     # Hyperparameters
├── models/                    # Saved model checkpoints
├── logs/                      # Training logs
├── train.py                   # Training script
├── demo.py                    # Visualization script
└── requirements.txt           # Dependencies
```

## 🧠 Technical Details

### DQN Components

1. **Q-Network**: Multi-layer perceptron that maps states to Q-values for each action
2. **Target Network**: Stabilizes training by providing consistent target values
3. **Experience Replay**: Stores transitions and samples randomly to break correlations
4. **Epsilon-Greedy**: Balances exploration (random actions) vs exploitation (best known action)

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_decay` | 0.995 | Epsilon decay per episode |
| `batch_size` | 64 | Replay buffer sample size |
| `buffer_capacity` | 100,000 | Max replay buffer size |
| `target_update_freq` | 10 | Steps between target network updates |

Edit `configs/cartpole_config.py` to customize.

## 📊 Expected Results

### CartPole-v1
- **Solved**: Average reward ≥195 over 100 episodes
- **Typical training time**: 200-300 episodes
- **Success indicator**: Consistent 500-step episodes

## 🎮 Environments

### Currently Implemented
- **CartPole-v1**: Balance a pole on a moving cart

### Coming Soon
- **LunarLander-v2**: Land a spacecraft safely on a pad

To add Lunar Lander:
1. Create `configs/lunarlander_config.py`
2. Change `env_name` to `'LunarLander-v2'`
3. Adjust hyperparameters (may need larger network, more episodes)

## 🛠️ Extensions

Potential improvements to implement:
- [ ] **Double DQN**: Reduce overestimation bias
- [ ] **Dueling DQN**: Separate value and advantage streams
- [ ] **Prioritized Experience Replay**: Sample important transitions more frequently
- [ ] **Noisy Networks**: Learnable exploration
- [ ] **Rainbow DQN**: Combine all improvements

## 📈 Monitoring Training

Training metrics saved to `models/<env>_<timestamp>/`:
- `training_results.png`: Reward and loss curves
- `checkpoint_ep*.pt`: Periodic model checkpoints
- `final_model.pt`: Final trained model

## 🔧 Troubleshooting

**Agent not learning?**
- Check epsilon decay - agent may be exploring too much/too little
- Increase buffer size or training episodes
- Adjust learning rate

**Training unstable?**
- Increase target network update frequency
- Reduce learning rate
- Add gradient clipping (modify `dqn_agent.py`)

## 📚 Learning Resources

- [Mnih et al. (2015) - Original DQN Paper](https://www.nature.com/articles/nature14236)
- [OpenAI Spinning Up - DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 📝 License

MIT License - Feel free to use this code for learning and experimentation!