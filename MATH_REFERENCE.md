# Mathematical Reference for DQN Implementation

This document contains all mathematical formulas, equations, and theoretical foundations used in the DRL-Control-Solutions repository.

---

## Table of Contents

1. [Core Concepts](#1-core-concepts)
2. [The Bellman Equation](#2-the-bellman-optimality-equation)
3. [Neural Network Approximation](#3-neural-network-approximation)
4. [Loss Function](#4-loss-function)
5. [Gradient Descent](#5-gradient-descent-update-rule)
6. [Epsilon-Greedy Strategy](#6-epsilon-greedy-action-selection)
7. [Network Architecture](#7-neural-network-forward-pass)
8. [Target Network](#8-target-network-update)
9. [Discount Factor](#9-discount-factor-gamma)
10. [Batch Training](#10-batch-training)
11. [Experience Replay](#11-experience-replay-sampling)
12. [Policy Derivation](#12-policy-decision-making)
13. [MSE Loss](#13-mean-squared-error-loss)
14. [TD Error](#14-temporal-difference-error)
15. [Complete Algorithm](#15-complete-training-algorithm)
16. [Additional Details](#16-additional-mathematical-details)

---

## 1. Core Concepts

### The Q-Function (Action-Value Function)

The fundamental concept - estimating expected future reward:

```
Q(s, a) = Expected total discounted reward starting from state s, taking action a
```

**Optimal Q-function:**
```
Q*(s, a) = max E[R_t | s_t = s, a_t = a]
           π
```

**Where:**
- `s` = state
- `a` = action  
- `R_t` = total discounted return from time t
- `π` = policy (strategy for choosing actions)

**Code Location:** This is what the entire `dqn_agent.py` learns to approximate.

---

## 2. The Bellman Optimality Equation

**THE fundamental equation of reinforcement learning:**

```
Q*(s, a) = E[r + γ · max Q*(s', a') | s, a]
                    a'
```

**In plain English:**
```
Q-value of (state, action) = immediate reward + discounted value of best next action
```

**Parameters:**
- `r` = immediate reward
- `γ` (gamma) = discount factor (0.99 in configs)
- `s'` = next state after taking action a
- `a'` = possible actions in next state
- `max Q*(s', a')` = value of best action in next state
      a'

**Code Implementation (`dqn_agent.py:88`):**
```python
target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
```

The `(1 - dones)` term ensures that when an episode ends, there's no future value added.

---

## 3. Neural Network Approximation

Since we can't store Q(s,a) for all state-action pairs (especially in continuous state spaces), we use a neural network:

```
Q(s, a; θ) ≈ Q*(s, a)
```

**Where:**
- `θ` = neural network parameters (weights and biases)
- `Q(s, a; θ)` = network's prediction of Q-value

**Network Parameter Counts:**
- **CartPole**: 4 → 128 → 128 → 2 ≈ 17,000 parameters
  - Input: 4D state (position, velocity, angle, angular velocity)
  - Output: 2D Q-values (left, right)

- **Lunar Lander**: 8 → 256 → 256 → 4 ≈ 135,000 parameters
  - Input: 8D state (x, y, velocities, angle, etc.)
  - Output: 4D Q-values (nothing, left, main, right)

---

## 4. Loss Function

**The TD (Temporal Difference) error that we minimize:**

```
L(θ) = E[(Q(s, a; θ) - y)²]
```

**Where the target y is defined as:**
```
y = r + γ · max Q(s', a'; θ⁻)
             a'
```

**Key Notation:**
- `θ` = Q-network parameters (updated every step)
- `θ⁻` = Target network parameters (updated every N steps)
- `E[·]` = expectation over batch of samples
- `y` = the "ground truth" target we're trying to match

**Code Implementation (`dqn_agent.py:91`):**
```python
loss = self.loss_fn(current_q_values, target_q_values)
# self.loss_fn = nn.MSELoss()
```

**Why squared error?** Penalizes large errors more heavily, smooth gradient.

---

## 5. Gradient Descent Update Rule

**How the network actually learns:**

```
θ ← θ - α · ∇_θ L(θ)
```

**Where:**
- `α` (alpha) = learning rate
  - CartPole: 1e-3 (0.001)
  - Lunar Lander: 5e-4 (0.0005)
- `∇_θ L(θ)` = gradient of loss with respect to parameters
- `←` = assignment/update operator

**Expanded gradient form:**
```
∇_θ L(θ) = E[(Q(s,a;θ) - y) · ∇_θ Q(s,a;θ)]
```

This is computed automatically by PyTorch's autograd.

**Code Implementation (`dqn_agent.py:93-95`):**
```python
self.optimizer.zero_grad()  # Clear old gradients
loss.backward()              # Compute ∇_θ L(θ) via backpropagation
self.optimizer.step()        # θ ← θ - α·∇_θ L(θ)
```

---

## 6. Epsilon-Greedy Action Selection

**Balancing exploration (trying new things) vs exploitation (using what we know):**

```
         ⎧ random action           with probability ε
a_t =    ⎨
         ⎩ argmax Q(s_t, a; θ)    with probability 1-ε
              a
```

**Epsilon decay schedule:**
```
ε_t = max(ε_end, ε_start · ε_decay^t)
```

**Configuration Values:**

| Environment | ε_start | ε_end | ε_decay | Episodes to reach ~0.01 |
|-------------|---------|-------|---------|-------------------------|
| CartPole    | 1.0     | 0.01  | 0.995   | ~920                    |
| Lunar Lander| 1.0     | 0.01  | 0.998   | ~2300                   |

**Why different decay rates?** Lunar Lander is harder and benefits from longer exploration.

**Code Implementation (`dqn_agent.py:50-60`):**
```python
def select_action(self, state, training=True):
    if training and np.random.rand() < self.epsilon:
        return np.random.randint(self.action_dim)  # Explore
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax(1).item()  # Exploit
```

**Epsilon decay (`dqn_agent.py:107`):**
```python
def decay_epsilon(self):
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

---

## 7. Neural Network Forward Pass

**Multi-layer Perceptron (MLP) architecture:**

```
Layer 1:  h₁ = ReLU(W₁·s + b₁)
Layer 2:  h₂ = ReLU(W₂·h₁ + b₂)
Output:   Q  = W₃·h₂ + b₃
```

**Where:**
- `s` = state vector (input)
- `W₁, W₂, W₃` = weight matrices
- `b₁, b₂, b₃` = bias vectors
- `ReLU(x) = max(0, x)` = Rectified Linear Unit activation
- `Q` = vector of Q-values (one per action)
- `·` = matrix multiplication

**Dimensional Analysis:**

**CartPole:**
```
Input → Hidden1 → Hidden2 → Output

s ∈ ℝ⁴  →  h₁ ∈ ℝ¹²⁸  →  h₂ ∈ ℝ¹²⁸  →  Q ∈ ℝ²

Weight shapes:
W₁: 128×4    (512 parameters)
b₁: 128      (128 parameters)
W₂: 128×128  (16,384 parameters)
b₂: 128      (128 parameters)
W₃: 2×128    (256 parameters)
b₃: 2        (2 parameters)
Total: 17,410 parameters
```

**Lunar Lander:**
```
Input → Hidden1 → Hidden2 → Output

s ∈ ℝ⁸  →  h₁ ∈ ℝ²⁵⁶  →  h₂ ∈ ℝ²⁵⁶  →  Q ∈ ℝ⁴

Weight shapes:
W₁: 256×8      (2,048 parameters)
b₁: 256        (256 parameters)
W₂: 256×256    (65,536 parameters)
b₂: 256        (256 parameters)
W₃: 4×256      (1,024 parameters)
b₃: 4          (4 parameters)
Total: 69,124 parameters
```

**Code Implementation (`networks.py:26`):**
```python
def forward(self, x):
    return self.network(x)
# Where self.network is Sequential(Linear, ReLU, Linear, ReLU, Linear)
```

---

## 8. Target Network Update

**Stabilization technique - periodically copy weights:**

```
θ⁻ ← θ    every N steps
```

**Update Frequencies:**
- CartPole: every 10 training steps
- Lunar Lander: every 100 training steps

**Why?** Without target network, we're "chasing a moving target":
```
Bad:  y = r + γ·max Q(s', a'; θ)     ← target changes every step
              a'

Good: y = r + γ·max Q(s', a'; θ⁻)    ← target stable for N steps
              a'
```

**Code Implementation (`dqn_agent.py:99-100`):**
```python
def update_target_network(self):
    self.target_network.load_state_dict(self.q_network.state_dict())
```

**Called from (`dqn_agent.py:98-99`):**
```python
if self.steps % self.target_update_freq == 0:
    self.update_target_network()
```

---

## 9. Discount Factor (Gamma)

**How much we value future rewards:**

```
Total Return: R_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + γ³·r_{t+3} + ...

Compact form: R_t = Σ γᵏ·r_{t+k}
                    k=0
```

**With γ = 0.99 (both environments):**

| Steps Ahead | Discount Factor | Effective Weight |
|-------------|-----------------|------------------|
| 0 (now)     | γ⁰ = 1.00       | 100%            |
| 1           | γ¹ = 0.99       | 99%             |
| 10          | γ¹⁰ = 0.904     | 90%             |
| 50          | γ⁵⁰ = 0.605     | 61%             |
| 100         | γ¹⁰⁰ = 0.366    | 37%             |
| 200         | γ²⁰⁰ = 0.134    | 13%             |
| 500         | γ⁵⁰⁰ = 0.007    | 0.7%            |

**Interpretation:** With γ=0.99, the agent effectively plans ~100-200 steps ahead.

**Why not γ=1.0?** Would give equal weight to all future rewards, making learning unstable and infinite-horizon problems ill-defined.

**Why not γ=0.9?** Would only care about ~10-20 steps ahead, too short-sighted for these problems.

---

## 10. Batch Training

**Mini-batch gradient descent for stability:**

```
L(θ) = (1/B) Σ (Q(s_i, a_i; θ) - y_i)²
             i=1

Where: y_i = r_i + γ · max Q(s'_i, a'; θ⁻)
                       a'
```

**Batch Sizes:**
- CartPole: B = 64
- Lunar Lander: B = 128

**Why batching?**
1. **Smoother gradients**: Average over multiple samples reduces variance
2. **Computational efficiency**: GPU parallelization
3. **Better generalization**: Less overfitting to individual samples

**Alternative (online learning):** B = 1, but very noisy and unstable.

---

## 11. Experience Replay Sampling

**Breaking temporal correlation by random sampling:**

```
Replay Buffer: D = {(s₁,a₁,r₁,s'₁,done₁), ..., (s_N,a_N,r_N,s'_N,done_N)}

Sample batch: B ~ Uniform(D)
```

**Buffer Capacities:**
- CartPole: N = 100,000 transitions
- Lunar Lander: N = 200,000 transitions

**Sampling Distribution:**
```
P(transition_i ∈ B) = 1/N  for all i
```

**Why uniform random?** 
- Breaks correlation between consecutive experiences
- Reuses data multiple times (sample efficiency)
- Prevents catastrophic forgetting

**Code Implementation (`dqn_agent.py:73`):**
```python
states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
```

**Storage (`replay_buffer.py`):**
```python
def push(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))
```

Uses `deque(maxlen=capacity)` which automatically removes oldest when full (FIFO).

---

## 12. Policy (Decision Making)

**The learned policy derived from Q-values:**

**During evaluation (exploitation only):**
```
π(s) = argmax Q(s, a; θ)
         a
```

**During training (with exploration):**
```
π_ε(s) = { argmax Q(s, a; θ)  with probability 1-ε
         {   a
         { random action       with probability ε
```

**Optimal policy theorem:**
If Q-values converge to Q*, then the greedy policy is optimal:
```
π*(s) = argmax Q*(s, a)
          a
```

---

## 13. Mean Squared Error Loss

**The specific loss function used:**

```
MSE = (1/B) Σ (prediction_i - target_i)²
            i=1

Expanded:
MSE = (1/B) Σ (Q(s_i,a_i;θ) - [r_i + γ·max Q(s'_i,a';θ⁻)])²
            i=1                        a'
```

**Gradient for backpropagation:**
```
∂MSE/∂θ = (2/B) Σ (Q(s_i,a_i;θ) - target_i) · ∂Q(s_i,a_i;θ)/∂θ
                i=1
```

PyTorch computes this automatically via `loss.backward()`.

**Alternative loss functions:**
- Huber loss (more robust to outliers)
- MAE (Mean Absolute Error)

MSE is standard for DQN and works well in practice.

---

## 14. Temporal Difference Error

**The error signal that drives learning:**

```
TD Error: δ = Q(s,a;θ) - TD_target

Where: TD_target = r + γ·max Q(s',a';θ⁻)
                          a'
```

**Interpretation:**
- `δ > 0`: Network **overestimated** → reduce Q(s,a)
- `δ < 0`: Network **underestimated** → increase Q(s,a)
- `δ ≈ 0`: Network accurate → minimal update

**Expected behavior during training:**
- Early episodes: Large |δ| (poor predictions)
- Late episodes: Small |δ| (converged)

**This is exactly what the loss measures:** MSE = E[δ²]

---

## 15. Complete Training Algorithm

**Full DQN pseudocode with mathematical notation:**

```
Algorithm: Deep Q-Network (DQN)

Input:
  - Environment with states S, actions A, rewards R
  - Hyperparameters: α (learning rate), γ (discount), ε_start, ε_end, ε_decay
  - Network architecture, batch size B, buffer capacity N, target update freq T

Initialize:
  - Q-network Q(s,a;θ) with random weights θ
  - Target network Q(s,a;θ⁻) where θ⁻ ← θ
  - Replay buffer D ← ∅ (empty)
  - Exploration rate ε ← ε_start
  - Step counter t ← 0

For episode = 1 to num_episodes:
    
    s ← environment.reset()
    
    For step = 1 to max_steps:
        
        # Action Selection (Epsilon-Greedy)
        With probability ε:
            a ~ Uniform(A)                    # Random action (explore)
        Otherwise:
            a ← argmax_a' Q(s, a'; θ)        # Best action (exploit)
        
        # Environment Interaction
        s', r, done ← environment.step(a)
        
        # Store Transition
        D ← D ∪ {(s, a, r, s', done)}
        
        # Experience Replay Learning
        If |D| ≥ B:                           # Wait until buffer has enough samples
            
            # Sample Mini-Batch
            {(s_i, a_i, r_i, s'_i, done_i)}_{i=1}^B ~ Uniform(D)
            
            # Compute Target Q-Values
            For each i ∈ {1, ..., B}:
                If done_i:
                    y_i ← r_i                                    # Terminal state
                Else:
                    y_i ← r_i + γ · max_{a'} Q(s'_i, a'; θ⁻)    # Bellman target
            
            # Compute Current Q-Values
            For each i ∈ {1, ..., B}:
                q_i ← Q(s_i, a_i; θ)
            
            # Compute Loss
            L(θ) ← (1/B) Σ_{i=1}^B (q_i - y_i)²
            
            # Gradient Descent Update
            θ ← θ - α · ∇_θ L(θ)
            
            # Increment Step Counter
            t ← t + 1
            
            # Periodic Target Network Update
            If t mod T == 0:
                θ⁻ ← θ
        
        # State Transition
        s ← s'
        
        # Check Episode Termination
        If done:
            Break
    
    # Decay Exploration Rate
    ε ← max(ε_end, ε · ε_decay)

Return: Trained Q-network Q(s,a;θ)
```

**This is exactly what `train.py` and `train_lunarlander.py` implement.**

---

## 16. Additional Mathematical Details

### ReLU Activation Function

**Definition:**
```
ReLU(x) = max(0, x) = { x  if x > 0
                      { 0  if x ≤ 0
```

**Derivative (for backpropagation):**
```
d/dx ReLU(x) = { 1  if x > 0
               { 0  if x ≤ 0
```

**Why ReLU?**
- Simple and fast to compute
- Avoids vanishing gradient problem
- Empirically works well for deep networks

### Adam Optimizer

**Adaptive Moment Estimation update rule:**

```
# First moment (mean of gradients)
m_t ← β₁·m_{t-1} + (1-β₁)·g_t

# Second moment (variance of gradients)
v_t ← β₂·v_{t-1} + (1-β₂)·g_t²

# Bias correction
m̂_t ← m_t / (1 - β₁^t)
v̂_t ← v_t / (1 - β₂^t)

# Parameter update
θ_t ← θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

**Default hyperparameters:**
- `β₁ = 0.9` (momentum)
- `β₂ = 0.999` (RMSProp-like)
- `ε = 1e-8` (numerical stability)

**Why Adam?** Adaptive learning rates per parameter, combines benefits of momentum and RMSProp.

### Convergence Theory

**Q-Learning convergence theorem (Watkins & Dayan, 1992):**

Under the following conditions:
1. All state-action pairs visited infinitely often
2. Learning rate satisfies: Σ α_t = ∞ and Σ α_t² < ∞
3. Rewards are bounded

Then: Q(s,a) → Q*(s,a) with probability 1

**DQN modifications:**
- Uses function approximation (weaker guarantees)
- Fixed learning rate (violates condition 2)
- Experience replay (alters sample distribution)

Despite theoretical limitations, DQN works well in practice.

---

## Hyperparameter Summary

### CartPole Configuration

```
State Space:      s ∈ ℝ⁴
Action Space:     a ∈ {0, 1}
Network:          4 → 128 → 128 → 2
Parameters:       θ ∈ ℝ¹⁷⁴¹⁰

Learning rate:    α = 1×10⁻³
Discount:         γ = 0.99
Epsilon start:    ε₀ = 1.0
Epsilon end:      ε_∞ = 0.01
Epsilon decay:    ε_decay = 0.995
Batch size:       B = 64
Buffer capacity:  N = 100,000
Target update:    T = 10 steps
Episodes:         M = 500
```

### Lunar Lander Configuration

```
State Space:      s ∈ ℝ⁸
Action Space:     a ∈ {0, 1, 2, 3}
Network:          8 → 256 → 256 → 4
Parameters:       θ ∈ ℝ⁶⁹¹²⁴

Learning rate:    α = 5×10⁻⁴
Discount:         γ = 0.99
Epsilon start:    ε₀ = 1.0
Epsilon end:      ε_∞ = 0.01
Epsilon decay:    ε_decay = 0.998
Batch size:       B = 128
Buffer capacity:  N = 200,000
Target update:    T = 100 steps
Episodes:         M = 2000
```

---

## Code-to-Math Mapping

**Quick reference table:**

| Math Notation | Code Variable | File:Line |
|---------------|---------------|-----------|
| Q(s,a;θ) | `self.q_network(state)` | dqn_agent.py:59 |
| Q(s,a;θ⁻) | `self.target_network(next_states)` | dqn_agent.py:87 |
| θ | `self.q_network.parameters()` | dqn_agent.py:43 |
| α | `lr` | configs/*_config.py |
| γ | `self.gamma` | dqn_agent.py:26 |
| ε | `self.epsilon` | dqn_agent.py:27 |
| B | `self.batch_size` | dqn_agent.py:29 |
| N | `buffer_capacity` | configs/*_config.py |
| T | `self.target_update_freq` | dqn_agent.py:30 |
| L(θ) | `loss` | dqn_agent.py:91 |
| ∇_θ L(θ) | `loss.backward()` | dqn_agent.py:94 |
| argmax_a Q(s,a) | `q_values.argmax()` | dqn_agent.py:60 |

---

## References

**Original DQN Paper:**
Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

**Key Equation from Paper:**
```
L(θ) = E_{(s,a,r,s')~D}[(r + γ max_{a'} Q(s',a';θ⁻) - Q(s,a;θ))²]
```

**Bellman Optimality:**
Bellman, R. (1957). "Dynamic Programming." Princeton University Press.

**Q-Learning Convergence:**
Watkins, C. J., & Dayan, P. (1992). "Q-learning." *Machine Learning*, 8(3-4), 279-292.

**Adam Optimizer:**
Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." *arXiv preprint arXiv:1412.6980*.

---

## End Notes

This document provides the complete mathematical foundation for understanding the DQN implementation in this repository. Every equation listed here is actually used in the code, and the code locations are provided for reference.

For blog post writing, consider:
1. Starting with intuitive explanations before equations
2. Using visual diagrams alongside math
3. Providing numerical examples with actual values
4. Showing the progression from simple Q-learning to deep Q-learning
5. Including training curves that show these equations in action

The math here is rigorous but accessible - suitable for a technical blog audience with basic calculus and linear algebra background.

