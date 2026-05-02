# Lab 8 - Option A: Comprehensive Algorithm Comparison (Temporal Difference Learning)
## MountainCar-v0, SARSA, Q-Learning & Dyna-Q

## Overview

This project explores **Temporal-Difference (TD) learning** methods for solving **reinforcement learning problems** in a continuous state environment using function approximation, discretization, and model-based planning.

### 1. Environment: MountainCar-v0
- Continuous state space:
  - Position ∈ [-1.2, 0.6]  
  - Velocity ∈ [-0.07, 0.07]
- Discrete action space:
  - Push left, no push, push right
- Sparse reward structure:
  - Reward = -1 per step until goal is reached
- Objective:
  - Learn a policy to reach the goal state efficiently

---

### 2. Reinforcement Learning Algorithms

Three algorithms from different RL families were implemented:

- **SARSA with Tile Coding**
  - On-policy TD control
  - Uses linear function approximation
  - Generalizes across continuous states

- **Tabular Q-Learning**
  - Off-policy TD control
  - Uses discretized state space
  - Learns optimal action-value function

- **Dyna-Q**
  - Model-based RL
  - Combines Q-learning with planning
  - Uses simulated updates from learned model

All methods learn an action-value function:

\[
Q(s, a)
\]

using TD updates.

---

### 3. State Representation

#### Tile Coding (SARSA)
- 8 overlapping tilings
- Each tiling: 8 × 8 grid
- Produces sparse feature vectors
- Enables generalization across states

#### Discretization (Q-Learning & Dyna-Q)
- State space divided into bins:
  - Position: 18 bins  
  - Velocity: 14 bins  
- Converts continuous state into discrete indices
- Enables tabular representation

---

### 4. Experiments & Evaluation

- Each algorithm evaluated using:
  - **3 random seeds**
  - **~700 episodes per run**
- Exploration strategy:
  - ε-greedy policy (ε = 0.1)

#### Metrics
- Learning curves (mean ± standard deviation)
- Final performance:
  - Average reward over last 50 episodes

#### Visualizations
- Learning curves with error bands
- Mean performance reference lines
- Final performance bar chart with error bars

---

## Setup

Install required dependencies:

```Python

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random
from collections import defaultdict

```
### Key Settings
- Discount Factor (γ):
γ = 0.99 – 1.0

- Learning Rate (α):
α = 0.1 (Q-learning, Dyna-Q)
α = 0.3 (SARSA, normalized by tilings)

- Exploration (ε):
ε = 0.1

- Planning Steps (Dyna-Q):
n = 10

- Discretization:
(18, 14) bins

- Tile Coding:
8 tilings, 8×8 grid

pip install numpy matplotlib gymnasium
