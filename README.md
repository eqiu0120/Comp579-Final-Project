# Distributional Reinforcement Learning: Reproducing and Evaluating C51

**COMP-579 Reinforcement Learning — Group 10**

## Overview

This project reproduces the **C51 (Categorical DQN)** algorithm introduced by Bellemare, Dabney, and Munos (2017) and evaluates it against a standard **DQN** baseline on lightweight environments. We also conduct an ablation study on the number of atoms in the categorical distribution to analyze its effect on learning stability, convergence speed, and final performance.

The core idea of distributional RL is to model the full *distribution* of returns Z(s, a) rather than just its expectation Q(s, a) = E[Z(s, a)]. C51 parametrizes this distribution using N discrete atoms evenly spaced over a fixed return range [V_MIN, V_MAX], and learns atom probabilities via a projected Bellman update minimized with cross-entropy loss.

## Project Goals

1. **Reproduce C51** from scratch and verify it outperforms DQN on standard benchmarks
2. **Ablation study**: vary number of atoms N ∈ {5, 11, 21, 51} and measure the effect on:
   - Final episode return
   - Convergence speed
   - Learning stability (variance across seeds)
3. **Compare** C51 vs DQN under identical conditions (same network architecture, optimizer, hyperparameters — differing only in output head and loss function)

## Environment

**LunarLander-v2** (Gymnasium)

- Dense rewards provide a clear learning signal for both DQN and C51, making the comparison meaningful
- Fast to train — convergence is visible within 1M steps without a GPU
- No visual preprocessing required — keeps the focus on the distributional RL component rather than CNN architecture choices
- Well-known benchmark with established DQN performance, making results easy to sanity-check

## Algorithms

### DQN (Baseline)
- Output: Q-values for each action — shape `[|A|]`
- Loss: MSE on TD targets `(r + γ max_a' Q(s', a') - Q(s, a))²`
- Target network updated every fixed number of steps

### C51 / Categorical DQN
- Output: Atom probabilities for each action — shape `[|A| × N]`
- N atoms: `z_i = V_MIN + i * Δz`, where `Δz = (V_MAX - V_MIN) / (N - 1)`
- Bellman update: project `T̂z_j = r + γz_j` onto the atom grid, distribute probability to neighbours
- Loss: Cross-entropy (KL divergence) between projected target distribution and predicted distribution
- Action selection: greedy on `E[Z(s, a)] = Σ_i z_i * p_i(s, a)`

### Atom Ablation
Run C51 with N ∈ {5, 11, 21, 51} atoms, all other hyperparameters fixed.

## Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 6.25e-5 |
| Discount factor γ | 0.99 |
| Replay buffer size | 100,000 |
| Batch size | 32 |
| Target network update frequency | 1,000 steps |
| ε-greedy start / end / decay | 1.0 / 0.01 / 100,000 steps |
| V_MIN / V_MAX | -10 / 10 |
| N atoms (C51 default) | 51 |
| Random seeds | 3–5 seeds per experiment |

## Evaluation Protocol

- Training: up to 1M environment steps per run
- Metric: mean episode return, smoothed over a rolling window
- Results reported as mean ± std across seeds
- Learning curves plotted for all conditions

## Repository Structure

```
.
├── README.md
├── dqn.py                  # Standard DQN implementation
├── c51.py                  # C51 / Categorical DQN implementation
├── train.py                # Training loop and evaluation
├── replay_buffer.py        # Experience replay
├── network.py              # Neural network architectures
├── utils.py                # Plotting, logging, seed utilities
├── experiments/
│   ├── run_dqn.sh
│   ├── run_c51.sh
│   └── run_ablation.sh
├── results/                # Saved training logs and plots
└── notebooks/
    └── analysis.ipynb      # Result analysis and plots for the paper
```

## Implementation Notes

Both DQN and C51 are implemented **from scratch** using PyTorch — no RL libraries (e.g. Stable Baselines, RLlib, CleanRL) are used. The only external dependencies are:

- `gymnasium` — environment interface only
- `torch` — neural network and autograd
- `numpy` — array utilities
- `matplotlib` — plotting

```bash
pip install gymnasium[box2d] torch numpy matplotlib
```

## Running Experiments

```bash
# Train DQN baseline
python train.py --agent dqn --seed 0

# Train C51
python train.py --agent c51 --seed 0 --n_atoms 51

# Atom ablation study
for n in 5 11 21 51; do
    python train.py --agent c51 --n_atoms $n --seed 0
done
```

## References

Bellemare, M. G., Dabney, W., & Munos, R. (2017). *A Distributional Perspective on Reinforcement Learning*. ICML 2017.

Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
