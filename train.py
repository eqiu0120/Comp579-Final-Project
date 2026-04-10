"""
Training script for DQN vs C51 on LunarLander-v2.

Usage examples
--------------
# DQN baseline
python train.py --agent dqn --seed 0

# C51 with 51 atoms
python train.py --agent c51 --seed 0 --n_atoms 51

# C51 atom ablation
for n in 5 11 21 51; do
    python train.py --agent c51 --n_atoms $n --seed 0
done
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import torch

from c51 import C51Agent
from dqn import DQNAgent
from replay_buffer import ReplayBuffer
from utils import set_seed, save_results, plot_results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # Agent
    p.add_argument("--agent",  type=str,   default="c51", choices=["dqn", "c51"])
    p.add_argument("--n_atoms", type=int,  default=51,
                   help="Number of atoms for C51 (ignored for DQN)")

    # Training
    p.add_argument("--total_steps",       type=int,   default=500_000)
    p.add_argument("--warmup_steps",      type=int,   default=10_000,
                   help="Steps of random exploration before learning starts")
    p.add_argument("--batch_size",        type=int,   default=32)
    p.add_argument("--buffer_size",       type=int,   default=100_000)
    p.add_argument("--update_every",      type=int,   default=1,
                   help="Perform a gradient step every N env steps")

    # Hyperparameters (shared between DQN and C51)
    p.add_argument("--lr",                type=float, default=6.25e-5)
    p.add_argument("--gamma",             type=float, default=0.99)
    p.add_argument("--eps_start",         type=float, default=1.0)
    p.add_argument("--eps_end",           type=float, default=0.01)
    p.add_argument("--eps_decay_steps",   type=int,   default=100_000)
    p.add_argument("--target_update_freq",type=int,   default=1_000)
    p.add_argument("--hidden_dim",        type=int,   default=128)

    # C51-specific
    p.add_argument("--v_min", type=float, default=-200.0)
    p.add_argument("--v_max", type=float, default=200.0)

    # Evaluation
    p.add_argument("--eval_every",  type=int, default=5_000,
                   help="Evaluate every N env steps")
    p.add_argument("--eval_eps",    type=int, default=10,
                   help="Number of deterministic episodes per evaluation")

    # Misc
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--no_plot",     action="store_true",
                   help="Skip saving plots (useful on headless servers)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(agent, env_name: str, n_episodes: int, device: str) -> float:
    """Run n_episodes deterministically (eps=0) and return mean return."""
    env = gym.make(env_name)
    returns = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                if isinstance(agent, DQNAgent):
                    action = agent.online_net(state_t).argmax(1).item()
                else:
                    action = agent.online_net.get_q_values(state_t, agent.atoms).argmax(1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
        returns.append(ep_return)
    env.close()
    return float(np.mean(returns))


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_name  = "LunarLander-v2"
    env       = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Build agent
    shared_kwargs = dict(
        state_dim         = state_dim,
        n_actions         = n_actions,
        lr                = args.lr,
        gamma             = args.gamma,
        eps_start         = args.eps_start,
        eps_end           = args.eps_end,
        eps_decay_steps   = args.eps_decay_steps,
        target_update_freq= args.target_update_freq,
        hidden_dim        = args.hidden_dim,
        device            = device,
    )
    if args.agent == "dqn":
        agent = DQNAgent(**shared_kwargs)
    else:
        agent = C51Agent(
            n_atoms = args.n_atoms,
            v_min   = args.v_min,
            v_max   = args.v_max,
            **shared_kwargs,
        )

    buffer = ReplayBuffer(args.buffer_size, state_dim)

    # Results label used for file names
    if args.agent == "c51":
        label = f"c51_atoms{args.n_atoms}_seed{args.seed}"
    else:
        label = f"dqn_seed{args.seed}"

    os.makedirs(args.results_dir, exist_ok=True)

    # Tracking
    eval_steps   = []   # env steps at each evaluation point
    eval_returns = []   # mean eval return at each evaluation point
    ep_returns   = []   # episodic returns during training (for loss curve)
    losses       = []

    state, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    last_eval = 0

    print(f"Training {label} on {env_name} for {args.total_steps:,} steps  |  device={device}")
    t0 = time.time()

    for step in range(1, args.total_steps + 1):

        # --- Action ---
        action = agent.select_action(state, env_step=step)

        # --- Step ---
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        buffer.push(state, action, reward, next_state, done)
        ep_return += reward
        state = next_state

        if done:
            ep_returns.append(ep_return)
            state, _ = env.reset()
            ep_return = 0.0

        # --- Learn ---
        if step >= args.warmup_steps and len(buffer) >= args.batch_size:
            if step % args.update_every == 0:
                batch = buffer.sample(args.batch_size)
                loss  = agent.update(*batch)
                losses.append(loss)

        # --- Evaluate ---
        if step - last_eval >= args.eval_every:
            mean_return = evaluate(agent, env_name, args.eval_eps, device)
            eval_steps.append(step)
            eval_returns.append(mean_return)
            last_eval = step
            elapsed = time.time() - t0
            print(
                f"  step {step:>7,}  |  eval return {mean_return:+8.1f}"
                f"  |  eps {agent.get_epsilon(step):.3f}"
                f"  |  {elapsed:.0f}s"
            )

    env.close()

    # --- Save ---
    save_results(
        path      = os.path.join(args.results_dir, f"{label}.npz"),
        eval_steps   = np.array(eval_steps),
        eval_returns = np.array(eval_returns),
        ep_returns   = np.array(ep_returns),
        losses       = np.array(losses),
    )

    if not args.no_plot:
        plot_results(
            eval_steps   = np.array(eval_steps),
            eval_returns = np.array(eval_returns),
            label        = label,
            save_path    = os.path.join(args.results_dir, f"{label}.png"),
        )

    print(f"\nDone. Results saved to {args.results_dir}/{label}.*")
    return eval_steps, eval_returns


if __name__ == "__main__":
    train(parse_args())
