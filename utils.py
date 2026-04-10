"""
Utility functions: seed setting, result I/O, and plotting.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(path: str, **arrays):
    """Save named numpy arrays to a .npz file."""
    np.savez(path, **arrays)


def load_results(path: str) -> dict:
    """Load arrays from a .npz file into a plain dict."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def smooth(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving-average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_results(
    eval_steps: np.ndarray,
    eval_returns: np.ndarray,
    label: str,
    save_path: str,
    window: int = 5,
):
    """Plot a single run's evaluation curve and save to disk."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eval_steps, eval_returns, alpha=0.3, color="steelblue")
    smoothed = smooth(eval_returns, window)
    ax.plot(eval_steps[len(eval_steps) - len(smoothed):], smoothed,
            color="steelblue", label=label)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Evaluation Return")
    ax.set_title(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_comparison(
    results: dict,
    title: str,
    save_path: str,
    window: int = 5,
):
    """
    Plot multiple runs on the same axes.

    Parameters
    ----------
    results : dict
        Mapping from label str to dict with keys 'eval_steps' and 'eval_returns'.
        eval_returns may be 2-D [n_seeds, n_evals] — mean ± std will be plotted.
    title   : str
    save_path : str
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for label, data in results.items():
        steps   = data["eval_steps"]         # [n_evals]
        returns = data["eval_returns"]       # [n_seeds, n_evals] or [n_evals]

        if returns.ndim == 1:
            mean = returns
            std  = None
        else:
            mean = returns.mean(0)
            std  = returns.std(0)

        # Smooth
        sm = smooth(mean, window)
        trim = len(mean) - len(sm)
        xs   = steps[trim:]

        ax.plot(xs, sm, label=label)
        if std is not None:
            sm_std = smooth(std, window)
            ax.fill_between(xs, sm - sm_std, sm + sm_std, alpha=0.15)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Mean Evaluation Return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved comparison plot → {save_path}")


# ---------------------------------------------------------------------------
# Convenience: aggregate multiple seeds from results directory
# ---------------------------------------------------------------------------

def aggregate_seeds(results_dir: str, prefix: str) -> dict:
    """
    Load all .npz files in results_dir whose name starts with prefix,
    stack eval_returns across seeds.

    Returns dict with 'eval_steps' and 'eval_returns' [n_seeds, n_evals].
    """
    files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith(prefix) and f.endswith(".npz")
    )
    if not files:
        raise FileNotFoundError(f"No files matching '{prefix}*.npz' in {results_dir}")

    all_returns = []
    steps = None
    for f in files:
        d = load_results(os.path.join(results_dir, f))
        all_returns.append(d["eval_returns"])
        if steps is None:
            steps = d["eval_steps"]

    return {
        "eval_steps":   steps,
        "eval_returns": np.stack(all_returns, axis=0),
    }
