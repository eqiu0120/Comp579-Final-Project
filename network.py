import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Simple MLP for DQN. Outputs one Q-value per action.
    Output shape: [batch, n_actions]
    """

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class C51Network(nn.Module):
    """
    MLP for C51. Outputs a probability distribution over N atoms for each action.
    Output shape: [batch, n_actions, n_atoms]  (after softmax)
    """

    def __init__(self, state_dim: int, n_actions: int, n_atoms: int, hidden_dim: int = 128):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions * n_atoms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns atom probabilities per action
        logits = self.net(x).view(-1, self.n_actions, self.n_atoms)
        return F.softmax(logits, dim=-1)  # [batch, n_actions, n_atoms]

    def get_q_values(self, x: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """
        Compute expected return E[Z(s,a)] = sum_i z_i * p_i(s,a) for each action.
        atoms: [n_atoms]
        Returns: [batch, n_actions]
        """
        probs = self.forward(x)  # [batch, n_actions, n_atoms]
        return (probs * atoms[None, None, :]).sum(-1)  # [batch, n_actions]
