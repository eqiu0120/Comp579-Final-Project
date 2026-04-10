import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import DQNNetwork


class DQNAgent:
    """
    Standard Deep Q-Network agent (Mnih et al., 2015).

    - Epsilon-greedy exploration with linear decay
    - MSE loss on TD targets
    - Hard target network update every `target_update_freq` gradient steps
    - Gradient clipping to norm 10
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 6.25e-5,
        gamma: float = 0.99,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 100_000,
        target_update_freq: int = 1_000,
        hidden_dim: int = 128,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq
        self.device = device

        self._update_count = 0  # counts gradient steps (for target net sync)

        self.online_net = DQNNetwork(state_dim, n_actions, hidden_dim).to(device)
        self.target_net = DQNNetwork(state_dim, n_actions, hidden_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_epsilon(self, env_step: int) -> float:
        """Linear epsilon schedule based on environment steps."""
        frac = min(env_step / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, env_step: int) -> int:
        if np.random.random() < self.get_epsilon(env_step):
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net(state_t).argmax(1).item()

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(self, states, actions, rewards, next_states, dones) -> float:
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Q(s, a) for the taken actions
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # r + γ * max_a' Q_target(s', a')  (zero out terminal transitions)
        with torch.no_grad():
            next_q   = self.target_net(next_states).max(1)[0]
            targets  = rewards + self.gamma * next_q * (1.0 - dones)

        loss = nn.MSELoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
