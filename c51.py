import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import C51Network


class C51Agent:
    """
    Categorical DQN (C51) agent (Bellemare, Dabney & Munos, 2017).

    The value distribution Z(s, a) is modelled as a discrete distribution
    over N atoms {z_0, ..., z_{N-1}} equally spaced in [V_MIN, V_MAX].

    Learning uses the projected Bellman update (Algorithm 1 in the paper):
      1. Compute Bellman targets  T̂z_j = r + γ * z_j  for each atom j
      2. Project T̂z_j onto the atom grid (linear interpolation to neighbours)
      3. Minimise cross-entropy between projected target and predicted distribution

    Action selection is greedy on E[Z(s,a)] = Σ_i z_i * p_i(s,a).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        n_atoms: int = 51,
        v_min: float = -200.0,
        v_max: float = 200.0,
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
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq
        self.device = device

        self._update_count = 0

        # Fixed atom support  z_i = V_MIN + i * Δz
        self.atoms = torch.linspace(v_min, v_max, n_atoms).to(device)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        self.online_net = C51Network(state_dim, n_actions, n_atoms, hidden_dim).to(device)
        self.target_net = C51Network(state_dim, n_actions, n_atoms, hidden_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_epsilon(self, env_step: int) -> float:
        frac = min(env_step / self.eps_decay_steps, 1.0)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, env_step: int) -> int:
        if np.random.random() < self.get_epsilon(env_step):
            return np.random.randint(self.n_actions)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.online_net.get_q_values(state_t, self.atoms).argmax(1).item()

    # ------------------------------------------------------------------
    # Projected Bellman update  (Algorithm 1 from the paper)
    # ------------------------------------------------------------------

    def update(self, states, actions, rewards, next_states, dones) -> float:
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        batch_size  = states.shape[0]

        with torch.no_grad():
            # Select greedy actions for next states using the target network
            next_probs   = self.target_net(next_states)                         # [B, A, N]
            next_q       = (next_probs * self.atoms[None, None, :]).sum(-1)     # [B, A]
            next_actions = next_q.argmax(1)                                     # [B]

            # Distribution of next best action: Z(s', a*)
            next_dist = next_probs[range(batch_size), next_actions]             # [B, N]

            # --- Projection step ---
            # T̂z_j = clip(r + γ * z_j, V_MIN, V_MAX)
            # dones masks the γ term so terminal states produce a point mass at r
            t_z = rewards[:, None] + self.gamma * self.atoms[None, :] * (1.0 - dones[:, None])
            t_z = t_z.clamp(self.v_min, self.v_max)                            # [B, N]

            # b_j = (T̂z_j - V_MIN) / Δz  ∈ [0, N-1]
            b = (t_z - self.v_min) / self.delta_z                              # [B, N]
            l = b.floor().long().clamp(0, self.n_atoms - 1)                    # [B, N]
            u = b.ceil().long().clamp(0, self.n_atoms - 1)                     # [B, N]

            # Interpolation weights
            # Edge case: when l == u (b is an integer), assign full probability to l
            same = (l == u)
            l_weight = torch.where(same, torch.ones_like(b),  u.float() - b)   # [B, N]
            u_weight = torch.where(same, torch.zeros_like(b), b - l.float())   # [B, N]

            # Accumulate projected probabilities into m  [B, N]
            m = torch.zeros(batch_size, self.n_atoms, device=self.device)
            # Batch-safe scatter: flatten to 1-D, use per-batch offsets
            offset = (torch.arange(batch_size, device=self.device) * self.n_atoms)[:, None]  # [B,1]
            m.view(-1).scatter_add_(0, (l + offset).view(-1), (next_dist * l_weight).view(-1))
            m.view(-1).scatter_add_(0, (u + offset).view(-1), (next_dist * u_weight).view(-1))

        # --- Cross-entropy loss: -Σ_i m_i * log p_i(s, a) ---
        probs     = self.online_net(states)                                     # [B, A, N]
        log_probs = torch.log(probs[range(batch_size), actions] + 1e-8)        # [B, N]
        loss      = -(m * log_probs).sum(1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()
