import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


# ============================================================
#   STATE ENCODER
# ============================================================
def encode_tetris_state(board,
                        current_piece,
                        held_piece,
                        board_height=20,
                        board_width=10):
    """
    Converts Tetris environment state into PyTorch input tensor.

    Output shape: (15, H, W)
    """
    if not isinstance(board, torch.Tensor):
        board = torch.tensor(board, dtype=torch.float32)

    board = board.reshape(board_height, board_width)
    board_ch = board.unsqueeze(0)

    def piece_planes(pid):
        planes = torch.zeros((7, board_height, board_width), dtype=torch.float32)
        if 0 <= pid < 7:
            planes[pid] = 1.0
        return planes

    cur = piece_planes(current_piece)
    held = piece_planes(held_piece)

    return torch.cat([board_ch, cur, held], dim=0)


# ============================================================
#   DQN MODEL (vanilla CNN-based)
# ============================================================
class TetrisDQN(nn.Module):
    def __init__(self, num_actions, in_channels=15):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        b = x.size(0)
        z = self.conv(x)
        z = z.view(b, -1)
        return self.fc(z)


# ============================================================
#   PRIORITIZED EXPERIENCE REPLAY BUFFER
# ============================================================
class PrioritizedReplayBuffer:
    """
    Implements proportional PER from Schaul et al. (2016).

    priority_i = (|td_error_i| + eps)^alpha

    Sampling probability:
        P(i) = priority_i / sum(priority_j)

    IS weights:
        w_i = (N * P(i))^-beta  (normalized)
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha

        self.buffer = []
        self.pos = 0

        # Priorities
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        # Beta annealing schedule
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def _beta_by_frame(self):
        return min(1.0, self.beta_start + (1 - self.beta_start) * self.frame / self.beta_frames)

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        beta = self._beta_by_frame()
        self.frame += 1

        # Compute importance-sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        # Unpack samples
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.stack(states)
        next_states = torch.stack(next_states)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


# ============================================================
#   PER-DQN TRAINING STEP
# ============================================================
def per_dqn_train_step(
    online_net,
    target_net,
    buffer,
    optimizer,
    batch_size,
    gamma=0.99,
    priority_eps=1e-6
):
    """
    Performs one PER-DQN training step.
    """

    states, actions, rewards, next_states, dones, weights, indices = buffer.sample(batch_size)

    # Compute Q(s,a)
    q_values = online_net(states)
    q_s_a = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Compute Q target using target network
    with torch.no_grad():
        next_q = target_net(next_states).max(dim=1)[0]
        target = rewards + gamma * next_q * (1 - dones)

    # TD error
    td_errors = target - q_s_a
    loss = (weights * (td_errors ** 2)).mean()

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update priorities
    new_priorities = td_errors.abs().detach().cpu().numpy() + priority_eps
    buffer.update_priorities(indices, new_priorities)

    return loss.item()
