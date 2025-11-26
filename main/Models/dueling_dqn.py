import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ============================================================
#   STATE ENCODER
# ============================================================
def encode_tetris_state(board,
                        current_piece,
                        held_piece,
                        board_height=20,
                        board_width=10):
    """
    Returns tensor of shape (15, H, W):
        1 board channel
        7 one-hot channels for current piece
        7 one-hot channels for held piece
    """
    if not isinstance(board, torch.Tensor):
        board = torch.tensor(board, dtype=torch.float32)

    board = board.reshape(board_height, board_width).float()
    board_ch = board.unsqueeze(0)  # (1, H, W)

    def planes(pid):
        t = torch.zeros((7, board_height, board_width), dtype=torch.float32)
        if 0 <= pid < 7:
            t[pid, :, :] = 1.0
        return t

    current = planes(current_piece)
    held = planes(held_piece)

    return torch.cat([board_ch, current, held], dim=0)  # (15, H, W)


# ============================================================
#   Dueling DQN Network
# ============================================================
class DuelingTetrisDQN(nn.Module):
    """
    CNN encoder → value stream + advantage stream → Q-values.
    Standard dueling architecture:
        Q = V + (A - mean(A))
    """
    def __init__(self, num_actions, in_channels=15):
        super().__init__()

        # Shared convolutional encoder
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        flattened_dim = 128 * 20 * 10

        # Value stream
        self.value_fc = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Advantage stream
        self.adv_fc = nn.Sequential(
            nn.Linear(flattened_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        batch = x.size(0)
        h = self.conv(x)
        h = h.view(batch, -1)

        V = self.value_fc(h)              # (B, 1)
        A = self.adv_fc(h)                # (B, num_actions)

        # Dueling combination
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


# ============================================================
#   Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(s2),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
#   Dueling DQN AGENT
# ============================================================
class DuelingDQNAgent:
    def __init__(
        self,
        num_actions,
        lr=1e-4,
        gamma=0.99,
        device="cpu",
        replay_size=100000,
        batch_size=64,
        tau=0.005,       # soft update
    ):
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.num_actions = num_actions

        # Networks
        self.online = DuelingTetrisDQN(num_actions).to(device)
        self.target = DuelingTetrisDQN(num_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = ReplayBuffer(replay_size)

        # Exploration
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 0.9995

    # ---------------
    #   Action selection
    # ---------------
    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            q = self.online(state.unsqueeze(0).to(self.device))
            return q.argmax(1).item()

    # ---------------
    #   Store transition
    # ---------------
    def push(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    # ---------------
    #   One training step (standard DQN)
    # ---------------
    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        s, a, r, s2, d = self.replay.sample(self.batch_size)

        s = s.to(self.device)
        s2 = s2.to(self.device)
        a = a.to(self.device)
        r = r.to(self.device)
        d = d.to(self.device)

        # Q(s, a)
        q = self.online(s)
        q_sa = q.gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q(s2)
        with torch.no_grad():
            next_q = self.target(s2).max(1)[0]
            target = r + self.gamma * next_q * (1 - d)

        loss = F.smooth_l1_loss(q_sa, target)

        # SGD step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft update target network
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.copy_(self.tau * p_o.data + (1 - self.tau) * p_t.data)

        # Epsilon decay
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

        return loss.item()