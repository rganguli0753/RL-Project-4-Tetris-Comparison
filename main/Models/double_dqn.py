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
    board: (H, W) 0/1 occupancy
    Returns tensor (15, H, W):
        1 board channel
        7 one-hot current piece planes
        7 one-hot held piece planes
    """
    if not isinstance(board, torch.Tensor):
        board = torch.tensor(board, dtype=torch.float32)

    board = board.reshape(board_height, board_width).float()
    board_ch = board.unsqueeze(0)

    def planes(pid):
        t = torch.zeros((7, board_height, board_width), dtype=torch.float32)
        if pid >= 0 and pid < 7:
            t[pid, :, :] = 1.0
        return t

    cur = planes(current_piece)
    held = planes(held_piece)

    return torch.cat([board_ch, cur, held], dim=0)


# ============================================================
#   Q-NETWORK (used for online + target networks)
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
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        b = x.size(0)
        out = self.conv(x)
        out = out.view(b, -1)
        return self.fc(out)


# ============================================================
#   REPLAY BUFFER
# ============================================================
class DoubleReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

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
#   DOUBLE DQN AGENT
# ============================================================
class DoubleDQNAgent:
    def __init__(
        self,
        num_actions,
        lr=1e-4,
        gamma=0.99,
        tau=0.005,
        device="cpu",
        replay_size=100000,
        batch_size=64,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions
        self.batch_size = batch_size

        # Online and target networks
        self.online = TetrisDQN(num_actions).to(device)
        self.target = TetrisDQN(num_actions).to(device)
        self.target.load_state_dict(self.online.state_dict())

        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.replay = DoubleReplayBuffer(replay_size)

        self.eps = 1.0   # For epsilon-greedy
        self.eps_min = 0.05
        self.eps_decay = 0.9995

    # ----------------------------
    # Epsilon-greedy action
    # ----------------------------
    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.num_actions - 1)

        with torch.no_grad():
            q = self.online(state.unsqueeze(0).to(self.device))
            return q.argmax(dim=1).item()

    # ----------------------------
    # Add experience to buffer
    # ----------------------------
    def push(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)

    # ----------------------------
    # Double DQN training step
    # ----------------------------
    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        states = states.to(self.device)
        next_states = next_states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)

        # Q(s, a) from online network
        q_values = self.online(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # ----------------------------
        # Double DQN target:
        # 1. Use online net to choose argmax action
        # 2. Use target net to evaluate that action
        # ----------------------------
        with torch.no_grad():
            next_q_online = self.online(next_states)
            best_actions = next_q_online.argmax(dim=1)

            next_q_target = self.target(next_states)
            next_q = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)

            target = rewards + self.gamma * next_q * (1 - dones)

        # Loss
        loss = F.smooth_l1_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # Soft target update
        for p_target, p_online in zip(self.target.parameters(), self.online.parameters()):
            p_target.data.copy_(self.tau * p_online.data + (1 - self.tau) * p_target.data)

        # Epsilon decay
        self.eps = max(self.eps * self.eps_decay, self.eps_min)

        return loss.item()
