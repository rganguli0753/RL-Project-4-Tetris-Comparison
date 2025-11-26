import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#   STATE ENCODING (board + current + held block)
# ============================================================
def encode_tetris_state(board,
                        current_piece,
                        held_piece,
                        board_height=20,
                        board_width=10):
    """
    Converts Tetris environment state into PyTorch input tensor.

    board: (H, W) 0/1 occupancy
    current_piece: int from 0..6 or -1
    held_piece: int from 0..6 or -1

    Output:
        tensor (15, H, W):
            1 channel: board
            7 channels: current piece one-hot planes
            7 channels: held piece one-hot planes
    """

    if not isinstance(board, torch.Tensor):
        board = torch.tensor(board, dtype=torch.float32)

    board = board.reshape(board_height, board_width).float()
    board_ch = board.unsqueeze(0)

    def piece_to_planes(piece_id):
        planes = torch.zeros((7, board_height, board_width), dtype=torch.float32)
        if 0 <= piece_id < 7:
            planes[piece_id, :, :] = 1.0
        return planes

    cur_planes = piece_to_planes(current_piece)
    held_planes = piece_to_planes(held_piece)

    # (15, H, W)
    return torch.cat([board_ch, cur_planes, held_planes], dim=0)


# ============================================================
#   SHARED CNN BACKBONE
# ============================================================
class TetrisCNN(nn.Module):
    """
    CNN feature extractor used by both actor and critic.
    Output is a 512-dimensional latent vector.
    """
    def __init__(self, in_channels=15):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(128 * 20 * 10, 512),
            nn.ReLU()
        )

    def forward(self, x):
        b = x.size(0)
        out = self.conv(x)
        out = out.view(b, -1)
        return self.linear(out)     # (B, 512)


# ============================================================
#   CONTINUOUS ACTION ACTOR (Gaussian policy)
# ============================================================
class TetrisContinuousActor(nn.Module):
    """
    Outputs a Gaussian distribution:
        mean:  (B, action_dim)
        log_std: (B, action_dim)

    Action is sampled as:
        a = mean + std * eps
    """

    def __init__(self, action_dim, in_channels=15):
        super().__init__()

        self.backbone = TetrisCNN(in_channels=in_channels)

        self.mu_head = nn.Linear(512, action_dim)
        self.log_std_head = nn.Linear(512, action_dim)

        # Reasonable default bounds
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, x):
        h = self.backbone(x)

        mu = self.mu_head(h)
        log_std = self.log_std_head(h)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mu, log_std

    def sample(self, x):
        mu, log_std = self(x)
        std = log_std.exp()
        eps = torch.randn_like(std)
        action = mu + eps * std
        return action, mu, log_std


# ============================================================
#   CONTINUOUS CRITIC (Q-network)
# ============================================================
class TetrisContinuousCritic(nn.Module):
    """
    Q(s, a) network.
    Input: state tensor + action vector
    Output: scalar Q value
    """
    def __init__(self, action_dim, in_channels=15):
        super().__init__()

        self.backbone = TetrisCNN(in_channels=in_channels)

        self.q_head = nn.Sequential(
            nn.Linear(512 + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, action):
        h = self.backbone(x)
        q_input = torch.cat([h, action], dim=-1)
        q = self.q_head(q_input)
        return q
