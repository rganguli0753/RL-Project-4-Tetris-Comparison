import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class MCTPNet(nn.Module):
    """
    Convolutional policy/value network for Tetris to be used with MCTP/MCTS.
    Input: tensor shape (B, C, H, W)
      - typical C: 1 (board occupancy) + 7 (one-hot current piece type plane) + 7 (one-hot held piece type plane) = 15
      - but encode_state below constructs this format
    Outputs:
      - policy logits: (B, num_actions)
      - value: (B, 1) in [-1, 1] via tanh
    """
    def __init__(self, in_channels: int, num_actions: int, base_channels: int = 64, n_resblocks: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # Residual trunk
        self.resblocks = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(n_resblocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(base_channels, base_channels // 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(base_channels // 2)
        self.policy_fc = nn.Linear((base_channels // 2) * 20 * 10, num_actions)  # assumes board H=20, W=10

        # Value head
        self.value_conv = nn.Conv2d(base_channels, base_channels // 2, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(base_channels // 2)
        self.value_fc1 = nn.Linear((base_channels // 2) * 20 * 10, base_channels)
        self.value_fc2 = nn.Linear(base_channels, 1)

        # init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, C, H, W)  -- expects H=20, W=10 (change linear sizes if different)
        returns:
          policy_logits: (B, num_actions)
          value: (B, 1)  (tanh applied)
        """
        batch = x.shape[0]
        out = self.stem(x)                 # (B, base_channels, H, W)
        out = self.resblocks(out)          # (B, base_channels, H, W)

        # Policy head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(batch, -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(batch, -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value = torch.tanh(v)

        return policy_logits, value


# -------------------------
# Helper: encode_state(...)
# -------------------------
# This helper encodes the tetris board + current piece id + held piece id into tensor the net expects.
# Piece indices: 0..6 for the seven tetromino types (I, O, T, S, Z, J, L) — ensure your environment uses the same mapping.

def encode_state(board: torch.Tensor,
                 current_piece: int,
                 held_piece: int,
                 board_height: int = 20,
                 board_width: int = 10) -> torch.Tensor:
    """
    board: tensor (H, W) with 0 empty, 1 filled (or integer occupancies). Accepts torch.Tensor or numpy-like.
    current_piece: int in 0..6 or -1 if none
    held_piece: int in 0..6 or -1 if none
    Returns: tensor (C, H, W)
      C = 1 + 7 + 7 = 15 by default
      - channel 0: board occupancy (0/1)
      - channels 1..7: current-piece one-hot plane (each plane filled with 1 if current_piece==that index, else 0)
      - channels 8..14: held-piece one-hot plane (same encoding)
    """
    if not isinstance(board, torch.Tensor):
        board = torch.tensor(board, dtype=torch.float32)
    board = board.reshape(board_height, board_width).float()

    # board channel
    board_ch = board.unsqueeze(0)  # (1, H, W)

    # one-hot planes for pieces
    def piece_planes(idx: int):
        planes = torch.zeros((7, board_height, board_width), dtype=torch.float32)
        if idx is not None and idx >= 0 and idx < 7:
            planes[idx, :, :] = 1.0
        return planes

    cur_planes = piece_planes(current_piece)
    held_planes = piece_planes(held_piece)

    tensor = torch.cat([board_ch, cur_planes, held_planes], dim=0)  # (15, H, W)
    return tensor

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # example env state:
    H, W = 20, 10
    dummy_board = torch.zeros((H, W))
    # put some blocks as example
    dummy_board[-1, :] = 1.0  # filled bottom row
    current_piece = 0  # I piece
    held_piece = 3     # S piece

    # encode
    input_tensor = encode_state(dummy_board, current_piece, held_piece, board_height=H, board_width=W)
    input_tensor = input_tensor.unsqueeze(0)  # batch dim -> (1, C, H, W)

    num_actions = 200  # example: enumerate all legal placements + hold as action indices
    net = MCTPNet(in_channels=input_tensor.shape[1], num_actions=num_actions)
    logits, value = net(input_tensor)
    probs = F.softmax(logits, dim=-1)

    print("policy logits shape:", logits.shape)  # (1, num_actions)
    print("probs shape:", probs.shape)
    print("value shape:", value.shape)            # (1,1)
