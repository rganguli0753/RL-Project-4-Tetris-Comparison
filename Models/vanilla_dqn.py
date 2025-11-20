import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#   STATE ENCODING
# ============================================================
def encode_tetris_state(board,
                        current_piece,
                        held_piece,
                        board_height=20,
                        board_width=10):
    """
    Converts Tetris environment state into PyTorch input tensor.

    board: (H, W) 0/1 occupancy
    current_piece: int from 0..6 or -1 if no active piece
    held_piece: int from 0..6 or -1 if none
    Returns: tensor (15, H, W):
        1 channel for board
        7 channels for current piece type (one-hot planes)
        7 channels for held piece type
    """
    if not isinstance(board, torch.Tensor):
        board = torch.tensor(board, dtype=torch.float32)

    board = board.reshape(board_height, board_width).float()
    board_ch = board.unsqueeze(0)  # (1, H, W)

    def piece_to_planes(piece_id):
        planes = torch.zeros((7, board_height, board_width), dtype=torch.float32)
        if 0 <= piece_id < 7:
            planes[piece_id, :, :] = 1.0
        return planes

    cur_planes = piece_to_planes(current_piece)
    held_planes = piece_to_planes(held_piece)

    # Final is (15, H, W)
    return torch.cat([board_ch, cur_planes, held_planes], dim=0)


# ============================================================
#   VANILLA DQN MODEL
# ============================================================
class VanillaDQN(nn.Module):
    """
    Vanilla DQN network for Tetris.
    Input:  (B, 15, 20, 10)
    Output: (B, num_actions)
    """
    def __init__(self, num_actions, in_channels=15):
        super().__init__()

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Fully connected Q-value head
        self.fc = nn.Sequential(
            nn.Linear(128 * 20 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        batch = x.size(0)
        out = self.conv(x)              # (B, 128, 20, 10)
        out = out.view(batch, -1)       # flatten
        q_values = self.fc(out)         # (B, num_actions)
        return q_values


# ============================================================
#   Example usage
# ============================================================
if __name__ == "__main__":
    # Example Tetris state
    H, W = 20, 10
    board = torch.zeros((H, W))
    board[-1] = 1  # bottom row filled

    current_piece = 0   # I piece
    held_piece = 3      # S piece

    # Encode the state
    state = encode_tetris_state(board, current_piece, held_piece)
    state = state.unsqueeze(0)   # batch dimension -> (1, 15, 20, 10)

    # Define action space (example: 200 discrete actions)
    num_actions = 200
    model = VanillaDQN(num_actions=num_actions)

    # Forward pass
    q_values = model(state)
    print("Q-values shape:", q_values.shape)  # (1, 200)
