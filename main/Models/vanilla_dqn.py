import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#   STATE ENCODING
# ============================================================
def encode_tetris_state(
    board: torch.Tensor,
    current_piece: int,
    held_piece: int,
    ghost_piece=None,
    current_piece_coords=None,
    board_height=20,
    board_width=10,
    block_height_on_board=0,
) -> torch.Tensor:
    """
    Encodes the Tetris state into a tensor with shape (17, H, W):
      - 0-6: one-hot for current piece type
      - 7-13: one-hot for held piece type
      - 14: board occupancy
      - 15: current piece occupancy
      - 16: ghost piece occupancy
      Optionally include block height for additional channel or feature.
    """
    device = board.device if isinstance(board, torch.Tensor) else torch.device("cpu")

    H, W = board_height, board_width

    # --- Initialize tensor ---
    state_tensor = torch.zeros((17, H, W), dtype=torch.float32, device=device)

    # --- Board occupancy ---
    state_tensor[14, : board.shape[0], : board.shape[1]] = torch.tensor(
        board, dtype=torch.float32, device=device
    )

    # --- Current piece one-hot channel ---
    if current_piece >= 0 and current_piece < 7:
        state_tensor[current_piece, :, :] = 1.0

    # --- Held piece one-hot channel ---
    if held_piece >= 0 and held_piece < 7:
        state_tensor[7 + held_piece, :, :] = 1.0

    # --- Current piece occupancy ---
    if current_piece_coords:
        for x, y in current_piece_coords:
            if 0 <= y < H and 0 <= x < W:
                state_tensor[15, y, x] = 1.0

    # --- Ghost piece occupancy ---
    if ghost_piece:
        gx, gy = ghost_piece["px"], ghost_piece["py"]
        for x, y in current_piece_coords:
            abs_x, abs_y = x + (gx - current_piece_coords[0][0]), y + (
                gy - current_piece_coords[0][1]
            )
            if 0 <= abs_y < H and 0 <= abs_x < W:
                state_tensor[16, abs_y, abs_x] = 1.0

    # --- Optional: add block height as a separate feature map ---
    # Normalize height by board height
    height_map = torch.full(
        (H, W), block_height_on_board / H, dtype=torch.float32, device=device
    )
    # This can be added as an extra channel if you want
    # state_tensor = torch.cat([state_tensor, height_map.unsqueeze(0)], dim=0)

    return state_tensor


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
            nn.Linear(128 * 20 * 10, 512), nn.ReLU(), nn.Linear(512, num_actions)
        )

    def forward(self, x):
        batch = x.size(0)
        out = self.conv(x)  # (B, 128, 20, 10)
        out = out.view(batch, -1)  # flatten
        q_values = self.fc(out)  # (B, num_actions)
        return q_values
