import numpy as np
import torch

from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from TetrisBattle.settings import GRID_WIDTH, GRID_DEPTH, PIECE_TYPE2NUM

from Models.vanilla_dqn import encode_tetris_state


class TetrisBattleEnvWrapper:
    """
    RL-friendly wrapper around TetrisDoubleEnv.

    - Uses the underlying battle environment (TetrisDoubleEnv)
    - Reads the logical board + piece/hold information directly from Tetris objects
    - Encodes states as (15, H, W) torch tensors using encode_tetris_state
    - Implements a custom *shaped reward* in this wrapper, ignoring env's built-in reward:
        * Reward for lines sent (attack)
        * Reward for lines cleared
        * Win / loss bonus at end of game
    """

    def __init__(
        self, device: str = "cpu", gridchoice: str = "none", debug: bool = False
    ):
        """
        Parameters
        ----------
        device : "cpu" or "cuda"
            Device to put returned tensors on.
        gridchoice : str
            Initial board pattern, e.g. "none", "classic", "comboking", "lunchbox".
        debug : bool
            If True, prints detailed reward debug info each step.
        """
        self.device = torch.device(device)
        self.board_height = GRID_DEPTH
        self.board_width = GRID_WIDTH

        self.env = TetrisDoubleEnv(
            gridchoice=gridchoice,
            obs_type="image",
            mode="rgb_array",
        )

        self.action_space = self.env.action_space
        self.debug = debug

        self.player_idx = 0

        self.prev_cleared = 0
        self.prev_sent = 0

    # -----------------------------------------------------

    def _extract_state(self) -> torch.Tensor:
        """
        Reads the current player's Tetris object and builds
        (board, current_piece_id, held_piece_id), then encodes to
        a (15, H, W) tensor.

        Returns
        -------
        state : torch.FloatTensor
            Shape (15, board_height, board_width)
        """
        gi = self.env.game_interface
        player_idx = gi.now_player
        tetris = gi.tetris_list[player_idx]["tetris"]

        #  board
        board_raw = tetris.get_board()
        board = board_raw.T

        # current piece id
        current_piece = -1
        if getattr(tetris, "block", None) is not None:
            block_type = tetris.block.block_type()
            if block_type in PIECE_TYPE2NUM:
                current_piece = int(PIECE_TYPE2NUM[block_type]) - 1

        # held piece id
        held_piece = -1
        if getattr(tetris, "held", None) is not None:
            held_type = tetris.held.block_type()
            if held_type in PIECE_TYPE2NUM:
                held_piece = int(PIECE_TYPE2NUM[held_type]) - 1

        state = encode_tetris_state(
            board,
            current_piece,
            held_piece,
            board_height=self.board_height,
            board_width=self.board_width,
        )

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        return state.to(self.device)

    # -----------------------------------------------------

    def reset(self) -> torch.Tensor:
        """
        Resets the underlying TetrisDoubleEnv and returns an encoded state tensor.

        We also:
          - Record which player index is currently now_player
          - Initialize our prev_cleared / prev_sent baselines for reward shaping
        """
        _ = self.env.reset()

        gi = self.env.game_interface
        # Treat the player who is about to move as "our agent"
        self.player_idx = gi.now_player

        tetris = gi.tetris_list[self.player_idx]["tetris"]
        self.prev_cleared = getattr(tetris, "cleared", 0)
        self.prev_sent = getattr(tetris, "sent", 0)

        if self.debug:
            print(
                f"[RESET] player_idx={self.player_idx}, "
                f"cleared={self.prev_cleared}, sent={self.prev_sent}"
            )

        return self._extract_state()

    # -----------------------------------------------------

    def _compute_shaped_reward(
        self, env_reward: float, done: bool, info: dict
    ) -> float:
        """
        Compute shaped reward from Tetris stats for our player:

        r = ATTACK_W * Δ(sent) + CLEAR_W * Δ(cleared)
            + WIN_BONUS / LOSS_PENALTY at terminal.

        env_reward is ignored in shaping (we track it only for debugging).
        """
        gi = self.env.game_interface
        tetris = gi.tetris_list[self.player_idx]["tetris"]

        # Totals so far
        total_cleared = getattr(tetris, "cleared", 0)
        total_sent = getattr(tetris, "sent", 0)

        # Deltas for *this step*
        delta_cleared = total_cleared - self.prev_cleared
        delta_sent = total_sent - self.prev_sent

        # Update baselines
        self.prev_cleared = total_cleared
        self.prev_sent = total_sent

        highest_row = self.get_highest_block_height()
        normalized_height = 1 - (highest_row / self.board_height)
        HEIGHT_PENALTY = 0.5

        # Reward weights
        ATTACK_W = 1.0  # reward for sending lines (attack)
        CLEAR_W = 0.25  # reward for clearing lines
        WIN_BONUS = 10.0
        LOSS_PENALTY = 10.0

        reward = 0.0
        reward += ATTACK_W * float(delta_sent)
        reward += CLEAR_W * float(delta_cleared)
        reward -= HEIGHT_PENALTY * normalized_height

        # Terminal bonus/penalty if winner info is present
        if done and "winner" in info:
            winner = info["winner"]
            if winner == self.player_idx:
                reward += WIN_BONUS
            else:
                reward -= LOSS_PENALTY

        if self.debug:
            print(
                f"[REWARD] env={env_reward:.3f} | "
                f"Δsent={delta_sent}, Δcleared={delta_cleared}, "
                f"winner={info.get('winner', None)} | shaped={reward:.3f}"
            )

        return reward

    # -----------------------------------------------------

    def step(self, action: int):
        """
        Performs an action in the environment and returns:
            next_state, shaped_reward, done, info

        NOTE: The env's own reward is ignored; we compute our own shaped reward.
        The original env reward is still stored in info['env_reward'] for debugging.
        """

        _, env_reward, done, info = self.env.step(int(action))

        shaped_reward = self._compute_shaped_reward(env_reward, done, info)
        next_state = self._extract_state()

        info = dict(info)
        info["env_reward"] = env_reward
        info["shaped_reward"] = shaped_reward

        return next_state, shaped_reward, bool(done), info

    # -----------------------------------------------------

    def render(self, mode="rgb_array"):
        # Reuse interface screenshot helper
        return self.env.game_interface.get_screen_shot()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()

    def get_highest_block_height(self):
        """
        Returns the row index of the highest occupied cell in the board.
        0 = top row, board_height-1 = bottom row.
        If the board is empty, returns board_height (no blocks).
        """
        tetris = self.env.game_interface.tetris_list[self.player_idx]["tetris"]
        board = tetris.get_board().T
        occupied_rows = np.where(board.any(axis=1))[0]
        if len(occupied_rows) == 0:
            return self.board_height  # empty board
        return occupied_rows[0]  # top-most block row
