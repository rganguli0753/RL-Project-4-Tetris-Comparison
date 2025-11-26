import numpy as np
import torch

from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from TetrisBattle.settings import GRID_WIDTH, GRID_DEPTH, PIECE_TYPE2NUM

from vanilla_dqn import encode_tetris_state


class TetrisBattleEnvWrapper:
    """
    RL-friendly wrapper around TetrisDoubleEnv.

    - Uses the underlying battle environment (TetrisDoubleEnv)
    - Reads the logical board + piece/hold information directly from the
      underlying Tetris objects
    - Returns encoded states as torch tensors of shape (15, 20, 10)
      (board + current piece + held piece)
    """

    def __init__(self, device: str = "cpu", gridchoice: str = "none"):
        """
        Parameters
        ----------
        device : "cpu" or "cuda"
            Device to put returned tensors on.
        gridchoice : str
            Initial board pattern, e.g. "none", "classic", "comboking", "lunchbox".
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

        # --- board ---
        board_raw = tetris.get_board()       
        # Transpose so it becomes (20, 10)
        board = board_raw.T 

        # --- current piece id ---
        current_piece = -1
        if getattr(tetris, "block", None) is not None:
            block_type = tetris.block.block_type()  
            if block_type in PIECE_TYPE2NUM:
                current_piece = int(PIECE_TYPE2NUM[block_type]) - 1

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

        # Ensure tensor + device
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        return state.to(self.device)

    def reset(self) -> torch.Tensor:
        """
        Resets the underlying TetrisDoubleEnv and returns an encoded state tensor.

        Returns
        -------
        state : torch.FloatTensor of shape (15, 20, 10)
        """
        _ = self.env.reset()  
        return self._extract_state()

    def step(self, action: int):
        """
        Performs an action in the environment and returns:
            next_state, reward, done, info
        where next_state is an encoded (15, 20, 10) tensor.

        Parameters
        ----------
        action : int
            Discrete action index in [0, 7].

        Returns
        -------
        next_state : torch.FloatTensor
            Shape (15, 20, 10)
        reward : float
        done : bool
        info : dict
        """
        _, reward, done, info = self.env.step(int(action))

        next_state = self._extract_state()

        return next_state, float(reward), bool(done), info

    def render(self, mode="rgb_array"):
        # We can just reuse the interface's screenshot helper
        return self.env.game_interface.get_screen_shot()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
