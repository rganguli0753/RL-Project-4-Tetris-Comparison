"""
Simple DQN training script for your Tetris setup.

- Imports VanillaDQN from Models.vanilla_dqn
- Imports TetrisBattleEnvWrapper from tetris_env_wrapper.py
- Implements a basic replay buffer, target network, epsilon-greedy policy, and training loop
- Saves the trained model to `trained_dqn.pth`

Adjust hyperparameters in HYPERPARAMS below.
"""

import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Import your model and environment wrapper
try:
    from Models.vanilla_dqn import VanillaDQN
except Exception as e:
    raise ImportError("Could not import VanillaDQN from Models.vanilla_dqn. "
                      "Make sure Models/vanilla_dqn.py exists and defines VanillaDQN.") from e

try:
    from tetris_env_wrapper import TetrisBattleEnvWrapper
except Exception as e:
    raise ImportError("Could not import TetrisBattleEnvWrapper from tetris_env_wrapper.py. "
                      "Ensure the file exists and defines class TetrisBattleEnvWrapper.") from e

# Experience tuple
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.buffer = deque(maxlen=self.capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


def select_action(q_net, state, epsilon, device, valid_mask=None):
    """
    Epsilon-greedy action selection.
    - state: torch tensor shape (C, H, W) or (1, C, H, W)
    - valid_mask: optional boolean numpy or torch array shape (num_actions,) or (1, num_actions)
    """
    num_actions = q_net.fc[-1].out_features if hasattr(q_net, "fc") else None

    if random.random() < epsilon:
        # random valid action if mask provided
        if valid_mask is not None:
            if isinstance(valid_mask, torch.Tensor):
                mask = valid_mask.cpu().numpy().astype(bool)
            else:
                mask = np.array(valid_mask).astype(bool)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                return random.randrange(num_actions) if num_actions is not None else 0
            return int(np.random.choice(valid_indices))
        else:
            return int(random.randrange(num_actions)) if num_actions is not None else 0
    else:
        q_net.eval()
        with torch.no_grad():
            x = state.unsqueeze(0).to(device) if state.dim() == 3 else state.to(device)
            q_values = q_net(x)  # (1, num_actions)
            q_values = q_values.cpu().squeeze(0).numpy()
            if valid_mask is not None:
                mask = np.array(valid_mask).astype(bool)
                q_values[~mask] = -1e9
            action = int(np.argmax(q_values))
        q_net.train()
        return action


def compute_td_loss(batch, q_net, target_net, gamma, device):
    states = torch.stack(batch.state).to(device)            # (B, C, H, W)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)  # (B,1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)
    next_states = torch.stack(batch.next_state).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)  # (B,1)

    q_values = q_net(states).gather(1, actions)  # (B,1)

    with torch.no_grad():
        next_q = target_net(next_states)  # (B, num_actions)
        max_next_q = next_q.max(dim=1, keepdim=True)[0]  # (B,1)
        target = rewards + (1.0 - dones) * gamma * max_next_q

    loss = nn.MSELoss()(q_values, target)
    return loss


def save_model(model, path="vanilla_dqn.pth"):
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def main():
    # -------------------------
    # Hyperparameters
    # -------------------------
    HYPERPARAMS = {
        "seed": 42,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_episodes": 60000,
        "max_steps_per_episode": 1000,
        "batch_size": 128,
        "replay_capacity": 100000,
        "gamma": 0.99,
        "lr": 1e-5,
        "target_update_freq": 1000,   # in gradient steps
        "start_training_after": 1000, # steps
        "train_freq": 4,              # train every N steps
        "epsilon_start": 1.0,
        "epsilon_final": 0.02,
        "epsilon_decay_steps": 200000, #made it 1 mil bc tetris needs longer exploration
        "grad_clip": 10.0,
        "save_path": "vanilla_dqn.pth",
        "log_interval": 10,
    }
    
    print("using device:", HYPERPARAMS["device"])

    # -------------------------
    # Setup
    # -------------------------
    random.seed(HYPERPARAMS["seed"])
    np.random.seed(HYPERPARAMS["seed"])
    torch.manual_seed(HYPERPARAMS["seed"])

    device = torch.device(HYPERPARAMS["device"])
    print(f"Using device: {device}")

    # Instantiate environment wrapper
    env = TetrisBattleEnvWrapper(device=str(device), debug=False)

    # Model: use the same architecture file you already have (no architecture changes)
    num_actions = env.action_space.n if hasattr(env.action_space, "n") else int(env.action_space)
    model = VanillaDQN(num_actions=num_actions, in_channels=15).to(device)
    target_model = VanillaDQN(num_actions=num_actions, in_channels=15).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["lr"])
    replay = ReplayBuffer(HYPERPARAMS["replay_capacity"])

    total_steps = 0
    grad_steps = 0
    episode_rewards = []

    start_time = time.time()

    # Epsilon schedule helper
    def get_epsilon(step):
        eps_start = HYPERPARAMS["epsilon_start"]
        eps_final = HYPERPARAMS["epsilon_final"]
        eps_decay = HYPERPARAMS["epsilon_decay_steps"]
        return eps_final + (eps_start - eps_final) * max(0, (eps_decay - step) / eps_decay)

    # -------------------------
    # Training loop
    # -------------------------
    for ep in range(1, HYPERPARAMS["num_episodes"] + 1):
        state = env.reset()  # (C, H, W) tensor on device
        ep_reward = 0.0

        for step in range(HYPERPARAMS["max_steps_per_episode"]):
            epsilon = get_epsilon(total_steps)
            action = select_action(model, state, epsilon, device, valid_mask=None)

            next_state, reward, done, info = env.step(action)
            ep_reward += float(reward)

            # Store transition (ensure tensors are on CPU for replay to avoid GPU memory blowup)
            replay.push(state.cpu(), action, reward, next_state.cpu(), done)

            state = next_state
            total_steps += 1

            # Training step
            if len(replay) >= HYPERPARAMS["batch_size"] and total_steps > HYPERPARAMS["start_training_after"]:
                if total_steps % HYPERPARAMS["train_freq"] == 0:
                    batch = replay.sample(HYPERPARAMS["batch_size"])
                    loss = compute_td_loss(batch, model, target_model, HYPERPARAMS["gamma"], device)
                    optimizer.zero_grad()
                    loss.backward()
                    # gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), HYPERPARAMS["grad_clip"])
                    optimizer.step()
                    grad_steps += 1

                    # Hard update target network periodically
                    if grad_steps % HYPERPARAMS["target_update_freq"] == 0:
                        target_model.load_state_dict(model.state_dict())

            if done:
                break

        episode_rewards.append(ep_reward)

        # Logging
        if ep % HYPERPARAMS["log_interval"] == 0:
            avg_reward = np.mean(episode_rewards[-HYPERPARAMS["log_interval"] :])
            elapsed = time.time() - start_time
            print(f"Episode {ep:4d} | Steps {total_steps:7d} | AvgReward(last {HYPERPARAMS['log_interval']}): {avg_reward:.3f} | Eps: {epsilon:.3f} | Time: {elapsed:.1f}s")

        # Periodic save
        if ep % 100 == 0:
            save_model(model, path=f"checkpoint_ep{ep}.pth")

    # Final save
    save_model(model, path=HYPERPARAMS["save_path"])
    print("Training complete.")


if __name__ == "__main__":
    main()