"""
Train multiple agents on the same Tetris environment (no prioritized replay).
Trains and saves:
 - VanillaDQN (Models/vanilla_dqn.VanillaDQN)
 - DuelingDQNAgent (Models/dueling_dqn.DuelingDQNAgent)
 - DoubleDQNAgent (Models/double_dqn.DoubleDQNAgent)
 - MCTPNet (Models/mctp_net.MCTPNet)
 - Continuous actor/critic (Models/continuous_dqn.TetrisContinuousActor/Critic)

Assumes:
 - tetris_env_wrapper.py defines TetrisBattleEnvWrapper
 - Model files exist under Models/ with the expected class names
"""

import os
import random
import time
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# Imports: models and env wrapper (no prioritized model)
# -------------------------
try:
    from Models.vanilla_dqn import VanillaDQN
except Exception as e:
    raise ImportError("Could not import VanillaDQN from Models.vanilla_dqn") from e

try:
    from Models.dueling_dqn import DuelingDQNAgent
except Exception as e:
    raise ImportError("Could not import DuelingDQNAgent from Models.dueling_dqn") from e

try:
    from Models.double_dqn import DoubleDQNAgent
except Exception as e:
    raise ImportError("Could not import DoubleDQNAgent from Models.double_dqn") from e

try:
    from Models.mctp_model import MCTPNet
except Exception as e:
    raise ImportError("Could not import MCTPNet from Models.mctp_net") from e

try:
    from Models.continuous_action import TetrisContinuousActor, TetrisContinuousCritic
except Exception as e:
    raise ImportError("Could not import TetrisContinuousActor/TetrisContinuousCritic from Models.continuous_dqn") from e

try:
    from tetris_env_wrapper import TetrisBattleEnvWrapper
except Exception as e:
    raise ImportError("Could not import TetrisBattleEnvWrapper from tetris_env_wrapper.py") from e

# -------------------------
# Simple replay buffer for on-policy/off-policy agents
# -------------------------
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


# -------------------------
# Helpers for VanillaDQN training
# -------------------------
def select_action_vanilla(q_net, state, epsilon, device, num_actions, valid_mask=None):
    if random.random() < epsilon:
        if valid_mask is not None:
            mask = valid_mask.cpu().numpy().astype(bool) if isinstance(valid_mask, torch.Tensor) else np.array(valid_mask).astype(bool)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                return random.randrange(num_actions)
            return int(np.random.choice(valid_indices))
        return int(random.randrange(num_actions))
    else:
        q_net.eval()
        with torch.no_grad():
            x = state.unsqueeze(0).to(device) if state.dim() == 3 else state.to(device)
            q_values = q_net(x).cpu().squeeze(0).numpy()
            if valid_mask is not None:
                mask = np.array(valid_mask).astype(bool)
                q_values[~mask] = -1e9
            action = int(np.argmax(q_values))
        q_net.train()
        return action


def compute_td_loss_vanilla(batch, q_net, target_net, gamma, device):
    states = torch.stack(batch.state).to(device)
    actions = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
    rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.stack(batch.next_state).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

    q_values = q_net(states).gather(1, actions)

    with torch.no_grad():
        next_q = target_net(next_states)
        max_next_q = next_q.max(dim=1, keepdim=True)[0]
        target = rewards + (1.0 - dones) * gamma * max_next_q

    loss = nn.MSELoss()(q_values, target)
    return loss


# -------------------------
# Continuous agent helpers (DDPG-style)
# -------------------------
def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


# -------------------------
# Hyperparameters
# -------------------------
H = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vanilla_episodes": 9000,
    "dueling_episodes": 9000,
    "double_episodes": 9000,
    "mctp_episodes": 9000,
    "continuous_episodes": 9000,
    "max_steps_per_episode": 1000,
    "batch_size": 64,
    "replay_capacity": 100000,
    "gamma": 0.99,
    "lr_vanilla": 1e-4,
    "lr_mctp": 3e-4,
    "lr_cont_actor": 1e-4,
    "lr_cont_critic": 1e-4,
    "target_update_freq": 1000,
    "start_training_after": 1000,
    "train_freq": 4,
    "epsilon_start": 1.0,
    "epsilon_final": 0.02,
    "epsilon_decay_steps": 200000,
    "grad_clip": 10.0,
    "vanilla_save": "completed_models/vanilla_trained.pth",
    "dueling_save": "completed_models/dueling_trained.pth",
    "double_save": "completed_models/double_trained.pth",
    "mctp_save": "completed_models/mctp_trained.pth",
    "continuous_actor_save": "completed_models/continuous_actor.pth",
    "continuous_critic_save": "completed_models/continuous_critic.pth",
    "log_interval": 10,
    "ddpg_tau": 0.005,
    "ddpg_noise_std": 0.1,
}

# -------------------------
# Setup
# -------------------------
random.seed(H["seed"])
np.random.seed(H["seed"])
torch.manual_seed(H["seed"])

device = torch.device(H["device"])
print(f"Using device: {device}")

# Instantiate environment wrapper
env = TetrisBattleEnvWrapper(device=str(device), debug=False)

# Determine action space size
if hasattr(env.action_space, "shape") and env.action_space.shape is not None:
    action_dim = int(np.prod(env.action_space.shape))
else:
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else int(env.action_space)

num_actions = env.action_space.n if hasattr(env.action_space, "n") else int(env.action_space)

# -------------------------
# Vanilla DQN setup
# -------------------------
vanilla_net = VanillaDQN(num_actions=num_actions, in_channels=15).to(device)
vanilla_target = VanillaDQN(num_actions=num_actions, in_channels=15).to(device)
vanilla_target.load_state_dict(vanilla_net.state_dict())
vanilla_target.eval()
vanilla_opt = optim.Adam(vanilla_net.parameters(), lr=H["lr_vanilla"])
vanilla_replay = ReplayBuffer(H["replay_capacity"])


# -------------------------
# Training bookkeeping
# -------------------------
total_steps = 0
vanilla_grad_steps = 0
vanilla_episode_rewards = []
dueling_episode_rewards = []
double_episode_rewards = []
mctp_episode_rewards = []
continuous_episode_rewards = []
start_time = time.time()


def get_epsilon(step):
    eps_start = H["epsilon_start"]
    eps_final = H["epsilon_final"]
    eps_decay = H["epsilon_decay_steps"]
    return eps_final + (eps_start - eps_final) * max(0, (eps_decay - step) / eps_decay)


# -------------------------
# Train VanillaDQN
# -------------------------
print("Starting VanillaDQN training...")
for ep in range(1, H["vanilla_episodes"] + 1):
    state = env.reset()
    ep_reward = 0.0

    for step in range(H["max_steps_per_episode"]):
        epsilon = get_epsilon(total_steps)
        action = select_action_vanilla(vanilla_net, state, epsilon, device, num_actions, valid_mask=None)

        next_state, reward, done, info = env.step(action)
        ep_reward += float(reward)

        vanilla_replay.push(state.cpu(), action, reward, next_state.cpu(), done)

        state = next_state
        total_steps += 1

        if len(vanilla_replay) >= H["batch_size"] and total_steps > H["start_training_after"]:
            if total_steps % H["train_freq"] == 0:
                batch = vanilla_replay.sample(H["batch_size"])
                loss = compute_td_loss_vanilla(batch, vanilla_net, vanilla_target, H["gamma"], device)
                vanilla_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(vanilla_net.parameters(), H["grad_clip"])
                vanilla_opt.step()
                vanilla_grad_steps += 1

                if vanilla_grad_steps % H["target_update_freq"] == 0:
                    vanilla_target.load_state_dict(vanilla_net.state_dict())

        if done:
            break

    vanilla_episode_rewards.append(ep_reward)

    if ep % 10 == 0:
        avg_reward = np.mean(vanilla_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(f"[Vanilla] Ep {ep:4d} | Steps {total_steps:7d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | Eps: {epsilon:.3f} | Time: {elapsed:.1f}s")

    if ep % 100 == 0:
        torch.save(vanilla_net.state_dict(), f"checkpoints/checkpoint_vanilla_ep{ep}.pth")

torch.save(vanilla_net.state_dict(), H["vanilla_save"])
print(f"Vanilla model saved to {H['vanilla_save']}")