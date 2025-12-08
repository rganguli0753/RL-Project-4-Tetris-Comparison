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
    raise ImportError(
        "Could not import TetrisContinuousActor/TetrisContinuousCritic from Models.continuous_dqn"
    ) from e

try:
    from tetris_env_wrapper import TetrisBattleEnvWrapper
except Exception as e:
    raise ImportError(
        "Could not import TetrisBattleEnvWrapper from tetris_env_wrapper.py"
    ) from e

# -------------------------
# Simple replay buffer for on-policy/off-policy agents
# -------------------------
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


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


H = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vanilla_episodes": 9000,
    "dueling_episodes": 9000,
    "double_episodes": 9000,
    "mctp_episodes": 9000,
    "continuous_episodes": 9000,
    "max_steps_per_episode": 10000,
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
    "epsilon_decay_steps": 1000000,
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
    action_dim = (
        env.action_space.n if hasattr(env.action_space, "n") else int(env.action_space)
    )

num_actions = (
    env.action_space.n if hasattr(env.action_space, "n") else int(env.action_space)
)


double_agent = DoubleDQNAgent(num_actions=num_actions, device=device, lr=1e-4)


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
# Train DoubleDQNAgent
# -------------------------
print("Starting DoubleDQNAgent training...")
for ep in range(1, H["double_episodes"] + 1):
    state = env.reset()
    ep_reward = 0.0

    for step in range(H["max_steps_per_episode"]):
        action = double_agent.act(state)
        next_state, reward, done, info = env.step(action)
        ep_reward += float(reward)

        double_agent.push(state.cpu(), action, reward, next_state.cpu(), done)
        _ = double_agent.train_step()

        state = next_state
        if done:
            break

    double_episode_rewards.append(ep_reward)

    if ep % 10 == 0:
        avg_reward = np.mean(double_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(
            f"[Double] Ep {ep:4d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | AgentEps: {double_agent.eps:.3f} | Time: {elapsed:.1f}s"
        )

    if ep % 100 == 0:
        torch.save(
            double_agent.online.state_dict(),
            f"checkpoints/checkpoint_double_ep{ep}.pth",
        )

torch.save(double_agent.online.state_dict(), H["double_save"])
print(f"Double DQN model saved to {H['double_save']}")
