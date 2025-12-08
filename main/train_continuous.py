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
    
    
def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)
        
        
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


actor = TetrisContinuousActor(action_dim=action_dim, in_channels=15).to(device)
actor_target = TetrisContinuousActor(action_dim=action_dim, in_channels=15).to(device)
actor_target.load_state_dict(actor.state_dict())
actor_opt = optim.Adam(actor.parameters(), lr=H["lr_cont_actor"])

critic = TetrisContinuousCritic(action_dim=action_dim, in_channels=15).to(device)
critic_target = TetrisContinuousCritic(action_dim=action_dim, in_channels=15).to(device)
critic_target.load_state_dict(critic.state_dict())
critic_opt = optim.Adam(critic.parameters(), lr=H["lr_cont_critic"])

cont_replay = ReplayBuffer(H["replay_capacity"])


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
# Train continuous actor/critic (DDPG-style)
# -------------------------
print("Starting continuous actor/critic (DDPG-style) training...")
cont_total_steps = 0
cont_grad_steps = 0

for ep in range(1, H["continuous_episodes"] + 1):
    state = env.reset()
    ep_reward = 0.0

    for step in range(H["max_steps_per_episode"]):
        actor.eval()
        with torch.no_grad():
            x = state.unsqueeze(0).to(device) if state.dim() == 3 else state.to(device)
            mu, _ = actor(x)
            action = mu.cpu().squeeze(0).numpy()
        actor.train()

        noise = np.random.normal(scale=H["ddpg_noise_std"], size=action.shape)
        action_noisy = action + noise

        if hasattr(env.action_space, "n"):
            a_idx = int(np.clip(int(np.round(action_noisy[0])), 0, env.action_space.n - 1))
            chosen_action = a_idx
            action_tensor = torch.tensor([a_idx], dtype=torch.float32)
        else:
            chosen_action = action_noisy
            action_tensor = torch.tensor(action_noisy, dtype=torch.float32)

        next_state, reward, done, info = env.step(chosen_action)
        ep_reward += float(reward)

        cont_replay.push(state.cpu(), action_tensor.cpu(), reward, next_state.cpu(), done)

        state = next_state
        cont_total_steps += 1

        if len(cont_replay) >= H["batch_size"]:
            batch = cont_replay.sample(H["batch_size"])
            states = torch.stack(batch.state).to(device)
            actions = torch.stack(batch.action).to(device).float()
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.stack(batch.next_state).to(device)
            dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

            with torch.no_grad():
                next_actions_mu, _ = actor_target(next_states)
                next_q = critic_target(next_states, next_actions_mu)
                q_target = rewards + (1.0 - dones) * H["gamma"] * next_q

            q_pred = critic(states, actions)
            critic_loss = F.mse_loss(q_pred, q_target)

            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), H["grad_clip"])
            critic_opt.step()

            actor_opt.zero_grad()
            mu_pred, _ = actor(states)
            actor_loss = -critic(states, mu_pred).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), H["grad_clip"])
            actor_opt.step()

            soft_update(actor_target, actor, H["ddpg_tau"])
            soft_update(critic_target, critic, H["ddpg_tau"])

            cont_grad_steps += 1

        if done:
            break

    continuous_episode_rewards.append(ep_reward)

    if ep % 10 == 0:
        avg_reward = np.mean(continuous_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(f"[Continuous] Ep {ep:4d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | Time: {elapsed:.1f}s")

    if ep % 100 == 0:
        torch.save(actor.state_dict(), f"checkpoints/checkpoint_cont_actor_ep{ep}.pth")
        torch.save(critic.state_dict(), f"checkpoints/checkpoint_cont_critic_ep{ep}.pth")

torch.save(actor.state_dict(), H["continuous_actor_save"])
torch.save(critic.state_dict(), H["continuous_critic_save"])
print(f"Continuous actor saved to {H['continuous_actor_save']}")
print(f"Continuous critic saved to {H['continuous_critic_save']}")