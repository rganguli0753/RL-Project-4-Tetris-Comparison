"""
Train multiple agents (including the continuous actor/critic) on the same Tetris environment.

Assumptions:
 - tetris_env_wrapper.py defines TetrisBattleEnvWrapper
 - Model files are under Models/ with the expected class names:
     Models/vanilla_dqn.py      -> VanillaDQN
     Models/dueling_dqn.py     -> DuelingDQNAgent
     Models/prioritized_dqn.py -> PrioritizedTetrisDQN, PrioritizedReplayBuffer, per_dqn_train_step
     Models/double_dqn.py      -> DoubleDQNAgent
     Models/mctp_net.py        -> MCTPNet
     Models/continuous_dqn.py  -> TetrisContinuousActor, TetrisContinuousCritic
Adjust file/module names if your layout differs.
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
# Imports: models and env wrapper
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
    from Models.prioritized_dqn import PrioritizedReplayBuffer, PrioritizedTetrisDQN, per_dqn_train_step
except Exception as e:
    raise ImportError("Could not import prioritized model components from Models.prioritized_dqn") from e

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


def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


# -------------------------
# Hyperparameters
# -------------------------
H = {
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "vanilla_episodes": 60000,
    "dueling_episodes": 60000,
    "prioritized_episodes": 60000,
    "double_episodes": 60000,
    "mctp_episodes": 60000,
    "continuous_episodes": 60000,
    "max_steps_per_episode": 1000,
    "batch_size": 64,
    "replay_capacity": 100000,
    "per_capacity": 100000,
    "gamma": 0.99,
    "lr_vanilla": 1e-4,
    "lr_prioritized": 1e-4,
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
    "prioritized_save": "completed_models/prioritized_trained.pth",
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
    # fallback: discrete action count
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else int(env.action_space)

# Determine number of discrete actions for DQN agents
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
# Dueling DQN agent
# -------------------------
dueling_agent = DuelingDQNAgent(num_actions=num_actions, device=device, lr=1e-4)

# -------------------------
# Prioritized DQN setup
# -------------------------
prioritized_net = PrioritizedTetrisDQN(num_actions=num_actions, in_channels=15).to(device)
prioritized_target = PrioritizedTetrisDQN(num_actions=num_actions, in_channels=15).to(device)
prioritized_target.load_state_dict(prioritized_net.state_dict())
prioritized_target.eval()
prioritized_opt = optim.Adam(prioritized_net.parameters(), lr=H["lr_prioritized"])
per_buffer = PrioritizedReplayBuffer(capacity=H["per_capacity"])

# -------------------------
# Double DQN agent
# -------------------------
double_agent = DoubleDQNAgent(num_actions=num_actions, device=device, lr=1e-4)

# -------------------------
# MCTPNet (actor-critic) setup
# -------------------------
mctp_net = MCTPNet(in_channels=15, num_actions=num_actions).to(device)
mctp_opt = optim.Adam(mctp_net.parameters(), lr=H["lr_mctp"])

# -------------------------
# Continuous actor/critic (DDPG-style)
# -------------------------
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
prioritized_episode_rewards = []
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
    state = env.reset()  # (C, H, W) on device
    ep_reward = 0.0

    for step in range(H["max_steps_per_episode"]):
        epsilon = get_epsilon(total_steps)
        action = select_action_vanilla(vanilla_net, state, epsilon, device, num_actions, valid_mask=None)

        next_state, reward, done, info = env.step(action)
        ep_reward += float(reward)

        # store on CPU to avoid GPU memory growth
        vanilla_replay.push(state.cpu(), action, reward, next_state.cpu(), done)

        state = next_state
        total_steps += 1

        # Training step
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

    if ep % H["log_interval"] == 0:
        avg_reward = np.mean(vanilla_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(f"[Vanilla] Ep {ep:4d} | Steps {total_steps:7d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | Eps: {epsilon:.3f} | Time: {elapsed:.1f}s")

    if ep % 100 == 0:
        torch.save(vanilla_net.state_dict(), f"checkpoints/checkpoint_vanilla_ep{ep}.pth")

torch.save(vanilla_net.state_dict(), H["vanilla_save"])
print(f"Vanilla model saved to {H['vanilla_save']}")

# -------------------------
# Train DuelingDQNAgent
# -------------------------
print("Starting DuelingDQNAgent training...")
for ep in range(1, H["dueling_episodes"] + 1):
    state = env.reset()
    ep_reward = 0.0

    for step in range(H["max_steps_per_episode"]):
        action = dueling_agent.act(state)
        next_state, reward, done, info = env.step(action)
        ep_reward += float(reward)

        dueling_agent.push(state.cpu(), action, reward, next_state.cpu(), done)
        _ = dueling_agent.train_step()  # agent handles internal checks

        state = next_state
        if done:
            break

    dueling_episode_rewards.append(ep_reward)

    if ep % H["log_interval"] == 0:
        avg_reward = np.mean(dueling_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(f"[Dueling] Ep {ep:4d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | AgentEps: {dueling_agent.eps:.3f} | Time: {elapsed:.1f}s")

    if ep % 100 == 0:
        torch.save(dueling_agent.online.state_dict(), f"checkpoints/checkpoint_dueling_ep{ep}.pth")

torch.save(dueling_agent.online.state_dict(), H["dueling_save"])
print(f"Dueling model saved to {H['dueling_save']}")

# -------------------------
# Train Prioritized PER DQN
# -------------------------
print("Starting Prioritized PER DQN training...")
per_total_steps = 0
per_grad_steps = 0

for ep in range(1, H["prioritized_episodes"] + 1):
    state = env.reset()
    ep_reward = 0.0

    for step in range(H["max_steps_per_episode"]):
        # epsilon-greedy for prioritized agent (simple schedule)
        eps = get_epsilon(per_total_steps)
        if random.random() < eps:
            action = random.randrange(num_actions)
        else:
            prioritized_net.eval()
            with torch.no_grad():
                x = state.unsqueeze(0).to(device) if state.dim() == 3 else state.to(device)
                qv = prioritized_net(x).cpu().squeeze(0).numpy()
                action = int(np.argmax(qv))
            prioritized_net.train()

        next_state, reward, done, info = env.step(action)
        ep_reward += float(reward)

        # push into PER buffer (store CPU tensors)
        per_buffer.push(state.cpu(), action, reward, next_state.cpu(), done)

        state = next_state
        per_total_steps += 1

        # Train step when buffer has enough samples
        if len(per_buffer.buffer) >= H["batch_size"]:
            loss_val = per_dqn_train_step(
                online_net=prioritized_net.to(device),
                target_net=prioritized_target.to(device),
                buffer=per_buffer,
                optimizer=prioritized_opt,
                batch_size=H["batch_size"],
                gamma=H["gamma"],
            )
            per_grad_steps += 1

            # Hard update target periodically
            if per_grad_steps % H["target_update_freq"] == 0:
                prioritized_target.load_state_dict(prioritized_net.state_dict())

        if done:
            break

    prioritized_episode_rewards.append(ep_reward)

    if ep % H["log_interval"] == 0:
        avg_reward = np.mean(prioritized_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(f"[Prioritized] Ep {ep:4d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | Eps: {eps:.3f} | Time: {elapsed:.1f}s")

    if ep % 100 == 0:
        torch.save(prioritized_net.state_dict(), f"checkpoints/checkpoint_prioritized_ep{ep}.pth")

torch.save(prioritized_net.state_dict(), H["prioritized_save"])
print(f"Prioritized model saved to {H['prioritized_save']}")

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

    if ep % H["log_interval"] == 0:
        avg_reward = np.mean(double_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(f"[Double] Ep {ep:4d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | AgentEps: {double_agent.eps:.3f} | Time: {elapsed:.1f}s")

    if ep % 100 == 0:
        torch.save(double_agent.online.state_dict(), f"checkpoints/checkpoint_double_ep{ep}.pth")

torch.save(double_agent.online.state_dict(), H["double_save"])
print(f"Double DQN model saved to {H['double_save']}")

# -------------------------
# Train MCTPNet (actor-critic)
# -------------------------
print("Starting MCTPNet (actor-critic) training...")
for ep in range(1, H["mctp_episodes"] + 1):
    state = env.reset()
    ep_reward = 0.0

    log_probs = []
    values = []
    rewards = []
    dones = []

    for step in range(H["max_steps_per_episode"]):
        x = state.unsqueeze(0).to(device) if state.dim() == 3 else state.to(device)
        logits, value = mctp_net(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action, device=device))

        next_state, reward, done, info = env.step(action)
        ep_reward += float(reward)

        log_probs.append(log_prob)
        values.append(value.squeeze(0))
        rewards.append(float(reward))
        dones.append(done)

        state = next_state
        if done:
            break

    # compute returns
    returns = []
    R = 0.0
    for r, d in zip(reversed(rewards), reversed(dones)):
        if d:
            R = 0.0
        R = r + H["gamma"] * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)

    if len(values) > 0:
        values_tensor = torch.stack(values).squeeze(-1)
        log_probs_tensor = torch.stack(log_probs)
        advantages = returns - values_tensor.detach()
        policy_loss = -(log_probs_tensor * advantages).mean()
        value_loss = F.mse_loss(values_tensor, returns)
        loss = policy_loss + 0.5 * value_loss

        mctp_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mctp_net.parameters(), H["grad_clip"])
        mctp_opt.step()

    mctp_episode_rewards.append(ep_reward)

    if ep % H["log_interval"] == 0:
        avg_reward = np.mean(mctp_episode_rewards[-H["log_interval"] :])
        elapsed = time.time() - start_time
        print(f"[MCTP] Ep {ep:4d} | AvgR(last {H['log_interval']}): {avg_reward:.3f} | Time: {elapsed:.1f}s")

    if ep % 100 == 0:
        torch.save(mctp_net.state_dict(), f"checkpoints/checkpoint_mctp_ep{ep}.pth")

torch.save(mctp_net.state_dict(), H["mctp_save"])
print(f"MCTPNet model saved to {H['mctp_save']}")

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
        # select action: actor + exploration noise
        actor.eval()
        with torch.no_grad():
            x = state.unsqueeze(0).to(device) if state.dim() == 3 else state.to(device)
            mu, _ = actor(x)
            action = mu.cpu().squeeze(0).numpy()
        actor.train()

        # add gaussian noise for exploration
        noise = np.random.normal(scale=H["ddpg_noise_std"], size=action.shape)
        action_noisy = action + noise
        # convert to tensor for env.step; if env expects discrete index, cast to int
        if hasattr(env.action_space, "n"):
            # discrete environment: map continuous to discrete by argmax-like mapping
            # simple mapping: round and clip to [0, n-1]
            a_idx = int(np.clip(int(np.round(action_noisy[0])), 0, env.action_space.n - 1))
            chosen_action = a_idx
            action_tensor = torch.tensor([a_idx], dtype=torch.float32)
        else:
            chosen_action = action_noisy
            action_tensor = torch.tensor(action_noisy, dtype=torch.float32)

        next_state, reward, done, info = env.step(chosen_action)
        ep_reward += float(reward)

        # store transition (store action tensor on CPU)
        cont_replay.push(state.cpu(), action_tensor.cpu(), reward, next_state.cpu(), done)

        state = next_state
        cont_total_steps += 1

        # training step
        if len(cont_replay) >= H["batch_size"]:
            batch = cont_replay.sample(H["batch_size"])
            states = torch.stack(batch.state).to(device)            # (B, C, H, W)
            actions = torch.stack(batch.action).to(device).float()  # (B, action_dim) or (B,1)
            rewards = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
            next_states = torch.stack(batch.next_state).to(device)
            dones = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

            # Critic update: MSE between Q and target Q
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

            # Actor update: maximize Q (equivalently minimize -Q)
            actor_opt.zero_grad()
            mu_pred, _ = actor(states)
            actor_loss = -critic(states, mu_pred).mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), H["grad_clip"])
            actor_opt.step()

            # Soft update targets
            soft_update(actor_target, actor, H["ddpg_tau"])
            soft_update(critic_target, critic, H["ddpg_tau"])

            cont_grad_steps += 1

        if done:
            break

    continuous_episode_rewards.append(ep_reward)

    if ep % H["log_interval"] == 0:
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

print("All training complete.")