"""
Evaluate trained DQN models on the Tetris environment.

Computes the following metrics for each model:
- Average episode reward
- Standard deviation of rewards
- Min/Max episode reward
- Average steps per episode
- Model inference time
- Model parameter count
- Total model size
"""

import json
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# -------------------------
# Imports: models and env wrapper (no prioritized model)
# -------------------------
from Models.vanilla_dqn import VanillaDQN
from Models.dueling_dqn import DuelingTetrisDQN
from Models.double_dqn import TetrisDQN as DoubleDQN
from Models.mctp_model import MCTPNet
from Models.continuous_action import TetrisContinuousActor
from tetris_env_wrapper import TetrisBattleEnvWrapper

MODEL_REGISTRY = {
    "vanilla": {
        "class": VanillaDQN,
        "kwargs": lambda na, ad, device=None: {"num_actions": na, "in_channels": 15},
        "path": "../completed_models/vanilla_trained.pth",
    },
    "dueling": {
        "class": DuelingTetrisDQN,
        "kwargs": lambda na, ad, device: {
            "num_actions": na,
            "in_channels": 17,
            "device": device,
        },
        "path": "../completed_models/dueling_trained.pth",
    },
    "double": {
        "class": DoubleDQN,
        "kwargs": lambda na, ad, device=None: {"num_actions": na, "in_channels": 15},
        "path": "../completed_models/double_trained.pth",
    },
    "mctp": {
        "class": MCTPNet,
        "kwargs": lambda na, ad, device=None: {"num_actions": na, "in_channels": 15},
        "path": "../completed_models/mctp_trained.pth",
    },
    "continuous_actor": {
        "class": TetrisContinuousActor,
        "kwargs": lambda na, ad, device=None: {"action_dim": ad, "in_channels": 15},
        "path": "../completed_models/continuous_actor.pth",
    },
}


# -------------------------
# Utility Functions
# -------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def evaluate_inference_time(
    model, env, num_steps=100, device="cuda", model_name="Unknown"
):
    """Compute average inference time per step in milliseconds."""
    state = env.reset()
    times = []
    model.eval()

    with torch.no_grad():
        for _ in range(num_steps):
            start = time.time()

            x = state.unsqueeze(0).to(device) if state.dim() == 3 else state.to(device)
            model(x)
            times.append((time.time() - start) * 1000)  # Convert to ms

            next_state, _, done, _ = env.step(env.action_space.sample())
            state = env.reset() if done else next_state

    return np.mean(times)


def evaluate_model(model, env, num_episodes=50, device="cuda", model_name="Unknown"):
    """Evaluate a trained model on the environment."""
    rewards, episode_lengths = [], []
    model.eval()

    print(f"\n[{model_name}] Evaluating on {num_episodes} episodes...")
    cv2.namedWindow("Agent", cv2.WINDOW_NORMAL)
    for episode in range(num_episodes):
        state = env.reset()
        ep_reward, ep_length = 0.0, 0.0

        for step in range(5000):  # Max steps per episode

            with torch.no_grad():
                x = (
                    state.unsqueeze(0).to(device)
                    if state.dim() == 3
                    else state.to(device)
                )

                # Select action based on agent type
                if isinstance(model, (VanillaDQN, DuelingTetrisDQN, DoubleDQN)):
                    q_values = model(x).cpu().squeeze(0).numpy()
                    action = int(np.argmax(q_values))

                elif isinstance(model, MCTPNet):
                    logits, _ = model(x)
                    probs = F.softmax(logits, dim=-1)
                    action = int(np.argmax(probs.cpu().numpy().squeeze(0)))

                elif isinstance(model, TetrisContinuousActor):
                    mu, _ = model(x)
                    action_continuous = mu.cpu().squeeze(0).numpy()
                    if hasattr(env.action_space, "n"):
                        action = int(
                            np.clip(
                                int(np.round(action_continuous[0])),
                                0,
                                env.action_space.n - 1,
                            )
                        )
                    else:
                        action = action_continuous

            # action = int(random.randrange(8))
            next_state, reward, done, _ = env.step(action)

            if step % 10 == 0 or action != 0:
                frame = env.render()
                cv2.imshow("Agent", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

            ep_reward += float(reward)
            # print("Cumulative Episode Reward: " + str(ep_reward))
            ep_length += 1
            state = next_state

            if done:
                break

        rewards.append(ep_reward)
        print("Episode reward: " + str(ep_reward))
        episode_lengths.append(ep_length)

        print(f"  Completed {episode + 1}/{num_episodes} episodes")

    metrics = {
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "avg_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
    }
    return metrics


def load_model(model_class, model_path, device="cuda", **kwargs):
    print(model_class)
    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def gather_metrics(model_name, model_class, model_path, env, device="cuda", **kwargs):
    """Load model, evaluate inference and gameplay, return full metrics dict."""
    model = load_model(model_class, model_path, device, **kwargs)
    params = count_parameters(model)
    size = get_model_size_mb(model)
    infer_time = evaluate_inference_time(
        model, env, device=device, model_name=model_name
    )
    gameplay_metrics = evaluate_model(model, env, device=device, model_name=model_name)
    gameplay_metrics.update(
        {
            "parameters": params,
            "model_size_mb": size,
            "inference_time_ms": infer_time,
            "model_name": model_name,
        }
    )
    return gameplay_metrics


def evaluate_all_models(num_eval_episodes=50, results_file="evaluation_results.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = TetrisBattleEnvWrapper(device=device, debug=False)

    if hasattr(env.action_space, "n"):
        num_actions = env.action_space.n
    elif hasattr(env.action_space, "shape") and env.action_space.shape is not None:
        num_actions = int(np.prod(env.action_space.shape))
    else:
        num_actions = int(env.action_space)
    action_dim = num_actions

    # Load existing results if file exists
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "models": {}}

    for key, entry in MODEL_REGISTRY.items():
        path = entry["path"]
        if not os.path.exists(path):
            print(f"Warning: {key} model file not found at {path}")
            continue
        print(f"\nEvaluating {key} model...")
        metrics = gather_metrics(
            key,
            entry["class"],
            path,
            env,
            device=device,
            **entry["kwargs"](num_actions, action_dim, device=device),
        )
        all_results["models"][path] = metrics
        all_results["models"][path]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"Avg reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")

    # Save updated results
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained DQN models on Tetris environment"
    )
    parser.add_argument("--model", type=str, help="Path to a single model file")
    args = parser.parse_args()

    if args.model:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Model file not found at {args.model}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        env = TetrisBattleEnvWrapper(device=str(device), debug=False)

        # Infer model type
        filename = os.path.basename(args.model).lower()
        for key, entry in MODEL_REGISTRY.items():
            if key in filename:
                model_name, model_class = key, entry["class"]
                kwargs = entry["kwargs"]
                break
        else:
            raise ValueError(f"Could not detect model type from filename: {filename}")

        if hasattr(env.action_space, "n"):
            num_actions = env.action_space.n
        elif hasattr(env.action_space, "shape") and env.action_space.shape is not None:
            num_actions = int(np.prod(env.action_space.shape))
        else:
            num_actions = int(env.action_space)
        action_dim = num_actions

        metrics = gather_metrics(
            model_name,
            model_class,
            args.model,
            env,
            **kwargs(num_actions, action_dim, device=device),
        )

        # Load existing results if file exists
        results_file = "evaluation_results.json"
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                all_results = json.load(f)
            if "models" not in all_results:
                all_results["models"] = {}
        else:
            all_results = {
                "models": {},
            }

        # Add/update metrics for the single model
        all_results["models"][args.model] = metrics
        all_results["models"][args.model]["timestamp"] = time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Save back
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to evaluation_results.json")
    else:
        evaluate_all_models()


if __name__ == "__main__":
    main()
