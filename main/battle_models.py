"""
Round-Robin Tournament for Tetris Battle Models
Runs every model against every other model in head-to-head matches.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2

from Models.vanilla_dqn import VanillaDQN
from Models.dueling_dqn import DuelingTetrisDQN
from Models.double_dqn import TetrisDQN as DoubleDQN
from Models.mctp_model import MCTPNet
from Models.continuous_action import TetrisContinuousActor
from tetris_env_wrapper import TetrisBattleEnvWrapper

MODEL_REGISTRY = {
    "vanilla": {
        "class": VanillaDQN,
        "path": "../completed_models/vanilla_trained.pth",
        "kwargs": lambda n: {"num_actions": n, "in_channels": 15},
    },
    "dueling": {
        "class": DuelingTetrisDQN,
        "path": "../completed_models/dueling_trained.pth",
        "kwargs": lambda n: {"num_actions": n, "in_channels": 15},
    },
    "double": {
        "class": DoubleDQN,
        "path": "../completed_models/double_trained.pth",
        "kwargs": lambda n: {"num_actions": n, "in_channels": 15},
    },
    "mctp": {
        "class": MCTPNet,
        "path": "../completed_models/mctp_trained.pth",
        "kwargs": lambda n: {"num_actions": n, "in_channels": 15},
    },
}


def load_model(model_name, model_info, num_actions, device):

    model_class = model_info["class"]
    model_path = model_info["path"]
    kwargs = model_info["kwargs"](num_actions)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = model_class(**kwargs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Loaded {model_name} from {model_path}")
    return model


def play_match(
    model1, model2, model1_name, model2_name, device, visualize=False, max_steps=20000
):
    """
    Play a head-to-head match between two models.
    Model 1 controls player 0, Model 2 controls player 1.
    """
    env = TetrisBattleEnvWrapper(device=device, debug=False)
    models = {0: model1, 1: model2}

    num_episodes = 10
    stats = {
        model1_name: {
            "lines_cleared": 0,
            "lines_sent": 0,
            "actions_taken": 0,
            "total_reward": 0.0,
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_wins": 0,
            "episode_losses": 0,
            "episode_draws": 0,
        },
        model2_name: {
            "lines_cleared": 0,
            "lines_sent": 0,
            "actions_taken": 0,
            "total_reward": 0.0,
            "episode_rewards": [],
            "episode_lengths": [],
            "episode_wins": 0,
            "episode_losses": 0,
            "episode_draws": 0,
        },
    }

    for episode in range(num_episodes):
        state = env.reset()
        ep_rewards = {0: 0.0, 1: 0.0}
        ep_length = 0
        done = False

        while not done and ep_length < max_steps:
            player_idx = env.env.game_interface.now_player
            model = models[player_idx]

            with torch.no_grad():
                x = (
                    state.unsqueeze(0).to(device)
                    if state.ndim == 3
                    else state.to(device)
                )

                # Select action based on model type
                if isinstance(model, (VanillaDQN, DuelingTetrisDQN, DoubleDQN)):
                    q_values = model(x).cpu().squeeze(0).numpy()
                    action = int(np.argmax(q_values))
                elif isinstance(model, MCTPNet):
                    logits, _ = model(x)
                    probs = F.softmax(logits, dim=-1)
                    action = int(np.argmax(probs.cpu().numpy().squeeze(0)))
                elif isinstance(model, TetrisContinuousActor):
                    mu, _ = model(x)
                    action_cont = mu.cpu().squeeze(0).numpy()
                    action = int(
                        np.clip(int(round(action_cont[0])), 0, env.action_space.n - 1)
                    )

            # Take step
            next_state, reward, done, info = env.step(action)
            env.env.take_turns()
            ep_rewards[player_idx] += reward
            stats[model1_name if player_idx == 0 else model2_name]["actions_taken"] += 1
            ep_length += 1
            state = next_state

            # Optional visualization
            if visualize and (ep_length % 10 == 0 or action != 0):
                frame = env.render()
                cv2.imshow("Battle", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)

        # Update episode stats
        for idx, name in zip([0, 1], [model1_name, model2_name]):
            tetris_obj = env.env.game_interface.tetris_list[idx]["tetris"]
            stats[name]["lines_cleared"] += getattr(tetris_obj, "cleared", 0)
            stats[name]["lines_sent"] += getattr(tetris_obj, "sent", 0)
            stats[name]["total_reward"] += ep_rewards[idx]
            stats[name]["episode_rewards"].append(ep_rewards[idx])
            stats[name]["episode_lengths"].append(ep_length)

        # Track episode winner
        winner = info.get("winner", None)
        if winner == 0:
            stats[model1_name]["episode_wins"] += 1
            stats[model2_name]["episode_losses"] += 1
        elif winner == 1:
            stats[model2_name]["episode_wins"] += 1
            stats[model1_name]["episode_losses"] += 1
        else:
            stats[model1_name]["episode_draws"] += 1
            stats[model2_name]["episode_draws"] += 1

        print(
            f"Episode {episode+1}/{num_episodes} | Rewards: {ep_rewards} | Winner: {winner}"
        )

    # Compute derived stats
    for name in [model1_name, model2_name]:
        rewards = np.array(stats[name]["episode_rewards"])
        lengths = np.array(stats[name]["episode_lengths"])
        stats[name].update(
            {
                "avg_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
                "min_reward": float(np.min(rewards)),
                "max_reward": float(np.max(rewards)),
                "avg_episode_length": float(np.mean(lengths)),
                "std_episode_length": float(np.std(lengths)),
                "min_episode_length": int(np.min(lengths)),
                "max_episode_length": int(np.max(lengths)),
            }
        )

    env.close()
    return {
        "stats": stats,
    }


def run_tournament(
    num_matches_per_pair=1, visualize=False, results_file="tournament_results.json"
):
    """
    Run a round-robin tournament where each model plays against every other model.

    Args:
        num_matches_per_pair: Number of matches per model pair.
        visualize: Whether to display matches.
        results_file: File to save results.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    num_actions = 8

    # Load models
    models = {}
    for name, info in MODEL_REGISTRY.items():
        try:
            models[name] = load_model(name, info, num_actions, device)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    if len(models) < 2:
        print("Error: At least 2 models required")
        return

    # Initialize stats
    stats = {
        "tournament_info": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_models": len(models),
            "matches_per_pair": num_matches_per_pair,
            "models": list(models.keys()),
        },
        "standings": {
            name: {
                "episode_wins": 0,
                "episode_losses": 0,
                "episode_draws": 0,
                "total_lines_cleared": 0,
                "total_lines_sent": 0,
                "total_actions_taken": 0,
                "total_reward": 0.0,
                "matches_played": 0,
            }
            for name in models
        },
        "matchups": {},
    }

    from itertools import combinations

    model_pairs = list(combinations(models.keys(), 2))
    total_matches = len(model_pairs) * num_matches_per_pair
    match_count = 0

    for model1_name, model2_name in model_pairs:
        matchup_key = f"{model1_name}_vs_{model2_name}"
        stats["matchups"][matchup_key] = []

        print(f"\nMATCHUP: {model1_name} vs {model2_name}")

        for match_num in range(num_matches_per_pair):
            match_count += 1
            print(
                f"\nMatch {match_num + 1}/{num_matches_per_pair} (Overall: {match_count}/{total_matches})"
            )

            match_result = play_match(
                models[model1_name],
                models[model2_name],
                model1_name,
                model2_name,
                device,
                visualize=visualize,
            )

            stats["matchups"][matchup_key].append(match_result)

            # Update standings
            for name in [model1_name, model2_name]:
                s = match_result["stats"][name]
                standing = stats["standings"][name]

                standing["episode_wins"] += s["episode_wins"]
                standing["episode_losses"] += s["episode_losses"]
                standing["episode_draws"] += s["episode_draws"]
                standing["total_lines_cleared"] += s["lines_cleared"]
                standing["total_lines_sent"] += s["lines_sent"]
                standing["total_actions_taken"] += s["actions_taken"]
                standing["total_reward"] += s["total_reward"]
                standing["matches_played"] += 1

    # Compute averages for standings
    for name, s in stats["standings"].items():
        mp = max(s["matches_played"], 1)
        s["avg_lines_cleared"] = s["total_lines_cleared"] / mp
        s["avg_lines_sent"] = s["total_lines_sent"] / mp
        s["avg_actions_taken"] = s["total_actions_taken"] / mp
        s["avg_reward"] = s["total_reward"] / mp

    # Sort standings
    sorted_standings = sorted(
        stats["standings"].items(),
        key=lambda x: (
            x[1]["episode_wins"],
            -x[1]["episode_losses"],
            x[1]["episode_draws"],
        ),
        reverse=True,
    )

    # Print standings
    print("\nFINAL STANDINGS")
    print(
        f"{'Rank':<6}{'Model':<12}{'W-L-D':<10}{'Avg Lines Sent':<15}{'Avg Lines Cleared':<15}{'Avg Reward':<12}"
    )
    for rank, (name, s) in enumerate(sorted_standings, 1):
        print(
            f"{rank:<6}{name:<12}{s['episode_wins']}-{s['episode_losses']}-{s['episode_draws']:<10}{s['avg_lines_sent']:<15.2f}{s['avg_lines_cleared']:<15.2f}{s['avg_reward']:<12.2f}"
        )

    # Save to file
    with open(results_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Tetris Model Tournament")
    parser.add_argument("--visualize", action="store_true", help="Display matches")
    args = parser.parse_args()

    run_tournament(
        visualize=args.visualize,
        results_file="tournament_results.json",
    )
