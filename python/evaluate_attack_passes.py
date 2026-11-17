"""Per-episode breakdown of a policy on the pure attack or pass drill: how often it completes a pass, and how often it scores with and without one.

Pure attack pays the pass bonus once for the first completed pass, and a goal (1) only after it, so the episode reward shows whether a pass happened.

Example:
  python3 python/evaluate_attack_passes.py runs/attack_pure_test/latest.pt runs/role_gen15/latest.pt
"""
import argparse
import os

import numpy as np
import torch

from soccer_env import SoccerVecEnv
from train_ppo import act as sample_actions, load_policy


# Pure rewards: pass drill pays 1 for the first completed pass; attack pays 0.5 for it and 0.5 for a goal after it.
THRESHOLD = {"pass": 0.5, "attack": 0.25}


def breakdown(path, episodes, seed, drill):
    state = torch.load(os.path.abspath(path))["policy"]
    policy = load_policy(state)
    env = SoccerVecEnv(4, 4, seed=seed, drill=drill, maxSteps=60, pureDrills=[drill])
    each = -(-episodes // env.num_envs)
    batch = env.reset(seed=seed)
    total = np.zeros(env.num_envs)
    counts = np.zeros(env.num_envs, dtype=int)
    rows = []
    while counts.min() < each:
        actions, _, _ = sample_actions(policy, batch, torch.device("cpu"))
        batch = env.step(actions.numpy())
        total += batch.reward
        for i in np.flatnonzero(batch.done & (batch.steps >= 0)):
            if counts[i] < each:
                goal = batch.score[i, 0] > 0
                rows.append((goal, total[i] > THRESHOLD[drill]))   # (scored, completed a pass)
            counts[i] += 1
            total[i] = 0
    env.close()
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--drill", default="attack", choices=["attack", "pass"])
    parser.add_argument("--seed", type=int, default=900_000_000)
    args = parser.parse_args()
    for path in args.checkpoints:
        rows = breakdown(path, args.episodes, args.seed, args.drill)
        goal, passed = rows[:, 0], rows[:, 1]
        print(f"{path}: {len(rows)} episodes | completed a pass {passed.mean():.3f} | scored {goal.mean():.3f} | "
              f"scored after a pass {(goal & passed).mean():.3f} | scored without a pass {(goal & ~passed).mean():.3f} | "
              f"pass but no goal {(~goal & passed).mean():.3f}")


if __name__ == "__main__":
    main()
