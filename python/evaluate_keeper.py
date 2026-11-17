"""Saves and goals conceded by a policy's keeper in the pure keeper drill (a lone rival striker shoots from 9 to 18 m).

An episode ends on the keeper's first catch or parry, a goal, or the ball leaving play, or after 6 s.

Example:
  python3 python/evaluate_keeper.py runs/role_gen15/latest.pt
"""
import argparse
import os

import numpy as np
import torch

from soccer_env import SoccerVecEnv
from train_ppo import act as sample_actions, load_policy


def breakdown(path, episodes, seed):
    state = torch.load(os.path.abspath(path))["policy"]
    policy = load_policy(state)
    env = SoccerVecEnv(4, 4, seed=seed, drill="keeper", maxSteps=60, pureDrills=["keeper"])
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
                rows.append((total[i] > 0.5, batch.score[i, 1] > 0))   # (saved, conceded)
            counts[i] += 1
            total[i] = 0
    env.close()
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=900_000_000)
    args = parser.parse_args()
    for path in args.checkpoints:
        rows = breakdown(path, args.episodes, args.seed)
        saved, conceded = rows[:, 0], rows[:, 1]
        print(f"{path}: {len(rows)} episodes | saved {saved.mean():.3f} | conceded {conceded.mean():.3f} | neither {(~saved & ~conceded).mean():.3f}")


if __name__ == "__main__":
    main()
