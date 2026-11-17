"""How long a policy keeps the ball in the pure rondo drill (four players around a small square, two rivals inside).

The episode ends when a rival touches the ball, the ball leaves play, or after 90 steps (9 s).

Example:
  python3 python/evaluate_rondo.py runs/role_gen19/latest.pt
"""
import argparse
import os

import numpy as np
import torch

from soccer_env import SoccerVecEnv
from train_ppo import act as sample_actions, load_policy


def lengths(path, episodes, seed):
    policy = load_policy(torch.load(os.path.abspath(path))["policy"])
    env = SoccerVecEnv(4, 4, seed=seed, drillMix={"p": 1.0, "kinds": ["rondo"], "maxSteps": 60}, pureDrills=["rondo"])
    each = -(-episodes // env.num_envs)
    batch = env.reset(seed=seed)
    counts = np.zeros(env.num_envs, dtype=int)
    steps = []
    while counts.min() < each:
        actions, _, _ = sample_actions(policy, batch, torch.device("cpu"))
        batch = env.step(actions.numpy())
        for i in np.flatnonzero(batch.done & (batch.steps >= 0)):
            if counts[i] < each:
                steps.append(int(batch.steps[i]))
            counts[i] += 1
    env.close()
    return np.array(steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=900_000_000)
    args = parser.parse_args()
    for path in args.checkpoints:
        steps = lengths(path, args.episodes, args.seed)
        print(f"{path}: {len(steps)} episodes | kept the ball a mean of {steps.mean():.1f} steps ({steps.mean() / 10:.1f} s) | "
              f"median {np.median(steps):.0f} | lasted the full 90 steps {np.mean(steps >= 90):.3f}")


if __name__ == "__main__":
    main()
