"""How often a policy wins the ball back, or concedes, in the pure defend drills (a counterattack against us).

An episode ends when a learner player touches the ball (regained), a goal is scored, the ball leaves play, or after 6 s.

Example:
  python3 python/evaluate_defend.py runs/role_gen15/latest.pt --drill defend2
"""
import argparse
import os

import numpy as np
import torch

from soccer_env import SoccerVecEnv
from train_ppo import act as sample_actions, load_policy


def outcomes(path, drill, episodes, seed):
    policy = load_policy(torch.load(os.path.abspath(path))["policy"])
    env = SoccerVecEnv(4, 4, seed=seed, drillMix={"p": 1.0, "kinds": [drill], "maxSteps": 60}, pureDrills=[drill])
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
                rows.append((total[i] > 0.5, batch.score[i, 1] > 0, int(batch.steps[i])))
            counts[i] += 1
            total[i] = 0
    env.close()
    return np.array(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--drill", default="defend", choices=["defend", "defend2"])
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=900_000_000)
    args = parser.parse_args()
    for path in args.checkpoints:
        rows = outcomes(path, args.drill, args.episodes, args.seed)
        regained, conceded, steps = rows[:, 0].astype(bool), rows[:, 1].astype(bool), rows[:, 2]
        print(f"{path}: {len(rows)} {args.drill} episodes | regained the ball {regained.mean():.3f} (after a median {np.median(steps[regained]) / 10 if regained.any() else float('nan'):.1f} s) | "
              f"conceded {conceded.mean():.3f} | neither {(~regained & ~conceded).mean():.3f}")


if __name__ == "__main__":
    main()
