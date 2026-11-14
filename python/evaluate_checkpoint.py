"""Plays a saved policy against the CPU profiles and reports goal difference with a confidence interval.

Example:
  python3 python/evaluate_checkpoint.py runs/dagger2/latest.pt --matches 64
"""
import argparse
import os
import sys

import numpy as np
import torch

from soccer_env import SoccerVecEnv
from train_ppo import Policy, act as sample_actions


def play(policy, opponent, matches, seed, deterministic):
    processes, per_process = 4, 2
    env = SoccerVecEnv(processes, per_process, seed=seed, opponent=opponent)
    each = -(-matches // env.num_envs)
    batch = env.reset(seed=seed)
    counts = np.zeros(env.num_envs, dtype=int)
    diffs = []
    while counts.min() < each:
        actions, _, _ = sample_actions(policy, batch, torch.device("cpu"), deterministic=deterministic)
        batch = env.step(actions.numpy())
        for i in np.flatnonzero(batch.done & (batch.steps >= 0)):
            if counts[i] < each:
                diffs.append(int(batch.score[i, 0] - batch.score[i, 1]))
            counts[i] += 1
    env.close()
    return np.array(diffs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--matches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=900_000_000)
    parser.add_argument("--greedy", action="store_true", help="use the most likely action instead of sampling")
    parser.add_argument("--anchor", nargs="*", default=["../models/trained-5v5.json"], help="exported policies to also play against")
    args = parser.parse_args()
    path = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(os.getcwd(), args.checkpoint)
    checkpoint = torch.load(path)
    obs_size = checkpoint["policy"]["body.0.weight"].shape[1]
    act_count = checkpoint["policy"]["act.weight"].shape[0]
    policy = Policy(obs_size, act_count, hidden=checkpoint["policy"]["body.0.weight"].shape[0])
    policy.load_state_dict(checkpoint["policy"])
    print(f"{args.checkpoint}, iteration {checkpoint.get('iteration')}, {'greedy' if args.greedy else 'sampled'} actions, held-out seeds")
    for opponent in ("default", "tuned", *(f"model:{os.path.abspath(a)}" for a in args.anchor)):
        diffs = play(policy, opponent, args.matches, args.seed, args.greedy)
        half = 1.96 * diffs.std(ddof=1) / np.sqrt(len(diffs))
        print(f"  vs {opponent if ":" not in opponent else "model " + os.path.basename(opponent)[:-5]:12s} {len(diffs)} matches: goal diff {diffs.mean():+.2f} +/- {half:.2f}   "
              f"wins {np.mean(diffs > 0):.2f}  draws {np.mean(diffs == 0):.2f}  losses {np.mean(diffs < 0):.2f}")


if __name__ == "__main__":
    main()
