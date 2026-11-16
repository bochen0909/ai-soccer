"""Scoring rate of a saved policy on a drill, over held-out seeds (the learner side alternates home and away).

Example:
  python3 python/evaluate_drill.py runs/role_gen15/latest.pt --drill shoot --episodes 512
"""
import argparse
import os

import numpy as np
import torch

from soccer_env import SoccerVecEnv
from train_ppo import Policy, act as sample_actions


def scoring_rate(policy, drill, episodes, seed, end_on_ball_loss, greedy):
    env = SoccerVecEnv(4, 4, seed=seed, drill=drill, maxSteps=60, endOnBallLoss=end_on_ball_loss,
                       reward={"possession": 0.0})
    each = -(-episodes // env.num_envs)
    batch = env.reset(seed=seed)
    counts = np.zeros(env.num_envs, dtype=int)
    goals = []
    while counts.min() < each:
        actions, _, _ = sample_actions(policy, batch, torch.device("cpu"), deterministic=greedy)
        batch = env.step(actions.numpy())
        for i in np.flatnonzero(batch.done & (batch.steps >= 0)):
            if counts[i] < each:
                goals.append(int(batch.score[i, 0] > 0))
            counts[i] += 1
    env.close()
    return np.array(goals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--drill", default="shoot")
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=900_000_000)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    checkpoint = torch.load(os.path.abspath(args.checkpoint))
    state = checkpoint["policy"]
    policy = Policy(state["body.0.weight"].shape[1], state["act.weight"].shape[0], hidden=state["body.0.weight"].shape[0])
    policy.load_state_dict(state)
    for label, loss in (("whole drill", False), ("ends on ball loss", True)):
        goals = scoring_rate(policy, args.drill, args.episodes, args.seed, loss, args.greedy)
        half = 1.96 * np.sqrt(goals.mean() * (1 - goals.mean()) / len(goals))
        print(f"{args.checkpoint} iter {checkpoint.get('iteration')} {args.drill} ({label}): "
              f"scored {goals.mean():.3f} +/- {half:.3f} over {len(goals)} episodes")


if __name__ == "__main__":
    main()
