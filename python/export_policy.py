"""Writes a trained policy as a JSON file the game can load (exact float32 weights, base64).

Example:
  python3 python/export_policy.py runs/ft1/latest.pt models/trained-5v5.json
  python3 python/export_policy.py runs/ft1/latest.pt models/trained-5v5.json --check test/fixtures/policy-check.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

from policy_io import write_policy_json
from train_ppo import load_policy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("out")
    parser.add_argument("--check", default=None, help="also write a test fixture: observation rows and the logits PyTorch gives")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)
    state = checkpoint["policy"]
    obs_size = state["body.0.weight"].shape[1]
    act_count = state["act.weight"].shape[0]
    size = write_policy_json(state, args.out, os.path.basename(os.path.dirname(args.checkpoint)), checkpoint.get("iteration"))
    print(f"wrote {args.out}: {obs_size} inputs, {state['body.0.weight'].shape[0]} hidden units, {act_count} acts, "
          f"{'with a goalkeeper network, ' if 'keeper_net.body.0.weight' in state else ''}{size / 1024:.0f} KB")

    if args.check:
        # Real observations from a short match, so the game's network can be tested against PyTorch exactly.
        from soccer_env import SoccerVecEnv
        policy = load_policy(state)
        env = SoccerVecEnv(1, 1, seed=11)
        batch = env.reset()
        rows = []
        for step in range(60):
            if step % 10 == 0:
                rows.append((batch.obs[0, 1 + step % 4].copy(), batch.mask[0, 1 + step % 4].copy()))
            actions = np.zeros((1, batch.obs.shape[1], 3), dtype=np.int32)
            batch = env.step(actions)
        env.close()
        obs = torch.from_numpy(np.stack([r[0] for r in rows]))
        mask = torch.from_numpy(np.stack([r[1] for r in rows]))
        with torch.no_grad():
            move, sprint, act, _ = policy(obs, mask)
        fixture = {"obs": obs.tolist(), "mask": mask.int().tolist(),
                   "moveLogits": move.logits.tolist(), "sprintLogits": sprint.logits.tolist(), "actLogits": act.logits.tolist()}
        os.makedirs(os.path.dirname(os.path.abspath(args.check)), exist_ok=True)
        json.dump(fixture, open(args.check, "w"))
        print(f"wrote {args.check}: {len(rows)} rows")


if __name__ == "__main__":
    main()
