"""Behavior cloning: train the shared policy to copy a teacher's recorded decisions.

Example:
  node scripts/record-demos.mjs --matches 40 --out demos/tuned
  python3 python/imitate.py --demos demos/tuned --name imitation1
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from soccer_env import SoccerVecEnv
from train_ppo import Policy, evaluate_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default="demos/tuned")
    parser.add_argument("--name", default="imitation1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--act-weight", type=float, default=5.0, help="scales the act loss, since acts are rare")
    parser.add_argument("--rare-boost", type=float, default=40.0,
                        help="how much more often rows where the teacher acts (shoot, tackle, save, pass) are sampled")
    parser.add_argument("--eval-episodes", type=int, default=4, help="matches per evaluation environment")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta = json.load(open(os.path.join(root, args.demos + ".json")))
    rows, d, c = meta["rows"], meta["obsSize"], meta["actCount"]
    obs = torch.from_numpy(np.fromfile(os.path.join(root, args.demos + ".obs"), dtype=np.float32).reshape(rows, d))
    mask = torch.from_numpy(np.fromfile(os.path.join(root, args.demos + ".mask"), dtype=np.uint8).reshape(rows, c).astype(bool))
    label = torch.from_numpy(np.fromfile(os.path.join(root, args.demos + ".label"), dtype=np.uint8).reshape(rows, 3).astype(np.int64))
    # Rows are stored match by match, so the last tenth is whole matches the policy never trains on.
    split = int(rows * 0.9)
    print(f"{rows} rows from {meta['matches']} matches ({meta['teacher']} teacher, goal diff {meta['teacherGoalDiff']:.2f}); "
          f"{split} train, {rows - split} validation")

    policy = Policy(d, c)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    steps_per_epoch = split // args.batch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)

    def losses(index):
        move, sprint, act, _ = policy(obs[index], mask[index])
        y = label[index]
        informative = mask[index].sum(-1) > 1   # rows where the act has a real choice
        act_loss = (-act.log_prob(y[:, 2]))[informative].mean() if informative.any() else torch.zeros(())
        total = -move.log_prob(y[:, 0]).mean() - sprint.log_prob(y[:, 1]).mean() + args.act_weight * act_loss
        accuracy = {
            "move": (move.probs.argmax(-1) == y[:, 0]).float().mean().item(),
            "sprint": (sprint.probs.argmax(-1) == y[:, 1]).float().mean().item(),
            "act": (act.probs.argmax(-1) == y[:, 2])[informative].float().mean().item() if informative.any() else float("nan"),
        }
        return total, accuracy

    # Shots, saves and passes are a few hundred rows among hundreds of thousands. Without oversampling the
    # cloned policy learns to never do them, and those are the actions that win matches.
    weights = torch.ones(split)
    weights[label[:split, 2] != 0] = args.rare_boost
    names = {1: "shoot L", 2: "shoot C", 3: "shoot R", 4: "tackle", 5: "save"}

    def recall():
        with torch.no_grad():
            _, _, act, _ = policy(obs[split:], mask[split:])
        top = act.probs.argmax(-1)
        y = label[split:, 2]
        parts = []
        for a, name in list(names.items()) + [(-1, "pass")]:
            sel = (y >= 6) if a == -1 else (y == a)
            if sel.sum() > 0:
                parts.append(f"{name} {(top[sel] == y[sel]).float().mean().item():.2f} (n={int(sel.sum())})")
        return "  ".join(parts)

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        order = torch.multinomial(weights, split, replacement=True)
        for step in range(steps_per_epoch):
            index = order[step * args.batch:(step + 1) * args.batch]
            loss, _ = losses(index)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        with torch.no_grad():
            val_index = torch.arange(split, rows)
            val_loss, val_acc = losses(val_index)
        print(f"epoch {epoch:2d}  val loss {val_loss.item():.3f}  accuracy move {val_acc['move']:.3f} sprint {val_acc['sprint']:.3f}"
              f" act (rows with a choice) {val_acc['act']:.3f}  [{time.time() - started:.0f}s]\n          recall: {recall()}", flush=True)

    run_dir = os.path.join(root, "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "args": vars(args), "iteration": 0}, os.path.join(run_dir, "latest.pt"))

    print("\nplaying the cloned policy against the CPU profiles (stochastic actions, 8 environments each)...")
    results = {}
    for opponent in ("default", "tuned"):
        env = SoccerVecEnv(2, 4, seed=5_000_000, opponent=opponent)
        results[opponent] = evaluate_policy(policy, env, torch.device("cpu"), args.eval_episodes)
        env.close()
        r = results[opponent]
        print(f"vs {opponent}: goal diff {r['goal_diff']:+.2f}  wins {r['win_rate']:.2f}  draws {r['draw_rate']:.2f}  ({r['episodes']} matches)")
    json.dump(results, open(os.path.join(run_dir, "eval.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
