"""DAgger: imitation that trains on the states the student itself reaches.

Plain behavior cloning only sees the states a teacher visits, so small mistakes put the student in situations it
never trained on. Here the student plays (sometimes with the teacher's action mixed in), the teacher labels every
state the student visits, and the student retrains on the growing dataset.

Example:
  python3 python/dagger.py --demos demos/tuned --name dagger1 --rounds 8
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from soccer_env import SoccerVecEnv
from train_ppo import Policy, act as sample_actions, evaluate_policy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_demos(prefix):
    meta = json.load(open(os.path.join(ROOT, prefix + ".json")))
    rows, d, c = meta["rows"], meta["obsSize"], meta["actCount"]
    obs = torch.from_numpy(np.fromfile(os.path.join(ROOT, prefix + ".obs"), dtype=np.float32).reshape(rows, d))
    mask = torch.from_numpy(np.fromfile(os.path.join(ROOT, prefix + ".mask"), dtype=np.uint8).reshape(rows, c).astype(bool))
    label = torch.from_numpy(np.fromfile(os.path.join(ROOT, prefix + ".label"), dtype=np.uint8).reshape(rows, 3).astype(np.int64))
    return obs, mask, label, meta


def fit(policy, optimizer, obs, mask, label, epochs, batch, rare_boost, act_weight):
    """Supervised passes over the dataset. Rows where the teacher acts are oversampled."""
    n = obs.shape[0]
    weights = torch.ones(n)
    weights[label[:, 2] != 0] = rare_boost
    steps = max(1, n // batch)
    last = 0.0
    for _ in range(epochs):
        order = torch.multinomial(weights, n, replacement=True)
        for step in range(steps):
            index = order[step * batch:(step + 1) * batch]
            move, sprint, act, _ = policy(obs[index], mask[index])
            y = label[index]
            informative = mask[index].sum(-1) > 1
            act_loss = (-act.log_prob(y[:, 2]))[informative].mean() if informative.any() else torch.zeros(())
            loss = -move.log_prob(y[:, 0]).mean() - sprint.log_prob(y[:, 1]).mean() + act_weight * act_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last = loss.item()
    return last


def rollout(envs, policy, steps, beta, rng):
    """Student plays; each agent step uses the teacher's action with probability beta. Returns teacher-labelled rows."""
    batches = [env.reset() for env in envs]
    rows_obs, rows_mask, rows_label = [], [], []
    mixed = 0
    total = 0
    for _ in range(steps):
        for index, env in enumerate(envs):
            batch = batches[index]
            actions, _, _ = sample_actions(policy, batch, torch.device("cpu"))
            actions = actions.numpy().astype(np.int32)
            keep = batch.agent_mask
            rows_obs.append(torch.from_numpy(batch.obs[keep]))
            rows_mask.append(torch.from_numpy(batch.mask[keep]))
            rows_label.append(torch.from_numpy(batch.teacher[keep].astype(np.int64)))
            use_teacher = rng.random(batch.agent_mask.shape) < beta
            actions = np.where(use_teacher[..., None], batch.teacher.astype(np.int32), actions)
            mixed += int(use_teacher[keep].sum()); total += int(keep.sum())
            batches[index] = env.step(actions)
    return torch.cat(rows_obs), torch.cat(rows_mask), torch.cat(rows_label), mixed / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demos", default="demos/tuned")
    parser.add_argument("--name", default="dagger1")
    parser.add_argument("--teacher", default="tuned", help="default, tuned, or params:<absolute path to a tuned parameters file>")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=600, help="policy steps per environment each round")
    parser.add_argument("--processes", type=int, default=4, help="Node processes per opponent profile")
    parser.add_argument("--envs-per-process", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4, help="supervised epochs per round")
    parser.add_argument("--initial-epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--rare-boost", type=float, default=40.0)
    parser.add_argument("--act-weight", type=float, default=5.0)
    parser.add_argument("--beta0", type=float, default=0.5, help="share of teacher actions in the first round, halved each round")
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--init", default=None)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    obs, mask, label, meta = load_demos(args.demos)
    hidden = torch.load(args.init)["policy"]["body.0.weight"].shape[0] if args.init else args.hidden
    policy = Policy(meta["obsSize"], meta["actCount"], hidden=hidden)
    if args.init:
        policy.load_state_dict(torch.load(args.init)["policy"])
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    run_dir = os.path.join(ROOT, "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    envs = [SoccerVecEnv(args.processes, args.envs_per_process, seed=7_000_000 + 1000 * i, opponent=opponent, teacher=args.teacher)
            for i, opponent in enumerate(("default", "tuned"))]
    evals = {name: SoccerVecEnv(1, 4, seed=5_000_000, opponent=name) for name in ("default", "tuned")}
    print(f"start: {obs.shape[0]} demonstration rows", flush=True)
    started = time.time()
    try:
        fit(policy, optimizer, obs, mask, label, args.initial_epochs, args.batch, args.rare_boost, args.act_weight)
        for round_ in range(1, args.rounds + 1):
            beta = args.beta0 * 0.5 ** (round_ - 1) if round_ <= 4 else 0.0
            new_obs, new_mask, new_label, mixed = rollout(envs, policy, args.rollout_steps, beta, rng)
            obs, mask, label = torch.cat([obs, new_obs]), torch.cat([mask, new_mask]), torch.cat([label, new_label])
            loss = fit(policy, optimizer, obs, mask, label, args.epochs, args.batch, args.rare_boost, args.act_weight)
            record = {"round": round_, "beta": beta, "teacher_share": mixed, "rows": int(obs.shape[0]), "loss": loss,
                      "seconds": time.time() - started}
            for name, env in evals.items():
                record[f"eval_{name}"] = evaluate_policy(policy, env, torch.device("cpu"), args.eval_episodes)
            torch.save({"policy": policy.state_dict(), "args": vars(args), "iteration": round_}, os.path.join(run_dir, "latest.pt"))
            log.write(json.dumps(record) + "\n"); log.flush()
            print(f"round {round_:2d}  beta {beta:.3f}  rows {record['rows']:7d}  loss {loss:.3f}"
                  f"  | vs default: {record['eval_default']['goal_diff']:+.2f} (wins {record['eval_default']['win_rate']:.2f})"
                  f"  | vs tuned: {record['eval_tuned']['goal_diff']:+.2f}  [{record['seconds']:.0f}s]", flush=True)
    finally:
        for env in envs + list(evals.values()):
            env.close()


if __name__ == "__main__":
    main()
