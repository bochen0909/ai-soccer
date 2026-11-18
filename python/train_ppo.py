"""Shared-weight PPO for the soccer environment.

One network controls every player on the learner team. Each player is a separate sample whose observation
already contains the whole match; all players share the team's reward. Illegal acts are masked out.

Example:
  python3 python/train_ppo.py --name first --iterations 200
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from policy_io import write_policy_json
from soccer_env import SoccerVecEnv

NEG = -1e9


class Policy(nn.Module):
    def __init__(self, obs_size, act_count, move_count=9, hidden=256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.move = nn.Linear(hidden, move_count)
        self.sprint = nn.Linear(hidden, 2)
        self.act = nn.Linear(hidden, act_count)
        self.value = nn.Linear(hidden, 1)
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 2 ** 0.5)
                nn.init.zeros_(layer.bias)
        for head, gain in ((self.move, 0.01), (self.sprint, 0.01), (self.act, 0.01), (self.value, 1.0)):
            nn.init.orthogonal_(head.weight, gain)

    def forward(self, obs, mask, keeper=None):
        h = self.body(obs)
        act_logits = self.act(h).masked_fill(~mask, NEG)
        return Categorical(logits=self.move(h)), Categorical(logits=self.sprint(h)), Categorical(logits=act_logits), self.value(h).squeeze(-1)

    def evaluate(self, obs, mask, actions=None, deterministic=False, keeper=None):
        move, sprint, act, value = self(obs, mask, keeper)
        if actions is None:
            actions = torch.stack([d.probs.argmax(-1) if deterministic else d.sample() for d in (move, sprint, act)], -1)
        logp = move.log_prob(actions[..., 0]) + sprint.log_prob(actions[..., 1]) + act.log_prob(actions[..., 2])
        entropy = move.entropy() + sprint.entropy() + act.entropy()
        return actions, logp, entropy, value


class TeamPolicy(Policy):
    """A plain Policy for the outfield players (its weights keep their usual names) and a separate network `keeper_net` for the keeper.

    `keeper` is a bool tensor, one entry per sample, true for the goalkeeper's samples. Both networks see every sample, and the
    choice is made per sample, so gradients reach only the network whose output was used.
    """
    def __init__(self, obs_size, act_count, move_count=9, hidden=256):
        super().__init__(obs_size, act_count, move_count, hidden)
        self.keeper_net = Policy(obs_size, act_count, move_count, hidden)

    def forward(self, obs, mask, keeper=None):
        outfield = Policy.forward(self, obs, mask)
        if keeper is None:
            return outfield
        goalkeeper = self.keeper_net(obs, mask)
        pick = keeper.unsqueeze(-1)
        heads = [Categorical(logits=torch.where(pick, g.logits, o.logits)) for o, g in zip(outfield[:3], goalkeeper[:3])]
        return (*heads, torch.where(keeper, goalkeeper[3], outfield[3]))


def is_team_state(state):
    return any(key.startswith("keeper_net.") for key in state)


def load_policy(state, team=None):
    """Builds a policy for a state dict. A plain (one network) state gives a Policy, or with team=True a TeamPolicy whose keeper
    starts as a copy of that network. A two network state always gives a TeamPolicy."""
    team = is_team_state(state) or bool(team)
    obs_size, hidden = state["body.0.weight"].shape[1], state["body.0.weight"].shape[0]
    act_count, move_count = state["act.weight"].shape[0], state["move.weight"].shape[0]
    policy = (TeamPolicy if team else Policy)(obs_size, act_count, move_count, hidden)
    if team and not is_team_state(state):
        policy.load_state_dict(state, strict=False)
        policy.keeper_net.load_state_dict(state)
    else:
        policy.load_state_dict(state)
    return policy


def to_tensors(batch, device):
    shape = batch.obs.shape
    obs = torch.as_tensor(batch.obs.reshape(-1, shape[-1]), device=device)
    mask = torch.as_tensor(batch.mask.reshape(-1, batch.mask.shape[-1]), device=device)
    return obs, mask


def act(policy, batch, device, deterministic=False):
    obs, mask = to_tensors(batch, device)
    e, a = batch.obs.shape[:2]
    # The keeper is roster slot 0.
    keeper = (torch.arange(e * a) % a == 0).to(device) if isinstance(policy, TeamPolicy) else None
    with torch.no_grad():
        actions, logp, _, value = policy.evaluate(obs, mask, deterministic=deterministic, keeper=keeper)
    return actions.reshape(e, a, 3), logp.reshape(e, a), value.reshape(e, a)


def evaluate_policy(policy, env, device, episodes_per_env):
    """Plays whole matches with the stochastic policy and returns mean goal difference and win rate."""
    batch = env.reset(seed=987_654)
    counts = np.zeros(env.num_envs, dtype=int)
    diffs = []
    while counts.min() < episodes_per_env:
        actions, _, _ = act(policy, batch, device)
        batch = env.step(actions.cpu().numpy())
        for i in np.flatnonzero(batch.done & (batch.steps >= 0)):
            if counts[i] < episodes_per_env:
                diffs.append(int(batch.score[i, 0] - batch.score[i, 1]))
            counts[i] += 1
    diffs = np.array(diffs)
    return {"goal_diff": float(diffs.mean()), "win_rate": float((diffs > 0).mean()),
            "draw_rate": float((diffs == 0).mean()), "episodes": len(diffs)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="run")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--envs-per-process", type=int, default=2)
    parser.add_argument("--rollout", type=int, default=128, help="policy steps per environment per iteration")
    parser.add_argument("--size", type=int, default=5, choices=[5, 11])
    parser.add_argument("--opponent", default="default", choices=["default", "tuned"])
    parser.add_argument("--period-ticks", type=int, default=None, help="shorter halves, for faster episodes")
    parser.add_argument("--drill", default=None, choices=["shoot", "attack", "pass", "three", "counter", "defend", "defend2", "giveandgo", "rondo", "keeper", "cross", "lob", "lane"], help="train on a short drill, not full matches")
    parser.add_argument("--drill-mix", type=float, default=0.0, help="league: share of episodes that start as a short drill")
    parser.add_argument("--replay-file", default=None, help="situations from record-failures.mjs, used by the drill kind \"replay\"")
    parser.add_argument("--drill-kinds", nargs="+", default=["three", "counter", "pass", "attack"],
                        choices=["shoot", "attack", "pass", "three", "counter", "defend", "defend2", "giveandgo", "rondo", "keeper", "cross", "lob", "lane", "replay"])
    parser.add_argument("--max-steps", type=int, default=None, help="policy steps before an episode is cut off (drills: 60)")
    parser.add_argument("--progress", type=float, default=None, help="ball progress reward weight (default 0.1)")
    parser.add_argument("--break-bonus", type=float, default=0.0, help="reward per opponent a completed pass takes out of the game (at most two)")
    parser.add_argument("--save-bonus", type=float, default=0.0, help="reward when our keeper catches or parries the ball")
    parser.add_argument("--pass-fail-penalty", type=float, default=0.0, help="penalty (give a positive number) when an opponent wins a pass before a teammate controls it")
    parser.add_argument("--out-penalty", type=float, default=0.0, help="penalty (positive) when we put the ball out of play")
    parser.add_argument("--shot-bonus", type=float, default=0.0, help="reward for a shot on target within 20 m, larger the closer")
    parser.add_argument("--own-third-penalty", type=float, default=0.0, help="penalty (positive) when an opponent takes the ball within 20 m of our goal line")
    parser.add_argument("--turnover-bonus", type=float, default=0.0, help="reward for winning the ball from the opponent, penalty for losing it")
    parser.add_argument("--assist-bonus", type=float, default=0.0, help="extra reward on a goal scored within 6 s of a completed pass")
    parser.add_argument("--pass-bonus", type=float, default=0.0, help="reward when a teammate receives a pass (drills and full matches)")
    parser.add_argument("--pure-goal", action="store_true",
                        help="drills: the only reward is the goal (no progress, possession, pass or other terms)")
    parser.add_argument("--end-on-ball-loss", action="store_true",
                        help="drills: the episode also ends when anyone but the starting ball carrier touches the ball")
    parser.add_argument("--striker-only", action="store_true",
                        help="drills: update the policy only from the last player on the team (the drill's ball carrier); "
                             "the other players still act with the same policy but their samples are ignored")
    parser.add_argument("--pure-drills", nargs="+", default=None, choices=["shoot", "attack", "pass", "keeper", "rondo", "defend", "defend2", "replay"],
                        help="episodes of these drill kinds (alone with --drill, or with --drill-mix) are pure: the only rewards are the goal "
                             "and, for attack and pass, a pass bonus paid once; they end when a rival touches the ball; the policy learns only from the "
                             "carrier (shoot) or the carrier and the runner (attack, pass). Other episodes are unchanged")
    parser.add_argument("--solo-goals", action="store_true", help="pure drills other than shoot: a goal also pays without a completed pass")
    parser.add_argument("--pure-rewards", default=None,
                        help='JSON overriding the per-drill rewards of pure drills, e.g. \'{"attack": {"passCompleted": 0.5, "goalScored": 0.5}}\'. '
                             'Defaults: shoot goal 1, pass first completed pass 1, attack pass 0.5 and goal 0.5 (the goal only after the pass)')
    parser.add_argument("--keeper-net", action="store_true",
                        help="give the goalkeeper its own network (a copy of the --init policy at the start). A checkpoint that already "
                             "has one keeps it")
    parser.add_argument("--value-warmup", type=int, default=0,
                        help="iterations that train only the value head, so a pretrained policy is not wrecked by a random critic")
    parser.add_argument("--league-anchor", nargs="+", default=None,
                        help="one or more policy JSON files to keep as fixed league opponents, e.g. one per generation; "
                             "turns on league training against the CPUs, these anchors, and snapshots of the policy itself")
    parser.add_argument("--league-params", nargs="+", default=None,
                        help="tuned parameter files (from `tune search`) to add as fixed baseline opponents; turns on league training")
    parser.add_argument("--exploit", action="store_true",
                        help="league: train only against the anchors (and snapshots), not the CPUs, to find the anchors' weaknesses; "
                             "the CPUs are still used for evaluation")
    parser.add_argument("--tuned-boost", type=float, default=1.0,
                        help="league: multiplies how often the tuned (Hard) CPU is drawn, on top of the win-rate weighting")
    parser.add_argument("--snapshot-every", type=int, default=25, help="league: iterations between snapshots of the policy (0: no snapshots, the pool is only the CPUs and the anchors)")
    parser.add_argument("--snapshots-kept", type=int, default=6, help="league: how many recent snapshots stay in the pool")
    parser.add_argument("--pfsp-floor", type=float, default=0.05,
                        help="league: smallest draw weight an opponent can have; lower it so beaten opponents are drawn rarely, not 5%% as often as an even one")
    parser.add_argument("--possession", type=float, default=None, help="possession reward per step in matches and drills (default 0.002)")
    parser.add_argument("--pfsp-power", type=float, default=2.0,
                        help="league: opponents the policy beats less often are drawn more often, by (1.1 - win rate)^power")
    parser.add_argument("--init", default=None, help="checkpoint to start from, for example a drill-trained policy")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=1, help="matches per eval environment")
    parser.add_argument("--eval-envs", type=int, default=16,
                        help="environments per evaluation opponent, spread over 8 processes; matches per evaluation = envs x episodes")
    parser.add_argument("--hidden", type=int, default=256, help="units per hidden layer for a new policy (--init keeps its own)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    options = {"size": args.size, "opponent": args.opponent}
    if args.period_ticks:
        options["periodTicks"] = args.period_ticks
    if args.drill:
        options["drill"] = args.drill
        options["maxSteps"] = args.max_steps or 60
        options["reward"] = {"possession": 0.0, "passCompleted": args.pass_bonus}
        if args.pure_goal:
            options["reward"] = {"goalScored": 1.0, "goalConceded": 0.0, "ballProgress": 0.0, "possession": 0.0, "passCompleted": 0.0}
        if args.end_on_ball_loss:
            options["endOnBallLoss"] = True
    elif args.max_steps:
        options["maxSteps"] = args.max_steps
    if args.pure_drills:
        if args.drill and args.drill not in args.pure_drills or not args.drill and args.drill_mix <= 0:
            raise SystemExit("--pure-drills needs --drill-mix, or --drill with a kind listed in --pure-drills")
        options["pureDrills"] = args.pure_drills
        if args.pure_rewards:
            options["pureRewards"] = json.loads(args.pure_rewards)
        if args.solo_goals:
            options["goalNeedsPass"] = False
    if (args.pure_goal or args.end_on_ball_loss or args.striker_only) and not args.drill:
        raise SystemExit("--pure-goal, --end-on-ball-loss and --striker-only need --drill")
    if args.drill_mix > 0 and not args.drill:
        options["drillMix"] = {"p": args.drill_mix, "kinds": args.drill_kinds, "maxSteps": 60}
        if args.replay_file:
            options["replayFile"] = os.path.abspath(args.replay_file)
    if (args.pass_bonus or args.assist_bonus or args.turnover_bonus or args.break_bonus or args.save_bonus or args.pass_fail_penalty or args.out_penalty or args.shot_bonus or args.own_third_penalty) and not args.drill:
        options["reward"] = {"passCompleted": args.pass_bonus, "assist": args.assist_bonus, "turnover": args.turnover_bonus, "passBreak": args.break_bonus,
                             "save": args.save_bonus, "passFailed": -args.pass_fail_penalty, "outOfBounds": -args.out_penalty,
                             "shotOnTarget": args.shot_bonus, "ownThirdGiveaway": -args.own_third_penalty}
    if args.progress is not None:
        options["reward"] = {**options.get("reward", {}), "ballProgress": args.progress}
    if args.possession is not None:
        options["reward"] = {**options.get("reward", {}), "possession": args.possession}
    run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", args.name)
    os.makedirs(run_dir, exist_ok=True)

    # The league: fixed opponents plus snapshots of the policy. `rate` is a running estimate of how often the
    # policy beats each one (1 for a win, 0.5 for a draw), which sets how often it is drawn next.
    league = {}
    next_snapshot_id = 10

    def pool_message():
        weights = {key: max(args.pfsp_floor, (1.1 - entry["rate"]) ** args.pfsp_power) * (args.tuned_boost if key == 1 else 1.0)
                   for key, entry in league.items()}
        return [{"id": key, "spec": entry["spec"], "weight": weights[key]} for key, entry in league.items()]

    eval_specs = {"default": "default", "tuned": "tuned"}
    if args.league_anchor or args.league_params:
        anchors = [("anchor_" + os.path.splitext(os.path.basename(p))[0], "model:" + os.path.abspath(p)) for p in args.league_anchor or []]
        anchors += [("params_" + os.path.splitext(os.path.basename(p))[0], "params:" + os.path.abspath(p)) for p in args.league_params or []]
        next_snapshot_id = max(next_snapshot_id, 2 + len(anchors))   # pool IDs 2.. are anchors, snapshots come after them
        entries = [(0, "default", "default"), (1, "tuned", "tuned")]
        for index, (name, spec) in enumerate(anchors):
            entries.append((2 + index, name, spec))
        for key, name, spec in entries:
            league[key] = {"name": name, "spec": spec, "rate": 0.5, "snapshot": None}
            if key >= 2:
                eval_specs[name] = spec
        if args.exploit:
            del league[0], league[1]
        options["pool"] = pool_message()
    env = SoccerVecEnv(args.processes, args.envs_per_process, seed=args.seed * 100, **options)
    eval_shape = (8, max(1, args.eval_envs // 8))
    def evaluate_against(spec):
        # Built per evaluation and closed again, so a long league (many anchors) does not keep a set of idle processes per opponent.
        evaluation_env = SoccerVecEnv(*eval_shape, seed=5_000_000, **{**options, "opponent": spec, "pool": None, "drillMix": None,
                                       "pureDrills": None, "endOnBallLoss": None, "goalNeedsPass": None})
        try:
            return evaluate_policy(policy, evaluation_env, device, args.eval_episodes)
        finally:
            evaluation_env.close()
    E, A, D, C = env.num_envs, env.agents, env.obs_size, env.act_count
    T = args.rollout
    init_state = torch.load(args.init)["policy"] if args.init else None
    if init_state is not None:
        policy = load_policy(init_state, team=args.keeper_net)
    else:
        policy = (TeamPolicy if args.keeper_net else Policy)(D, C, hidden=args.hidden)
    team = isinstance(policy, TeamPolicy)
    if team:
        print("the goalkeeper has its own network")
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)
    log = open(os.path.join(run_dir, "log.jsonl"), "a")
    print(f"{E} envs x {A} agents, obs {D}, acts {C}, {T} steps per rollout -> {E * T * A} samples per iteration")

    batch = env.reset()
    total_steps = 0
    started = time.time()
    finished_diffs = []
    try:
        for iteration in range(1, args.iterations + 1):
            for group in optimizer.param_groups:
                group["lr"] = args.lr * (1 - (iteration - 1) / args.iterations)
            store = {k: [] for k in ("obs", "mask", "act", "logp", "val", "rew", "done", "agent", "pure", "learn")}
            for _ in range(T):
                actions, logp, value = act(policy, batch, device)
                store["obs"].append(batch.obs); store["mask"].append(batch.mask)
                store["act"].append(actions.numpy()); store["logp"].append(logp.numpy())
                store["val"].append(value.numpy()); store["agent"].append(batch.agent_mask)
                store["pure"].append(batch.pure)
                store["learn"].append(batch.learn)
                batch = env.step(actions.numpy())
                store["rew"].append(batch.reward); store["done"].append(batch.done)
                ok = batch.done & (batch.steps >= 0)
                finished_diffs.extend((batch.score[ok, 0] - batch.score[ok, 1]).tolist())
                for i in np.flatnonzero(ok):
                    entry = league.get(int(batch.opponent[i]))
                    if entry is not None:   # an opponent that left the pool can still finish its running episodes
                        diff = batch.score[i, 0] - batch.score[i, 1]
                        entry["rate"] += 0.05 * ((1.0 if diff > 0 else 0.5 if diff == 0 else 0.0) - entry["rate"])
            total_steps += T * E

            with torch.no_grad():
                _, _, last_value = act(policy, batch, device)
            rew = np.stack(store["rew"])[..., None]            # [T, E, 1]
            done = np.stack(store["done"])[..., None].astype(np.float32)
            val = np.stack(store["val"])                       # [T, E, A]
            adv = np.zeros_like(val)
            last_gae = np.zeros((E, A), dtype=np.float32)
            next_value = last_value.numpy()
            for t in reversed(range(T)):
                non_terminal = 1.0 - done[t]
                delta = rew[t] + args.gamma * next_value * non_terminal - val[t]
                last_gae = delta + args.gamma * args.lam * non_terminal * last_gae
                adv[t] = last_gae
                next_value = val[t]
            returns = adv + val

            agents_used = np.stack(store["agent"])             # [T, E, A]
            if args.striker_only:
                only = np.zeros(A, dtype=bool); only[A - 1] = True
                agents_used = agents_used & only
            if args.pure_drills:
                # In a pure drill only the carrier and the runner learn; the env reports them as a bit mask of roster slots.
                learn = np.stack(store["learn"])[..., None]    # [T, E, 1]
                bits = ((learn >> np.arange(A)) & 1).astype(bool)
                agents_used = agents_used & ((learn == 0) | bits)
            agent = agents_used.reshape(-1)
            flat = lambda x: torch.as_tensor(np.stack(x) if isinstance(x, list) else x).reshape(-1, *np.shape(x[0])[2:])[agent]
            b_obs, b_mask, b_act = flat(store["obs"]), flat(store["mask"]), flat(store["act"])
            b_logp = torch.as_tensor(np.stack(store["logp"]).reshape(-1)[agent])
            b_adv = torch.as_tensor(adv.reshape(-1)[agent])
            b_ret = torch.as_tensor(returns.reshape(-1)[agent])
            b_keeper = torch.as_tensor(np.broadcast_to(np.arange(A) == 0, (T, E, A)).reshape(-1)[agent])
            if team:
                # The two networks learn from different samples, so each group's advantages are normalised on their own.
                for group in (b_keeper, ~b_keeper):
                    if group.sum() > 1:
                        b_adv[group] = (b_adv[group] - b_adv[group].mean()) / (b_adv[group].std() + 1e-8)
            else:
                b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

            n = b_obs.shape[0]
            size = n // args.minibatches
            stats = {"kl": 0.0, "entropy": 0.0, "value_loss": 0.0, "clip_frac": 0.0}
            updates = 0
            for _ in range(args.epochs):
                order = torch.randperm(n)
                stop = False
                for start in range(0, n - size + 1, size):
                    idx = order[start:start + size]
                    _, logp, entropy, value = policy.evaluate(b_obs[idx], b_mask[idx], b_act[idx], keeper=b_keeper[idx] if team else None)
                    ratio = (logp - b_logp[idx]).exp()
                    unclipped = ratio * b_adv[idx]
                    clipped = ratio.clamp(1 - args.clip, 1 + args.clip) * b_adv[idx]
                    policy_loss = -torch.min(unclipped, clipped).mean()
                    value_loss = 0.5 * (value - b_ret[idx]).pow(2).mean()
                    if iteration <= args.value_warmup:
                        loss = 0.5 * value_loss
                    else:
                        loss = policy_loss + 0.5 * value_loss - args.entropy * entropy.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    optimizer.step()
                    with torch.no_grad():
                        kl = ((ratio - 1) - (ratio.log())).mean().item()
                    stats["kl"] += kl; stats["entropy"] += entropy.mean().item()
                    stats["value_loss"] += value_loss.item()
                    stats["clip_frac"] += ((ratio - 1).abs() > args.clip).float().mean().item()
                    updates += 1
                    if kl > args.target_kl * 1.5:
                        stop = True
                        break
                if stop:
                    break
            stats = {k: v / max(1, updates) for k, v in stats.items()}

            record = {"iteration": iteration, "steps": total_steps, "sps": total_steps / (time.time() - started),
                      "episodes": len(finished_diffs),
                      "train_goal_diff": float(np.mean(finished_diffs)) if finished_diffs else None, **stats}
            finished_diffs = []
            if iteration % args.eval_every == 0 or iteration == args.iterations:
                for name, spec in eval_specs.items():
                    record[f"eval_{name}"] = evaluate_against(spec)
                torch.save({"policy": policy.state_dict(), "args": vars(args), "iteration": iteration},
                           os.path.join(run_dir, "latest.pt"))
                torch.save({"policy": policy.state_dict(), "args": vars(args), "iteration": iteration},
                           os.path.join(run_dir, f"iter_{iteration:04d}.pt"))
            if league:
                if args.snapshot_every > 0 and iteration % args.snapshot_every == 0:
                    path = os.path.join(run_dir, "league", f"snapshot_{iteration:04d}.json")
                    write_policy_json(policy.state_dict(), path, args.name, iteration)
                    # A new snapshot is as strong as the policy itself, so it starts at an even rate.
                    league[next_snapshot_id] = {"name": f"snapshot_{iteration}", "spec": "model:" + path, "rate": 0.5, "snapshot": iteration}
                    next_snapshot_id += 1
                    for key in sorted(k for k, v in league.items() if v["snapshot"] is not None)[:-args.snapshots_kept]:
                        del league[key]
                env.set_pool(pool_message())
                record["league"] = {entry["name"]: round(entry["rate"], 3) for entry in league.values()}
            log.write(json.dumps(record) + "\n"); log.flush()
            summary = (f"it {iteration:4d} steps {total_steps:8d} sps {record['sps']:6.0f} "
                       f"train diff {record['train_goal_diff'] if record['train_goal_diff'] is None else round(record['train_goal_diff'], 2)} "
                       f"ent {stats['entropy']:.2f} kl {stats['kl']:.4f}")
            for name in eval_specs:
                if f"eval_{name}" in record:
                    e = record[f"eval_{name}"]
                    summary += f" | vs {name}: diff {e['goal_diff']:+.2f} win {e['win_rate']:.2f}"
            if league and "eval_default" in record:
                summary += "\n          league win rates: " + "  ".join(f"{entry['name']} {entry['rate']:.2f}" for entry in league.values())
            print(summary, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
