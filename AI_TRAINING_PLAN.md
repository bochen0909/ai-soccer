# AI training plan

Working notes on how we get from the current baseline AI to a trained one. Update this file as steps finish or the plan changes. The interface itself is documented in [TRAINING.md](TRAINING.md); the engine design is in [DESIGN.md](DESIGN.md).

## Where we are

- Engine, headless runner, replay and `evaluate` (match, verify, benchmark) exist.
- The baseline AI (`BaselineTeamAI`) is the reference opponent.
- The training adapter is built: `SoccerTrainingEnv` in `src/training/` (observation encoder, action decoder and masks, reward, reset and step). Its tests are in `test/training.test.mjs`.
- A trained 5v5 policy ships in `models/trained-5v5.json`. It beats Hard and the default CPU on held-out seeds (see "Role features broke the plateau").

## Guiding principle

Build the cheapest thing that gives a trustworthy measurement first. Reinforcement learning (RL) is expensive and hard to debug, so we earn the right to use it: first a fitness number we trust, then a strong opponent, then a verified environment. PPO is not the next step.

## Approach, in order

### 1. Tuning harness (next)

Search over the baseline's traits (aggression, risk tolerance, awareness) and its scoring weights.

- **Method:** random search first, then CMA-ES or a genetic algorithm if it pays off.
- **Fitness:** goal difference over N seeded matches against the current baseline, using the `evaluate` machinery.
- **Why first:** no ML stack, uses the existing plugin interface, and produces a stronger opponent for later stages.
- **Open question:** how many matches per candidate before the fitness is reliable. Measure the variance across seeds before choosing N.
- **Limit:** cannot discover strategies the scoring rules do not express.
- **Exit check:** if tuning cannot beat the baseline reliably, the evaluation is too noisy and RL would fail for the same reason. Fix that first.

### 2. Sanity-check the environment

Before any RL run, confirm the adapter gives a learnable signal.

- Scripted policies through `SoccerTrainingEnv` (always chase the ball, always shoot when legal, random legal actions) should produce different, sensible rewards and scores.
- Check masks never leave an agent with no legal act, and illegal-action counts stay near zero for scripted policies.
- Check that observation values stay bounded over full matches (the test only covers short ones).
- Re-run the throughput benchmark after any change. Current numbers: about 750 steps per second at 5v5 and 350 at 11v11 on one core with random actions.

### 3. Curriculum drills, then full matches with PPO

Use the scenario reset contract to build drills, then move to full matches.

1. Shoot against the goalkeeper from varied spots.
2. Pass to an open teammate.
3. 2v1 and 3v2 attacks.
4. Defending: tackle, mark, recover.
5. Full 5v5, then 11v11.

Train against the tuned baseline from step 1.

- **Needs:** a process bridge (stdio or socket) and a Gymnasium or PettingZoo wrapper in Python, plus a persistent Node worker pool for throughput.
- **Trainer:** PPO via Stable-Baselines3, CleanRL or RLlib. Mask the logits with `actionMask`.
- **Policy:** one shared-weight network applied to every agent.
- **Reward:** goals dominate. Anneal `shapingScale` toward 0 so the policy cannot exploit the shaping terms.

### 4. Only if PPO stalls

- **Imitation learning:** record the baseline and human play (the replay system already stores actions), behavior-clone a policy, then fine-tune with RL. This avoids the early phase where random agents never score.
- **Self-play with a league:** train against saved past versions, the baseline and tuned variants, not only the latest policy. Track Elo from `evaluate` results. Needs the opponent to be a second `external` team.
- **Domain randomization:** vary player abilities across episodes so the policy generalizes across rosters. Abilities are already observation features.

## Todo list

Tuning and evaluation
- [x] Measure fitness variance across seeds (5v5 baseline vs itself: std dev 1.55 goals per match, home advantage +0.23; about 240 matches resolve a 0.2 goal edge). Search default of 24 matches per candidate only separates large differences, so confirm results with `tune compare` on 300 or more held-out matches
- [ ] Re-measure variance for 11v11
- [x] Tuning harness: `npm run tune -- variance|search|compare` (evolution strategy over 3 traits and 7 weight multipliers, worker pool, alternating sides, held-out comparison)
- [x] First real search and held-out comparison (5v5, 20 generations x 16 candidates x 24 matches, seed 1): the tuned team beat the default baseline by +6.24 goals per match (95% interval +/-0.25) over 300 held-out matches, 297 wins, 3 draws, 0 losses, 6.98 goals for and 0.74 against. It holds 83% of the possession, presses hard (aggression 0.96) and shoots from about 6.5 m. This shows it exploits the default baseline's weak defense, not that it plays well against a strong opponent
- [x] Save the best configuration as a named opponent: `AI_PROFILES.tuned` in `src/presets/ai.ts`, selectable in the browser app (5-a-side only)
- [ ] Tuned vs tuned sanity check (expect about 0 plus the home advantage)
- [ ] Iterated search: use the tuned team as the opponent and repeat, building a league of progressively harder opponents
- [ ] Measure the default keeper's save rate on close shots (6 to 7 m); add keeper constants (line depth, save timing, coverage) to the search space
- [ ] Tune 11v11 separately; the current tuned profile is disabled for 11-a-side in the app
- [ ] Elo or win-rate report in `evaluate` for comparing controllers

Environment
- [x] Scripted-policy sanity checks: `npm run check-env` (idle, random and chase policies at 5v5 and 11v11; return equals score difference with shaping off, masks always allow "none", observations bounded)
- [x] Drill scenarios: `shoot` and `attack` (2v1) in `src/training/drills.ts`
- [x] Pass drill and pass-completion reward (`passCompleted`, off by default, 0.3 in drills): the policy learned a passing play
- [ ] Next rungs: 2v1 attack from the pass policy, then 3v2, then full matches initialised from the best drill policy
- [ ] Fast-forward dead phases (goal pause, halftime, restart placement) so steps are not wasted on them
- [ ] Opponent can be a second `external` team (self-play)
- [ ] Per-agent rewards if the shared team reward turns out to be too coarse for credit assignment

Training infrastructure
- [x] stdio bridge (`scripts/env-server.mjs`) and Python client (`python/soccer_env.py`); no Gymnasium dependency, a vectorized client with masks and auto-reset
- [x] Persistent worker pool: several Node processes, several environments each, batched stepping (about 2,500 to 3,000 steps per second on 12 cores)
- [ ] Throughput target and profiling (the learner decides every tick, and cloning observations per tick is the main overhead)
- [x] Policy export and in-engine inference: `python/export_policy.py` writes exact float32 weights to `models/trained-5v5.json`; `NeuralTeamAI` (team AI "neural") runs them in the engine; a test pins its outputs to PyTorch's. In-game, 64 held-out matches: +1.20 ±0.67 against the default CPU and -0.11 ±0.69 against Hard. The browser app offers it as "CPU Trained" (5-a-side)
- [x] League training: `train_ppo.py --league-anchor` trains against a pool of the default CPU, Hard, an anchor model and snapshots of itself, weighted toward the opponents it beats least (`SET_POOL` in the env server). Results in the table below (league1)
- [x] Several anchors, one per generation: `--league-anchor a.json b.json ...` (up to 8), exported generations live in `models/anchors/` (gen1-ft1, gen2-league1). A new generation should be promoted only after a 64-match win over the previous one. Not yet used in a full run
- [ ] Re-run imitation with more teacher diversity (default and tuned teachers, 11v11) and a bigger network if the policy plateaus

## RL results so far

Everything below used the PPO trainer (`python/train_ppo.py`) with one shared network over all five players, masked action heads, 16 parallel matches and about 2,500 to 3,000 steps per second.

| Run | Setup | Result |
| --- | --- | --- |
| run2 | Full 5v5 matches against the default CPU, shaped reward, 200 iterations (about 400,000 steps) | Flat. Evaluation goal difference stayed between -3 and -6 per match against the default CPU and between -10 and -14 against Hard (the starting policy scores about -5 and -12). Entropy fell only slightly, from 2.89 to 2.66 |
| shoot1 | Shooting drill: striker with the ball 8 to 16 m out against the keeper, 6 s episodes, 200 iterations | Learned within about 20 iterations, then plateaued at 50 to 70% scoring. For comparison: idle 0%, random 12%, scripted "shoot centre" 5 to 35%, scripted "shoot at a corner from 12 to 16 m" about 58 to 60%. The policy has matched the best simple rule, mostly by aiming at corners |
| attack1 | 2v1 attack drill, started from the shooting policy, 250 iterations | 0 to 4% scoring throughout. A scripted run-and-shoot policy scores 0 to 5%, so the drill can only be solved by passing or dribbling past the defender, and the reward arrives only after a whole pass, run and shot line up |

| pass1 | Pass drill (defender right in front of the carrier, open runner near goal), pass bonus 0.3, started from shoot1, 250 iterations | Learned a passing play: scoring rose from about 3% to 25 to 33% in evaluation (20 to 30% in training). A scripted pass-then-shoot policy scores about 2.5% there, and a solo run-and-shoot policy 1% |
| attack2 | 2v1 attack drill from pass1 (pass bonus 0.3), 250 iterations | Learned: 15 to 30% scoring in evaluation (the same drill from the shooting policy alone stayed at 0 to 4%) |
| full1 | Full 5v5 matches against the default CPU, started from attack2, 300 iterations | No gain. Evaluation goal difference -4 to -10 against the default CPU (mostly -6 to -7) and -11 to -14 against Hard, about the same as from scratch. The attacking skills from the drills did not carry over, probably because drills start in open play with the ball and never practise kickoffs, restarts or defending |
| imitation1/2 | Behavior cloning of the tuned CPU (40 matches, 310,000 rows, 10 to 12 epochs) | The clone matched the teacher's choices on held-out matches (move 92%, shoot 94%, tackle 99%, save 100% once rare acts were oversampled 40x) but played badly: -6 against the default CPU, -10 against Hard. Passing the teacher's own decisions through the same action interface (8 move directions, pass-to-slot) scores +3.1 against the default CPU, so the interface was not the limit. The gap was the usual cloning drift into states the teacher never visited |
| dagger1 | DAgger: the student plays, the teacher labels each visited state, retrain on the growing dataset; 8 rounds | Steady gain: against the default CPU -7.0, -5.9, -4.8, -4.0, -3.5, -3.9, -1.4 (12% wins), -2.4; against Hard -13.8 down to -6.9. Best full-match result so far, still below the teacher's +3 |
| dagger2 | Longer DAgger: 16 rounds, 900-step rollouts, 1.46 million rows | Against the default CPU the 8-match readings went -6.3 up to +0.1 to +0.6 from round 9 on. Held-out (64 matches each): **+0.47 ±0.50 against the default CPU** (48% wins, 22% draws, 30% losses) and -4.88 ±0.75 against Hard (3% wins) |
| ft1 | PPO fine-tuning from dagger2 with the goal reward: lr 1e-4, clip 0.1, 10 value-only warm-up iterations, 300 iterations, 16 matches per evaluation | Held-out (64 matches each): **+0.70 ±0.62 against the default CPU** (48% wins, 17% draws, 34% losses) and **+0.45 ±0.67 against Hard** (44% / 12% / 44%). Hard improved from -4.9 to about even. Mid-run readings of +2 were partly luck, since 16-match evaluations swing by 2 to 3 goals; only the 64-match held-out numbers should be quoted |
| league1 | League PPO from ft1 under the random-kickoff rule: pool of default, Hard, ft1 as anchor and 6 snapshots, 600 iterations, lr 1e-4, clip 0.1 | Held-out (64 matches each, Python): **+1.36 ±0.60 against the default CPU**, **+0.09 ±0.66 against Hard**, **+1.69 ±0.50 against ft1** (78% wins). In the engine over 256 matches (half home, half away): +1.66 ±0.29 against the default CPU and -0.17 ±0.30 against Hard (the 64-match engine run had read -0.63). Under the random-kickoff rule ft1 scored +1.41 ±0.64 and -0.86 ±0.60. The 16-match evaluations during training swung from -2.9 to +0.2 against Hard and were not a usable signal |
| league2 | League PPO from league1: pool of default, Hard (drawn twice as often, `--tuned-boost 2`), anchors ft1 and league1, 6 snapshots, 600 iterations, about 10 minutes | Engine, 256 matches each: +1.53 ±0.34 against the default CPU and **+0.32 ±0.34 against Hard**. Python, 128 matches: **-0.53 ±0.35 against league1** (31% wins, 50% losses). Better than league1 against Hard but it loses to league1 head to head, so it was not promoted; league1 stays the exported policy. Its checkpoint is `runs/league2/latest.pt` |
| league3 | league2's recipe run for 2000 iterations from league1 (lr 1e-4, entropy 0.005, 8 snapshots) | Python, 128 matches: +0.95 ±0.27 against the default CPU, -0.68 ±0.38 against Hard, -0.12 ±0.28 against league1. Entropy rose from 0.55 to about 1.0 and the KL fell to 0, so it was not promoted. Longer training against the same pool does not help |
| big_league | 512 hidden units (`--hidden 512`): DAgger, 300 PPO iterations, 800 league iterations with ft1 and league1 as anchors | Python, 128 matches: **+1.97 ±0.38 against the default CPU** (best so far) but **-3.64 ±0.59 against Hard** and -0.12 ±0.37 against league1. It does not learn to beat Hard, so it was not promoted |

Note: the ft1 numbers in the table above were measured under the old rule where home always kicked off. Random kickoff is now always on.

What this tells us:
- The best trained player (ft1) is level with the Hard CPU and slightly ahead of the default CPU in full 5v5 matches. It was trained by cloning the tuned CPU, correcting drift with DAgger, and fine-tuning with PPO on the goal reward. Plain PPO from scratch and drill-first PPO never got past about -5
- The pipeline works end to end: environment, masks, bridge, trainer, drills and evaluation.
- Dense, short tasks are learned in minutes. Full matches and multi-step plays need more than plain PPO from scratch with a goals-only signal.
- The default keeper almost always stops centre shots and often misses corner shots. Corner placement is the main skill the shooting drill taught.

Next, in the order I'd try them:
1. **Imitation first (in progress: DAgger works).** Record the tuned CPU's decisions as (observation, action) pairs, behavior-clone a policy from them, then fine-tune with RL. This gives RL a team that already defends and holds the ball. The conversion from the baseline's world-space decisions to the policy's discrete actions is approximate (8 move directions, pass to slot k).
1b. **Hybrid control.** Let the baseline AI play the goalkeeper and restarts and let the policy control only outfield players in open play.
1c. **Mixed training.** Alternate drill episodes with full matches in one run, so attacking skills don't replace defending.
2. **Longer and larger runs** once a drill moves, with more parallel environments, and a tuned entropy coefficient.
3. **Reward shaping for full matches:** shots on target, passes received, and tackles won, annealed with `shapingScale`.
4. **Imitation start:** behavior-clone the tuned CPU's decisions (hard to beat as a teacher, since it wins 297 of 300 against the default) and then fine-tune with RL.
5. **Evolution on the network weights** as an alternative to PPO, using the same environment.

Findings from the imitation work worth keeping:
- The tuned teacher passes rarely (a few passes per match) and mostly dribbles forward, shoots close and tackles. Rare decisions (shots, saves) vanish from a cloned policy unless they are oversampled.
- A teacher's kick toward open space (kickoff, clearance) names no receiver. The policy can only pass to a teammate slot, so these map to the best-aligned legal slot (0 of 1,578 teacher requests are now lost).
- `scripts/record-demos.mjs` records a teacher's decisions in the policy's format. `python/imitate.py` clones them, `python/dagger.py` runs DAgger with the env server's `teacher` option (the server returns what a baseline profile would do in every state the student visits).

## Known issues

- **Checked, not a problem:** the trained policy looked worse as the home team in a 64-match engine run. With 256 matches each way it scores +1.48 at home and +1.84 away against the default CPU, and -0.28 / -0.06 against Hard. The gaps are inside the noise (about ±0.4 per side), and the default CPU itself has 0.33 goals of home advantage. The policy is trained only as the home team and plays away fine.
- **Fixed:** the physics engine threw `Physics exceeded its collision event limit` when several players piled up against a wall (seed 100050 at 5v5, tick 2959). The wall acts like an infinitely heavy body, so contacts between the stacked players traded zero-time impulses forever. The fallback in `src/engine/physics.ts` stops the jammed players, as an inelastic pile-up would, and only throws if the same substep overruns again. A regression test is in `test/headless.test.mjs`. Seeds 100000 to 100119 now all complete.
- The tuner still counts any remaining engine crash as a draw and reports the seeds, so one bad seed cannot kill a search.
- Players that end up overlapping (about 0.1 m in the pile above) are not pushed apart after they stop. They separate once they move away from each other. Revisit if overlaps persist visibly.

## Role features broke the plateau (gen3)

The league runs stalled because the student never imitated Hard well. Pure DAgger reached only +1 against the default CPU where Hard scores +7.7, and its loss stayed near 0.7. The cause is that the teacher is a team coordinator, not a per-player rule: it picks one chaser by comparing every teammate's interception time, assigns markers by greedy matching, and moves toward a mode-dependent formation anchor. A shared-weight MLP that sees one raw snapshot has to rediscover all of that for every player.

The fix is a trailing **role block** in each observation row (`analyzeRoles` in `src/controllers/baseline.ts`, written by `src/training/observation.ts`): attack/defend/loose mode, whether the player is the chaser, its interception time, rank and gap, the interception point, the walking destination (chase, mark or anchor), sprint hint, and for the carrier the shot target, shot clearance, range ratio, pressure, forward lane and best pass slot. Observation size is now 63 + 21n + 23 + n (196 at 5v5); older policies are still read through `layoutFromSize`, and `roleFeatures: false` gives the old layout.

Results, 128 shared-seed matches per pair, sides alternating (`npm run arena`):

| Model | vs Hard | vs default | vs previous best (league1) |
| --- | --- | --- | --- |
| league1 (old shipped) | -0.30 +/- 0.66 (64 matches) | +1.67 | |
| dagger_role (imitation only, 16 DAgger rounds) | -1.73 +/- 0.47 | +2.61 | +0.63 +/- 0.35 |
| role_league1 (dagger_role + 500 league iterations) | +3.48 +/- 0.46 (111 W, 8 D, 9 L) | +4.28 +/- 0.38 | +3.91 +/- 0.42 |
| **role_league2 (gen4, shipped: role_league1 + 600 league iterations, anchors gen2 and gen3)** | **+4.46 +/- 0.43** (121 W, 4 D, 3 L) | **+4.38 +/- 0.35** | **+0.99 +/- 0.24 against role_league1** |
| role_league3 (gen5, 600 more iterations, anchors gen2 to gen4) | +5.01 +/- 0.43 | +4.70 +/- 0.44 | +0.05 +/- 0.26 against gen4, so not promoted |
| role512_league1 (512 units, 30 DAgger rounds, 600 league iterations) | +1.12 +/- 0.44 | | -3.67 +/- 0.35 against gen4 |

role_league2 (gen4) is now `models/trained-5v5.json`; every shipped generation stays in `models/anchors/` as an opponent. Recipe: record demos, `dagger.py --name dagger_role`, then `train_ppo.py` with `--league-anchor models/trained-5v5.json --value-warmup 10 --iterations 500 --lr 1e-4 --clip 0.1 --entropy 0.001 --tuned-boost 2`.

The league gain repeated once (gen3 to gen4) and then plateaued again (gen4 to gen5 is a tie), and a wider network trained from a longer DAgger run was much worse. The next bottleneck is no longer the teacher gap: gen4 already beats Hard by more than 4 goals, and the only opponents left that can beat it are its own ancestors. Ideas for the next round: a stronger or more varied sparring pool (tuned variants with random traits, exploiters trained against gen4, ability randomization), a per-player reward, and 11-a-side.

Other findings from this round:

- **Iterated tuning against Hard** (`tuning/search-vs-hard-seed2.json`, 16 generations) only reached about parity with Hard. The 10-parameter space is saturated near Hard, so it is not a stronger teacher.
- **An exploiter** (league1 fine-tuned against league1 only, `--exploit`) beat league1 by +0.73 +/- 0.35 but lost to Hard by 0.89. Training against one opponent gives a cycle, not progress.
- **Greedy play** is within noise of sampled play (+0.25 +/- 0.33), so the exported policy keeps sampling.
- A centralized critic was not tried: every row already contains the whole match, so the value head sees the full state.
- `npm run arena -- PLAYER...` compares any mix of `default`, `tuned`, `model:PATH` and `greedy:` players on shared seeds with a rating table. Use it for every promotion.

## Lessons from the league runs

What we learned from league1 to league3 and the 512-unit run. Read this before starting another run.

- **Measure with at least 128 matches.** A 16-match evaluation swings by 2 to 3 goals and gave misleading signals many times. 64 matches have a margin of about ±0.6, 128 about ±0.35, 256 about ±0.3. Quote nothing smaller, and judge a promotion by head to head play against the previous best model.
- **Training reward and training evaluations are not the promotion test.** The in-run "vs tuned" numbers were far noisier than a 128-match run afterwards, and the win rates inside the league pool describe the pool, not the held-out matches.
- **The league approach plateaued at league1.** league2 (more weight on Hard), league3 (2000 iterations) and a 512-unit network all lost or tied against league1 head to head. More iterations on the same pool is not the way forward.
- **Longer training can make a policy worse.** In league3 the entropy rose from 0.55 to about 1.0 and the KL fell to about 0, so learning stalled. Watch entropy and KL, and stop a run when entropy rises.
- **A wider network did not help against Hard.** big_league (512 units) was the best against the default CPU (+1.97) but lost to Hard by 3.6 goals. Size is not the bottleneck. The most likely cause is that imitation did not capture Hard's pressing and PPO did not recover it.
- **Strength against one opponent does not carry to another.** Each model is best against what it trained on. Always evaluate against the default CPU, Hard and the previous best before promoting.
- **Engine speed is not the bottleneck.** The engine step costs 1.6 ms (2.5 ms with a neural opponent), 8 workers scale about 5.4 times, the PPO update takes about 0.13 s on CPU, and a run reaches about 2,000 steps per second. No C++ or Rust rewrite is needed.
- **Evaluations used to dominate run time.** `--eval-envs` (default 16) and fewer evaluations (`--eval-every 250` to 400) fix that.
- **Anchors must be exported JSON policies, with absolute paths.** A `.pt` file or a relative path makes the environment server exit. Keep one anchor per generation in `models/anchors/` and only add a generation that was promoted.
- **Hidden size is read from the checkpoint.** `--hidden` only applies to a new policy. With `--init` the size comes from the checkpoint, and evaluation and export do the same.
- **Ideas not yet tried:** a per-player reward instead of one shared across the team, a different imitation teacher mix, larger DAgger rounds against Hard only, separate networks for attack and defense, and keeper and 11-a-side training.

## Ideas and open questions

- **Control granularity:** shared-weight per-player policy (current choice) versus one central policy for the whole team. Parameter sharing scales better, but a central policy may coordinate better. Revisit only if per-player play looks uncoordinated.
- **Action space:** `pass(target)` shrinks exploration a lot but hides lead and power from the policy. If it limits play quality, add finer power choices or a lead offset.
- **Observation:** pass-lane clearance is derived by the encoder. If the policy ignores it, try more derived features (offside line, nearest-defender angle) before adding engine data.
- **Reward hacking:** possession and ball-progress terms can be gamed (holding the ball in a corner, kicking it forward). Watch rollouts, not only the reward curve.
- **Episode length:** a full match is 7,200 ticks at 5v5 (1,200 policy steps). Short drills and truncated episodes give faster feedback.
- **Determinism:** seeds plus replay verification let us reproduce any training episode. Record seed, opponent, abilities and config with every evaluation.
- **Evaluation discipline:** judge every candidate by goal difference over a fixed set of held-out seeds, not by training reward.

## Decisions log

| Decision | Reason |
| --- | --- |
| Adapter built before tuning | Needed for every later stage, and cheap to verify with tests |
| Tuning before PPO | Gives a trusted fitness measure and a stronger opponent without an ML stack |
| Shared-weight per-player policy | Scales across 5v5 and 11v11 and fits the masked, padded interface |
| Attack-frame observations | Lets one policy play either side |
| Browser app can run human vs CPU and CPU vs CPU with Default or Hard profiles | Lets us watch and play against tuning results directly |
| Target-based pass actions | Avoids learning continuous aim and power from scratch |

## How to reproduce the trained player

```
npm run build
node scripts/record-demos.mjs --matches 40 --out demos/tuned              # teacher decisions as training data
python3 python/dagger.py --demos demos/tuned --name dagger2 --rounds 16 --rollout-steps 900
python3 python/train_ppo.py --name ft1 --init runs/dagger2/latest.pt --lr 1e-4 --entropy 0.001 --clip 0.1 \
  --value-warmup 10 --iterations 300 --eval-every 25
python3 python/evaluate_checkpoint.py runs/league1/latest.pt --matches 64       # held-out check in Python
python3 python/train_ppo.py --name league1 --init runs/ft1/latest.pt --league-anchor models/trained-5v5.json \
  --iterations 600 --lr 1e-4 --clip 0.1 --entropy 0.001 --eval-every 50 --snapshot-every 25 --snapshots-kept 6   # use the old model as anchor before exporting
python3 python/export_policy.py runs/league1/latest.pt models/trained-5v5.json --check test/fixtures/policy-check.json
npm run tune -- compare --model models/trained-5v5.json --opponent-profile tuned --matches 64   # held-out check in the engine
```

Run each Python script from the `python/` folder (they import each other). Runs take about 5 to 15 minutes each on a 12-core laptop.

## Restart support rule and pass bonus (gen5)

Observed in the browser: players dribble to a shot with little passing, and at throw-ins and corners teammates stand far away. Measured: at the moment a throw-in was taken the nearest teammate was ~19 m from the ball (corner ~28 m), for the Hard AI too, because `placeRestart` left everyone where they stood and nobody moves in the restart window. The trained model holds the ball in a dribble ~52% of live ticks versus ~33% for Hard.

- A pass bonus (`--pass-bonus` for full matches) of 0.05 and 0.3 did not change dribble share (55%, 54%) or strength. Reward nudges do not change this behavior.
- New engine rule `rules.restartSupportM` (arcade: 7 m): the team awarded a throw-in or corner starts with outfield players pulled toward the ball (nearest within 7 m, next 14 m, next 21 m). Off when unset.
- `role_sup1` (gen5): 400 iterations from gen4 under the new rule, anchors gen3 and gen4. Head to head against gen4 over 128 matches: +0.55 +/- 0.25. Promoted to `models/trained-5v5.json`; anchor `models/anchors/gen5-role_sup1.json`.
- Still open: dribble share is unchanged (~52%).

## Sparring variants and the assist reward (gen6)

- Five Hard-AI variants (more pressing, wider, dribbling, short passing, risky; `tuning/variants/`, local) all lose to gen5 by about 4 goals a match (41 to 47 wins of 48). They add only easy opponents, so no league run was made with them. `arena` now accepts `params:<path>`.
- New reward `assist` (`--assist-bonus`): added to a goal scored within 6 s of a teammate receiving a pass. `role_assist1` (400 iterations from gen5, bonus 0.5) beats gen5 by 0.45 +/- 0.26 over 128 matches. Promoted as gen6; anchor `models/anchors/gen6-role_assist1.json`.
- Passing is not rarer than for Hard: roughly 8 completed teammate passes per match for gen5 and gen6 against about 6.6 for Hard, although the dribble share of live ticks stays ~52 to 54% against ~29 to 33%. The model carries the ball longer between passes, rather than not passing.

## GRF-style drills mixed into league training (gen7)

Borrowed from Google Research Football's academy: short situations where the right play is a pass. New drills `three` (3v2, one defender presses the carrier) and `counter` (counterattack from the halfway line), and `--drill-mix P` / `--drill-kinds` start a share of league episodes from a drill (ending on a goal or dead ball, max 60 steps). Evaluation stays on full matches.

`role_drill1` (400 iterations from gen6, 20% drills, assist bonus 0.5, anchors gen5 and gen6) beats gen6 by 0.52 +/- 0.26 over 128 matches (W62 D36 L30) and completes about 9.7 teammate passes per match against 8.2 for gen6 and 6.6 for Hard. Promoted as gen7; anchor `models/anchors/gen7-role_drill1.json`.

`role_drill2` (400 iterations from gen7, 35% drills over five kinds, anchors gen6 and gen7) beat gen7 by only 0.27 +/- 0.23 (W50 D43 L35) with about 9.1 passes per match: not a clear gain, so not promoted. More drills gave diminishing returns; gen7 stays shipped.

## Ball progress reward ablation

Same recipe as gen7 (from gen6, 20% drills, assist 0.5, 400 iterations) with `--progress` 0.03 and 0, against gen7 (0.1), 128 matches each:

| progress | vs gen7 | passes per match | dribble share |
|---|---|---|---|
| 0.03 | gen7 -0.16 +/- 0.23 (W37 D44 L47) | 8.5 | 55% |
| 0 | gen7 +0.02 +/- 0.23 (W45 D39 L44) | 9.2 | 56% |
| 0.1 (gen7) | - | 9.7 | 58% |

No measurable effect on strength or on dribbling. The term neither helps nor hurts at this stage, so it was left at 0.1 and gen7 stays shipped. Goals and the assist bonus carry the learning.

## Turnover reward

New reward `turnover` (`--turnover-bonus`): +w when the last touch of the ball passes from the opponent to the learner's team (tackle, interception, forced error, an opponent pass going out), -w for the reverse, so it cannot be farmed; goals reset the tracking. `role_turn1` (gen7 recipe plus `--turnover-bonus 0.05`) vs gen7 over 128 matches: +0.23 +/- 0.26 (W54 D32 L42), 9.2 passes per match, dribble share 55%. Not a clear gain, so not promoted; gen7 stays shipped.

## Defensive drills and failure replay (gen8)

- New drills `defend` (2v1 against us, one defender back) and `defend2` (3v2 against us), where the rival carries toward our goal and the learner starts without the ball.
- `scripts/record-failures.mjs` plays a model against opponents (Hard, Default, earlier generations) and saves the state 300 ticks (5 s) before each goal it concedes as a scenario. 240 matches of gen7 gave 123 situations (`demos/failures.json`, local). `mirrorScenario` lets a situation recorded for the home team serve the away team. Training uses them with drill kind `replay` and `--replay-file`.
- `role_def1` (defensive drills plus turnover 0.05) vs gen7: +0.23 +/- 0.27 (W52 D39 L37), not clear, not promoted.
- `role_replay1` (400 iterations from gen7, 30% drills with three eighths of them failure replays, assist 0.5, turnover 0.05) vs gen7: +0.51 +/- 0.26 (W67 D32 L29) on seed 1 and +0.17 +/- 0.24 (W54 D37 L37) on seed 2; 121 wins to 66 over the 256 matches. The run's own small evaluation had it behind gen7 (-0.31), which is noise at that size. Promoted as gen8; anchor `models/anchors/gen8-role_replay1.json`.
