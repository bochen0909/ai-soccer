# AI training plan

Working notes on how we get from the current baseline AI to a trained one. Update this file as steps finish or the plan changes. The interface itself is documented in [TRAINING.md](TRAINING.md); the engine design is in [DESIGN.md](DESIGN.md).

## Where we are

- Engine, headless runner, replay and `evaluate` (match, verify, benchmark) exist.
- The baseline AI (`BaselineTeamAI`) is the reference opponent.
- The training adapter is built: `SoccerTrainingEnv` in `src/training/` (observation encoder, action decoder and masks, reward, reset and step). Its tests are in `test/training.test.mjs`.
- Nothing has been trained or tuned yet. The reward and observation design are untested against a learning signal.

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
| league1 | League PPO from ft1 under the random-kickoff rule: pool of default, Hard, ft1 as anchor and 6 snapshots, 600 iterations, lr 1e-4, clip 0.1 | Held-out (64 matches each, Python): **+1.36 ±0.60 against the default CPU**, **+0.09 ±0.66 against Hard**, **+1.69 ±0.50 against ft1** (78% wins). In the engine, same seeds: +1.55 ±0.48 against the default CPU and -0.63 ±0.65 against Hard. Under the random-kickoff rule ft1 scored +1.41 ±0.64 and -0.86 ±0.60. The 16-match evaluations during training swung from -2.9 to +0.2 against Hard and were not a usable signal |

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

- **Fixed:** the physics engine threw `Physics exceeded its collision event limit` when several players piled up against a wall (seed 100050 at 5v5, tick 2959). The wall acts like an infinitely heavy body, so contacts between the stacked players traded zero-time impulses forever. The fallback in `src/engine/physics.ts` stops the jammed players, as an inelastic pile-up would, and only throws if the same substep overruns again. A regression test is in `test/headless.test.mjs`. Seeds 100000 to 100119 now all complete.
- The tuner still counts any remaining engine crash as a draw and reports the seeds, so one bad seed cannot kill a search.
- Players that end up overlapping (about 0.1 m in the pile above) are not pushed apart after they stop. They separate once they move away from each other. Revisit if overlaps persist visibly.

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
  --value-warmup 10 --iterations 300 --eval-every 25 --eval-episodes 4
python3 python/evaluate_checkpoint.py runs/league1/latest.pt --matches 64       # held-out check in Python
python3 python/train_ppo.py --name league1 --init runs/ft1/latest.pt --league-anchor models/trained-5v5.json \
  --iterations 600 --lr 1e-4 --clip 0.1 --entropy 0.001 --eval-every 50 --snapshot-every 25 --snapshots-kept 6   # use the old model as anchor before exporting
python3 python/export_policy.py runs/league1/latest.pt models/trained-5v5.json --check test/fixtures/policy-check.json
npm run tune -- compare --model models/trained-5v5.json --opponent-profile tuned --matches 64   # held-out check in the engine
```

Run each Python script from the `python/` folder (they import each other). Runs take about 5 to 15 minutes each on a 12-core laptop.
