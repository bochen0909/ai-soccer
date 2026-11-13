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
- [ ] Easier attack drill ladder and pass-completion reward (see RL results)
- [ ] Fast-forward dead phases (goal pause, halftime, restart placement) so steps are not wasted on them
- [ ] Opponent can be a second `external` team (self-play)
- [ ] Per-agent rewards if the shared team reward turns out to be too coarse for credit assignment

Training infrastructure
- [x] stdio bridge (`scripts/env-server.mjs`) and Python client (`python/soccer_env.py`); no Gymnasium dependency, a vectorized client with masks and auto-reset
- [x] Persistent worker pool: several Node processes, several environments each, batched stepping (about 2,500 to 3,000 steps per second on 12 cores)
- [ ] Throughput target and profiling (the learner decides every tick, and cloning observations per tick is the main overhead)
- [ ] Policy export and in-engine inference as a controller plugin (for example ONNX)

## RL results so far

Everything below used the PPO trainer (`python/train_ppo.py`) with one shared network over all five players, masked action heads, 16 parallel matches and about 2,500 to 3,000 steps per second.

| Run | Setup | Result |
| --- | --- | --- |
| run2 | Full 5v5 matches against the default CPU, shaped reward, 200 iterations (about 400,000 steps) | Flat. Evaluation goal difference stayed between -3 and -6 per match against the default CPU and between -10 and -14 against Hard (the starting policy scores about -5 and -12). Entropy fell only slightly, from 2.89 to 2.66 |
| shoot1 | Shooting drill: striker with the ball 8 to 16 m out against the keeper, 6 s episodes, 200 iterations | Learned within about 20 iterations, then plateaued at 50 to 70% scoring. For comparison: idle 0%, random 12%, scripted "shoot centre" 5 to 35%, scripted "shoot at a corner from 12 to 16 m" about 58 to 60%. The policy has matched the best simple rule, mostly by aiming at corners |
| attack1 | 2v1 attack drill, started from the shooting policy, 250 iterations | 0 to 4% scoring throughout. A scripted run-and-shoot policy scores 0 to 5%, so the drill can only be solved by passing or dribbling past the defender, and the reward arrives only after a whole pass, run and shot line up |

What this tells us:
- The pipeline works end to end: environment, masks, bridge, trainer, drills and evaluation.
- Dense, short tasks are learned in minutes. Full matches and multi-step plays need more than plain PPO from scratch with a goals-only signal.
- The default keeper almost always stops centre shots and often misses corner shots. Corner placement is the main skill the shooting drill taught.

Next, in the order I'd try them:
1. **Easier attack drill ladder.** Start with the defender absent and a pass to the runner required, then add the defender. Add a pass-completion bonus, such as a reward when a pass reaches a teammate who then has a shot.
2. **Longer and larger runs** once a drill moves, with more parallel environments, and a tuned entropy coefficient.
3. **Reward shaping for full matches:** shots on target, passes received, and tackles won, annealed with `shapingScale`.
4. **Imitation start:** behavior-clone the tuned CPU's decisions (hard to beat as a teacher, since it wins 297 of 300 against the default) and then fine-tune with RL.
5. **Evolution on the network weights** as an alternative to PPO, using the same environment.

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
