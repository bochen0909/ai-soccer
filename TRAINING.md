# Training interface

`SoccerTrainingEnv` (in `src/training/`) lets a trainer control one team while the other runs any registered controller, by default the baseline AI. It reuses `MatchSession` and `SoccerEngine` unchanged: the learner team is an ordinary `TeamAI` plugin named `external`.

```js
import { SoccerTrainingEnv } from "./dist/index.js";

const env = new SoccerTrainingEnv({ maxPlayersPerTeam: 11 });
let result = env.reset(seed);
while (!result.terminated && !result.truncated) {
  result = env.step(actions); // Int32 [move, sprint, act] per agent
}
```

One policy step advances `ticksPerStep` ticks (default 6, 10 Hz). Movement and sprint hold for the whole step. A kick, tackle, or save request is submitted once, on the first tick.

## Agents and symmetry

The trainer controls every player on the learner team, one agent per player, in roster order. Use one shared-weight policy for all agents. Roster slot `k` is the same player in observations (teammate block `k`) and in actions (pass to `k`).

Everything is expressed in the team's **attack frame**: world coordinates are rotated by 180 degrees for a team that attacks toward negative x. The policy therefore always attacks toward +x, and learning transfers between the home and away sides. Actions are decoded back to world coordinates.

`maxPlayersPerTeam` pads observations and actions so one network fits 5v5 and 11v11. Absent agents have `agentMask = 0`, zero observations, and only "none" legal.

## Observation

One row of `layout.size = 63 + 21 * maxPlayers` floats per agent. Lengths divide by half the pitch (positions) or the pitch length (offsets); speeds divide by 10 m/s (ball 30 m/s).

| Block | Size | Contents |
| --- | --- | --- |
| Ego | 39 | Position, velocity, facing (cos, sin), stamina, balance, keeper flag, 11 abilities, action phase one-hot (4), previous-step action result one-hot (none, accepted, completed, interrupted, rejected), has-ball, ball-in-reach, formation position, ball distance and offset, opposing-goal offset and distance |
| Ball | 12 | Position, velocity, control one-hot (free, own dribble, opponent dribble, own keeper holds, opponent keeper holds), last toucher side |
| Match | 12 | Period time left, period, score difference, phase flags (live, own restart, opponent restart, paused), is-taker, restart kind one-hot (kickoff, throw-in, corner, goal kick) |
| Teammates | 11 x N | Per roster slot: present (0 for the agent itself), offset, position, velocity, keeper flag, stamina, distance, pass-lane clearance from the ball |
| Opponents | 10 x N | Per roster slot: present, offset, position, velocity, keeper flag, distance to ego, distance to ball |

## Actions

Each agent submits `[move, sprint, act]`.

| Field | Values |
| --- | --- |
| `move` | 0 stop, 1..8 eight directions in 45 degree steps counterclockwise from the attack direction |
| `sprint` | 0 or 1 |
| `act` | 0 none, 1..3 shoot (left, center, right of goal), 4 tackle, 5 keeper save, `6 + k` pass to slot k, `6 + N + k` lofted pass to slot k |

The adapter turns target choices into engine requests: it leads the receiver by 0.32 s, then derives direction and power from range (the same formulas as the baseline). The same `pass` act becomes a throw-in at a throw-in, and a roll or throw when the keeper holds the ball.

`actionMask` (`[agents, actCount]`) marks legal acts. A pass needs the ball within reach, restarts allow only the taker, a save needs the keeper in its area, and so on. Illegal acts are replaced by "none" and counted in `info.illegalActions`. Mask the policy's logits with it.

## Reward

Team-level scalar, in `DEFAULT_REWARD`:

| Term | Default | When |
| --- | --- | --- |
| `goalScored` / `goalConceded` | +1 / -1 | A goal event |
| `ballProgress` | 0.1 per pitch length | Ball advance toward the opposing goal across a step of uninterrupted live play |
| `possession` | 0.002 | Per step, plus for own control and minus for opponent control |

`env.shapingScale` multiplies the two shaping terms. Anneal it toward 0 so the policy ends up optimizing goals only.

## Teacher labels, imitation and export

- `scripts/record-demos.mjs` records a baseline profile's decisions in this document's observation and action format. `intentToChoice` converts a controller's decision to `[move, sprint, act]`: movement snaps to the nearest of 8 directions, and a pass becomes "pass to slot k", using the named receiver or else the best-aligned legal teammate.
- The env server's `teacher` option returns, for every state it reports, what a baseline profile would do there. That is what DAgger (`python/dagger.py`) uses to label the states the student visits.
- `python/export_policy.py` writes a checkpoint to a JSON file of exact float32 weights, and `NeuralTeamAI` (team AI "neural", `src/controllers/neural.ts`) plays with it. A controller's `AIContext` includes `opponents` (the other roster in order), because the policy's input lists opponents in roster order.

## Not built yet

- A process bridge (stdio or socket) for Python trainers, and a worker pool. One environment runs about 750 steps per second at 5v5 and 350 at 11v11 on one core.
- Self-play: the opponent cannot yet be a second `external` team.
- A stronger trained player: league training against several opponents, and 11-a-side.
