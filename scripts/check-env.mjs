import { parseArgs } from "node:util";
import {
  ACT_NONE, ACT_SAVE, ACT_SHOOT_CENTER, ACT_TACKLE, ACTION_FIELDS, createArcadeConfig, createElevenAsideConfig,
  EGO_INDEX, profileTeamOptions, SoccerTrainingEnv,
} from "../dist/index.js";

const usage = `Usage: npm run --silent check-env -- [--matches N] [--seed N] [--size 5|11] [--opponent default|tuned]

Plays scripted policies through SoccerTrainingEnv and checks that rewards, masks and observations behave.
Policies: idle (everyone stands still), random (uniform over legal acts), chase (nearest outfield player
chases the ball, tackles and shoots when legal; the rest hold position).`;

const { values } = parseArgs({
  options: { help: { type: "boolean" }, matches: { type: "string" }, seed: { type: "string" },
    size: { type: "string" }, opponent: { type: "string" } },
});
if (values.help) { console.log(usage); process.exit(0); }
const matches = Number(values.matches ?? 6);
const baseSeed = Number(values.seed ?? 1);
const size = values.size === "11" ? 11 : 5;
const opponent = values.opponent ?? "default";

/** Move choice 1..8 for a direction in the attack frame (0 stops). */
function directionChoice(dx, dy) {
  const angle = (Math.atan2(dy, dx) + 2 * Math.PI) % (2 * Math.PI);
  return 1 + (Math.round(angle / (Math.PI / 4)) % 8);
}

function rowOf(env, result, agent) {
  return result.observation.subarray(agent * env.layout.size, (agent + 1) * env.layout.size);
}

function maskOf(env, result, agent) {
  const { actCount } = env.actionSpace;
  return result.actionMask.subarray(agent * actCount, (agent + 1) * actCount);
}

const policies = {
  idle: (env) => new Array(env.agentCount * ACTION_FIELDS).fill(0),
  random(env, result, random) {
    const actions = new Array(env.agentCount * ACTION_FIELDS).fill(0);
    for (let agent = 0; agent < env.agentCount; agent += 1) {
      if (!result.agentMask[agent]) continue;
      const legal = [...maskOf(env, result, agent)].flatMap((value, act) => value ? [act] : []);
      actions[agent * 3] = Math.floor(random() * 9);
      actions[agent * 3 + 1] = random() < 0.5 ? 1 : 0;
      actions[agent * 3 + 2] = legal[Math.floor(random() * legal.length)];
    }
    return actions;
  },
  chase(env, result) {
    const actions = new Array(env.agentCount * ACTION_FIELDS).fill(0);
    let chaser = -1;
    let nearest = Infinity;
    for (let agent = 0; agent < env.agentCount; agent += 1) {
      if (!result.agentMask[agent]) continue;
      const row = rowOf(env, result, agent);
      if (row[EGO_INDEX.keeper] === 1) continue;
      if (row[EGO_INDEX.ballDistance] < nearest) { nearest = row[EGO_INDEX.ballDistance]; chaser = agent; }
    }
    for (let agent = 0; agent < env.agentCount; agent += 1) {
      if (!result.agentMask[agent]) continue;
      const mask = maskOf(env, result, agent);
      const row = rowOf(env, result, agent);
      if (agent === chaser) {
        const hasBall = row[EGO_INDEX.hasBall] === 1;
        // With the ball, run at goal. Without it, run at the ball.
        const target = hasBall ? [row[EGO_INDEX.goalOffsetX], row[EGO_INDEX.goalOffsetY]]
          : [row[EGO_INDEX.ballOffsetX], row[EGO_INDEX.ballOffsetY]];
        actions[agent * 3] = directionChoice(target[0], target[1]);
        actions[agent * 3 + 1] = 1;
        actions[agent * 3 + 2] = mask[ACT_SHOOT_CENTER] && row[EGO_INDEX.goalDistance] < 0.2 ? ACT_SHOOT_CENTER
          : mask[ACT_TACKLE] ? ACT_TACKLE : ACT_NONE;
      } else if (mask[ACT_SAVE]) {
        actions[agent * 3 + 2] = ACT_SAVE;
      }
      // Restart takers pass to the first legal teammate so the match keeps moving.
      if (actions[agent * 3 + 2] === ACT_NONE) {
        const pass = [...mask].findIndex((value, act) => value && act >= env.actionSpace.passBase && act < env.actionSpace.loftBase);
        if (pass >= 0 && mask[ACT_SHOOT_CENTER] === 0) actions[agent * 3 + 2] = pass;
      }
    }
    return actions;
  },
};

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const problems = [];
const note = (message) => { if (!problems.includes(message)) problems.push(message); };

function play(name, shaping) {
  const config = size === 11 ? createElevenAsideConfig() : createArcadeConfig();
  const env = new SoccerTrainingEnv({
    config, opponent: profileTeamOptions(opponent),
    reward: shaping ? undefined : { ballProgress: 0, possession: 0 },
  });
  const stats = { diff: 0, reward: 0, steps: 0, illegal: 0, maxAbs: 0, shootLegal: 0, passLegal: 0, liveSteps: 0 };
  for (let match = 0; match < matches; match += 1) {
    const random = mulberry32(baseSeed * 1000 + match);
    let result = env.reset(baseSeed * 1000 + match);
    let matchReward = 0;
    while (!result.terminated && !result.truncated) {
      const actions = policies[name](env, result, random);
      result = env.step(actions);
      matchReward += result.reward;
      stats.steps += 1;
      stats.illegal += result.info.illegalActions;
      for (const value of result.observation) {
        if (!Number.isFinite(value)) note("non-finite observation value");
        stats.maxAbs = Math.max(stats.maxAbs, Math.abs(value));
      }
      for (let agent = 0; agent < env.agentCount; agent += 1) {
        const mask = maskOf(env, result, agent);
        if (result.agentMask[agent] && mask[ACT_NONE] !== 1) note('"none" was masked out');
        if (!result.agentMask[agent] && mask.some((value, act) => act !== ACT_NONE && value)) note("padded agent has legal acts");
        if (result.agentMask[agent] && mask[ACT_SHOOT_CENTER]) stats.shootLegal += 1;
        if (result.agentMask[agent] && mask.slice(env.actionSpace.passBase, env.actionSpace.loftBase).some(Boolean)) stats.passLegal += 1;
      }
    }
    const [own, other] = result.info.score;
    stats.diff += own - other;
    stats.reward += matchReward;
    if (!shaping && Math.abs(matchReward - (own - other)) > 1e-9) note(`goal-only return ${matchReward} differs from score difference ${own - other}`);
  }
  return stats;
}

const f = (value, digits = 2) => value.toFixed(digits);
console.log(`Scripted policies vs ${opponent} baseline, ${matches} matches each (${size}v${size}), first-listed team plays home`);
console.log("policy   goal diff/match  return/match  illegal acts  max |obs|  steps with shot legal / pass legal");
for (const name of Object.keys(policies)) {
  const goalsOnly = play(name, false);
  const shaped = play(name, true);
  console.log(`${name.padEnd(8)} ${f(goalsOnly.diff / matches).padStart(15)} ${f(shaped.reward / matches).padStart(13)}` +
    ` ${String(goalsOnly.illegal).padStart(13)} ${f(Math.max(goalsOnly.maxAbs, shaped.maxAbs)).padStart(10)}` +
    `   ${goalsOnly.shootLegal} / ${goalsOnly.passLegal}`);
}
console.log(problems.length === 0 ? "\nNo problems found." : `\nProblems:\n  - ${problems.join("\n  - ")}`);
process.exit(problems.length === 0 ? 0 : 1);
