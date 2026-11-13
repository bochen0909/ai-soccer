import { appendFileSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { parseArgs } from "node:util";
import {
  ACT_NONE, AI_PROFILES, BaselineTeamAI, buildAgentContexts, createActionSpace, createArcadeConfig,
  createDefaultAIRegistry, createObservationLayout, FEEDBACK_STATUSES, intentToChoice, MatchSession,
  profileTeamOptions, writeActionMask, writeObservation,
} from "../dist/index.js";

const usage = `Usage: node scripts/record-demos.mjs [--out PATH] [--matches N] [--teacher default|tuned] [--seed N]

Plays a baseline profile (the teacher) through a normal match session and records every decision it makes as
training data for a policy: one row per player per decision, in the policy's observation and action format.
Opponents alternate between the default and tuned profiles, and the teacher alternates between home and away.
Writes PATH.obs (float32), PATH.mask (uint8), PATH.label (uint8: move, sprint, act) and PATH.json.`;

const { values } = parseArgs({
  options: { help: { type: "boolean" }, out: { type: "string" }, matches: { type: "string" },
    teacher: { type: "string" }, seed: { type: "string" } },
});
if (values.help) { console.log(usage); process.exit(0); }
const out = values.out ?? "demos/tuned";
const matches = Number(values.matches ?? 40);
const teacher = values.teacher ?? "tuned";
const baseSeed = Number(values.seed ?? 3_000_000);
if (!Object.hasOwn(AI_PROFILES, teacher)) throw new RangeError("--teacher must be default or tuned");
mkdirSync(dirname(out), { recursive: true });
for (const extension of ["obs", "mask", "label"]) rmSync(`${out}.${extension}`, { force: true });

const config = createArcadeConfig();
const players = config.teams[0].players.length;
const layout = createObservationLayout(players);
const space = createActionSpace(players);
const stats = { rows: 0, requests: 0, lost: 0, acts: new Array(space.actCount).fill(0), goalDiff: [], decisions: 0 };

for (let match = 0; match < matches; match += 1) {
  const learnerIndex = match % 2;
  const opponentProfile = (match >> 1) % 2 === 0 ? "default" : "tuned";
  const obsChunks = [];
  const maskChunks = [];
  const labelChunks = [];

  // The teacher is the stock baseline. This wrapper only watches its decisions.
  const registry = createDefaultAIRegistry();
  registry.register("teacher", () => {
    const inner = new BaselineTeamAI();
    return {
      id: "teacher", version: inner.version,
      reset: (context) => inner.reset(context),
      dispose: () => inner.dispose(),
      decide(observation, assignment, context) {
        const intents = inner.decide(observation, assignment, context);
        const byPlayer = new Map(intents.map((intent) => [intent.playerId, intent]));
        const feedback = new Map(context.actionResults.map((result) => [result.action.playerId, result.status]));
        const obs = new Float32Array(players * layout.size);
        const mask = new Uint8Array(players * space.actCount);
        const labels = new Uint8Array(players * 3);
        const agents = buildAgentContexts(observation, config, learnerIndex);
        agents.forEach((agent, slot) => {
          writeObservation(agent, layout, feedback.has(agent.ego.playerId) ? FEEDBACK_STATUSES.indexOf(feedback.get(agent.ego.playerId)) + 1 : 0,
            learnerIndex, obs, slot * layout.size);
          writeActionMask(agent, space, mask, slot * space.actCount);
          const intent = byPlayer.get(agent.ego.playerId) ?? { playerId: agent.ego.playerId, movement: { x: 0, y: 0 }, sprint: false };
          const choice = intentToChoice(intent, agent, space, mask.subarray(slot * space.actCount, (slot + 1) * space.actCount));
          labels.set([choice.move, choice.sprint, choice.act], slot * 3);
          if (intent.request) { stats.requests += 1; if (choice.lost) stats.lost += 1; }
          stats.acts[choice.act] += 1;
        });
        obsChunks.push(Buffer.from(obs.buffer));
        maskChunks.push(Buffer.from(mask.buffer));
        labelChunks.push(Buffer.from(labels.buffer));
        stats.decisions += 1;
        return intents;
      },
    };
  });

  const learnerId = config.teams[learnerIndex].id;
  const opponentId = config.teams[1 - learnerIndex].id;
  const session = new MatchSession(registry);
  session.reset(config, baseSeed + match, {
    teams: {
      [learnerId]: { pluginId: "teacher", config: structuredClone(AI_PROFILES[teacher].config) },
      [opponentId]: profileTeamOptions(opponentProfile),
    },
  });
  let result;
  try {
    do { result = session.step(); } while (result.status !== "finished");
  } catch (error) {
    console.error(`match ${match} failed (${error instanceof Error ? error.message : error}); its rows are dropped`);
    session.dispose();
    continue;
  }
  const { score } = session.snapshot();
  stats.goalDiff.push(learnerIndex === 0 ? score[0] - score[1] : score[1] - score[0]);
  session.dispose();
  appendFileSync(`${out}.obs`, Buffer.concat(obsChunks));
  appendFileSync(`${out}.mask`, Buffer.concat(maskChunks));
  appendFileSync(`${out}.label`, Buffer.concat(labelChunks));
  stats.rows += obsChunks.length * players;
  if ((match + 1) % 5 === 0) console.log(`recorded ${match + 1}/${matches} matches, ${stats.rows} rows`);
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
const meta = {
  rows: stats.rows, obsSize: layout.size, actCount: space.actCount, agents: players, matches: stats.goalDiff.length,
  teacher, teacherGoalDiff: mean(stats.goalDiff), requests: stats.requests, lostRequests: stats.lost, actCounts: stats.acts,
};
writeFileSync(`${out}.json`, `${JSON.stringify(meta, null, 2)}\n`);
console.log(`\n${meta.matches} matches, ${meta.rows} rows (${stats.decisions} decisions)`);
console.log(`teacher goal difference ${meta.teacherGoalDiff.toFixed(2)} per match against the default and tuned profiles`);
console.log(`requests the policy's actions could not express or the mask forbade: ${stats.lost} of ${stats.requests} (${(100 * stats.lost / Math.max(1, stats.requests)).toFixed(1)}%)`);
console.log(`act histogram (none, shoot L/C/R, tackle, save, pass slots, loft slots): ${stats.acts.join(" ")}`);
void ACT_NONE;
