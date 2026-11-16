// Hosts several SoccerTrainingEnv instances and serves them over stdio for a Python trainer.
// Options include drill ("shoot", "attack", "pass", "three", "counter", "defend", "defend2", "giveandgo" or "rondo"), or drillMix {p, kinds, maxSteps} (kind "replay" draws from replayFile, scenarios recorded by record-failures.mjs) to start some episodes as drills for short situation drills that end on a goal or dead ball.
// Frames, both directions: u32 little-endian length, u8 type, body.
//   1 INIT  body JSON  -> 129 JSON {envs, agents, obsSize, actCount, moveCount}
//   2 RESET body JSON  -> 130 binary step payload
//   3 STEP  body int32[envs * agents * 3] -> 131 binary step payload
//   4 CLOSE
//   5 SET_POOL body JSON [{id, spec, weight}] (no reply): the opponents later episodes are drawn from, by weight.
//     A spec is "default", "tuned", "params:<path to a tuned parameters file>" or "model:<path to a policy JSON file>".
// Step payload, each section padded to 4 bytes: obs f32[E*A*D], mask u8[E*A*ACT], agentMask u8[E*A],
// reward f32[E], done u8[E], score i32[E*2] (own, opponent; valid when done), steps i32[E] (length of the ended episode,
// or -1 when the engine failed and the episode was dropped), opponent i32[E] (pool ID of the ended episode's
// opponent), teacher u8[E*A*3] (with the `teacher` option: the
// [move, sprint, act] a baseline profile would pick in the returned state, for imitation; zeros otherwise),
// pure u8[E] (1 when the returned observation belongs to a pure drill episode, see pureDrills).
// Option pureDrills (with drillMix): drill kinds whose episodes are "pure": goal-only reward, and they end when anyone but
// the carrier touches the ball. The trainer should learn only from the carrier's samples in them.
// An environment that ends is reset at once, so the returned observation starts the next episode.
import { readFileSync } from "node:fs";
import {
  AI_PROFILES, BaselineTeamAI, buildAgentContexts, createActionSpace, createArcadeConfig, createDefaultAIRegistry,
  createDrill, createElevenAsideConfig, mirrorScenario, decodeWeights, intentToChoice, profileTeamOptions, registerNeuralNetwork,
  SoccerTrainingEnv, writeActionMask,
} from "../dist/index.js";

/** Every reward term off but scoring. */
const PURE_REWARD = {
  goalScored: 1, goalConceded: 0, ballProgress: 0, possession: 0, passCompleted: 0, assist: 0, turnover: 0,
  passBreak: 0, save: 0, passFailed: 0, outOfBounds: 0, shotOnTarget: 0,
};
const INIT = 1, RESET = 2, STEP = 3, CLOSE = 4, SET_POOL = 5;
let slots = [];
let layout;
// Opponents the learner can meet: {id, spec, weight}. A spec is "default", "tuned" or "model:<path to a policy file>".
let pool = [];
const networks = new Map();

function networkFor(path) {
  if (!networks.has(path)) networks.set(path, decodeWeights(JSON.parse(readFileSync(path, "utf8"))));
  return networks.get(path);
}

/** Baseline settings for a profile name, or for "params:<path>" a tuned parameters file (as written by `tune search`). */
function baselineConfig(name) {
  if (!name.startsWith("params:")) return structuredClone(AI_PROFILES[name].config);
  const data = JSON.parse(readFileSync(name.slice("params:".length), "utf8"));
  return structuredClone(data.config ?? data.best?.config ?? data);
}

function makeEnv(options, learnerTeamId, spec) {
  const config = options.size === 11 ? createElevenAsideConfig() : createArcadeConfig();
  if (options.periodTicks) config.periodTicks = [options.periodTicks, options.periodTicks];
  const registry = createDefaultAIRegistry();
  let opponent;
  if (spec.startsWith("model:")) {
    registerNeuralNetwork(registry, networkFor(spec.slice("model:".length)), "neural");
    opponent = { pluginId: "neural" };
  } else {
    opponent = spec.startsWith("params:") ? { pluginId: "baseline", config: baselineConfig(spec) } : profileTeamOptions(spec);
  }
  return new SoccerTrainingEnv({
    config, learnerTeamId, opponent, registry,
    ticksPerStep: options.ticksPerStep, maxSteps: options.maxSteps, reward: options.reward,
    maxPlayersPerTeam: options.maxPlayersPerTeam, roleFeatures: options.roleFeatures, endOnDeadBall: options.drill !== undefined || options.endOnDeadBall, endOnBallLoss: options.endOnBallLoss,
  });
}

let replays = [];
/** A recorded situation (written for the home team as learner), mirrored when the learner is the away team. */
function replayScenario(side, seed) {
  const scenario = replays[seed % replays.length];
  return side === 0 ? scenario : mirrorScenario(scenario);
}

function init(options) {
  if (options.replayFile) replays = JSON.parse(readFileSync(options.replayFile, "utf8"));
  pool = options.pool ?? [{ id: 0, spec: options.opponent ?? "default", weight: 1 }];
  slots = Array.from({ length: options.envs }, (_, index) => ({
    index, episode: 0, side: 0, steps: 0, envs: new Map(), current: undefined, last: undefined, opponentId: pool[0].id,
    random: mulberry32((options.seedBase ?? 0) + index * 101 + 7),
  }));
  const probe = makeEnv(options, "home", pool[0].spec);
  layout = {
    envs: options.envs, agents: probe.agentCount, obsSize: probe.layout.size,
    actCount: probe.actionSpace.actCount, moveCount: probe.actionSpace.moveCount,
  };
  slots.options = options;
  return layout;
}

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

/** Picks the next opponent in proportion to the pool weights. */
function pickOpponent(slot) {
  const total = pool.reduce((sum, entry) => sum + entry.weight, 0);
  let draw = slot.random() * total;
  for (const entry of pool) {
    draw -= entry.weight;
    if (draw <= 0) return entry;
  }
  return pool.at(-1);
}

function seedFor(slot) {
  const { seedBase = 0 } = slots.options;
  return (seedBase + slot.index * 7919 + slot.episode * 104729) >>> 0;
}

/** Starts the next episode in a slot. The learner alternates between the home and away team. */
function begin(slot, seed) {
  slot.side = slot.episode % 2;
  const opponent = pickOpponent(slot);
  // With drillMix, some episodes are short drills (ending on a goal or dead ball) instead of whole matches.
  const mix = slots.options.drillMix;
  const drill = slots.options.drill ?? (mix && slot.random() < mix.p ? mix.kinds[Math.floor(slot.random() * mix.kinds.length)] : undefined);
  const pure = !slots.options.drill && drill !== undefined && (slots.options.pureDrills ?? []).includes(drill);
  const key = `${slot.side}|${opponent.spec}|${pure ? "pure" : drill ? "drill" : "match"}`;
  if (!slot.envs.has(key)) {
    // A rondo has no goal: it ends after 90 steps, and holding the ball earns its own reward instead of progress.
    const rondo = drill === "rondo" && !slots.options.drill;
    const envOptions = drill && !slots.options.drill ? {
      ...slots.options, drill, maxSteps: rondo ? 90 : mix.maxSteps ?? 60,
      reward: rondo ? { ...slots.options.reward, possession: 0.01, ballProgress: 0 } : pure ? PURE_REWARD : slots.options.reward,
      endOnBallLoss: pure || slots.options.endOnBallLoss,
    } : slots.options;
    slot.envs.set(key, makeEnv(envOptions, slot.side === 0 ? "home" : "away", opponent.spec));
  }
  slot.current = slot.envs.get(key);
  slot.pure = pure;
  slot.opponentId = opponent.id;
  slot.steps = 0;
  const { teacher } = slots.options;
  if (teacher) {
    slot.teacher = new BaselineTeamAI();
    slot.teacher.reset({
      teamId: slot.current.learnerTeamId, teamIndex: slot.side, players: slot.current.config.teams[slot.side].players,
      config: baselineConfig(teacher), contractVersion: 1, seed: 0,
    });
  }
  for (let attempt = 0; ; attempt += 1) {
    const episodeSeed = (seed ?? seedFor(slot)) + attempt * 15_485_863 >>> 0;
    try {
      // Drills start from a built situation. Without one, the match starts from kickoff.
      const scenario = drill === "replay" ? replayScenario(slot.side, episodeSeed)
        : drill ? createDrill(drill, slot.current.config, slot.side, episodeSeed) : undefined;
      slot.last = slot.current.reset(episodeSeed, scenario);
      break;
    } catch (error) {
      // A seed that builds an invalid situation is skipped, not fatal.
      process.stderr.write(`env-server: seed ${episodeSeed} could not start an episode: ${error instanceof Error ? error.message : error}\n`);
      if (attempt >= 5) throw error;
    }
  }
  slot.episode += 1;
}

/** What a baseline profile would do in the slot's current state, as the policy's [move, sprint, act] choices. */
function teach(slot) {
  const config = slot.current.config;
  const observation = slot.current.observation();
  const space = createActionSpace(layout.agents);
  const learnerIds = config.teams[slot.side].players.map(({ id }) => id);
  const intents = slot.teacher.decide(structuredClone(observation), { playerIds: learnerIds, revision: 0 },
    { actionResults: [], passIntention: null });
  const byPlayer = new Map(intents.map((intent) => [intent.playerId, intent]));
  const choices = new Uint8Array(layout.agents * 3);
  const mask = new Uint8Array(space.actCount);
  buildAgentContexts(observation, config, slot.side).forEach((agent, index) => {
    writeActionMask(agent, space, mask, 0);
    const intent = byPlayer.get(agent.ego.playerId) ?? { playerId: agent.ego.playerId, movement: { x: 0, y: 0 }, sprint: false };
    const choice = intentToChoice(intent, agent, space, mask);
    choices.set([choice.move, choice.sprint, choice.act], index * 3);
  });
  return choices;
}

function payload(results, ended) {
  const { envs, agents, obsSize, actCount } = layout;
  const pad = (bytes) => (bytes + 3) & ~3;
  const sizes = {
    obs: envs * agents * obsSize * 4, mask: pad(envs * agents * actCount), agentMask: pad(envs * agents),
    reward: envs * 4, done: pad(envs), score: envs * 8, steps: envs * 4, opponent: envs * 4,
    teacher: pad(envs * agents * 3), pure: pad(envs),
  };
  const total = Object.values(sizes).reduce((sum, value) => sum + value, 0);
  const buffer = new ArrayBuffer(total);
  let offset = 0;
  const take = (bytes) => { const start = offset; offset += bytes; return start; };
  const obs = new Float32Array(buffer, take(sizes.obs), envs * agents * obsSize);
  const mask = new Uint8Array(buffer, take(sizes.mask), envs * agents * actCount);
  const agentMask = new Uint8Array(buffer, take(sizes.agentMask), envs * agents);
  const reward = new Float32Array(buffer, take(sizes.reward), envs);
  const done = new Uint8Array(buffer, take(sizes.done), envs);
  const score = new Int32Array(buffer, take(sizes.score), envs * 2);
  const steps = new Int32Array(buffer, take(sizes.steps), envs);
  const opponent = new Int32Array(buffer, take(sizes.opponent), envs);
  const teacher = new Uint8Array(buffer, take(sizes.teacher), envs * agents * 3);
  const pure = new Uint8Array(buffer, take(sizes.pure), envs);
  slots.forEach((slot, index) => { pure[index] = slot.pure ? 1 : 0; });
  if (slots.options.teacher) slots.forEach((slot, index) => teacher.set(teach(slot), index * agents * 3));
  results.forEach((result, index) => {
    obs.set(result.observation, index * agents * obsSize);
    mask.set(result.actionMask, index * agents * actCount);
    agentMask.set(result.agentMask, index * agents);
  });
  ended.forEach((end, index) => {
    reward[index] = end.reward;
    done[index] = end.done ? 1 : 0;
    score[index * 2] = end.score[0];
    score[index * 2 + 1] = end.score[1];
    steps[index] = end.failed ? -1 : end.steps;
    opponent[index] = end.opponent ?? 0;
  });
  return Buffer.from(buffer);
}

function reset(options) {
  const seeds = options.seeds;
  slots.forEach((slot, index) => {
    if (options.seedBase !== undefined) slots.options.seedBase = options.seedBase;
    slot.episode = options.episode ?? slot.episode;
    begin(slot, seeds?.[index]);
  });
  return payload(slots.map((slot) => slot.last), slots.map((slot) => ({ reward: 0, done: false, score: [0, 0], steps: 0, opponent: slot.opponentId })));
}

function step(actions) {
  const per = layout.agents * 3;
  const ended = [];
  const results = slots.map((slot, index) => {
    let result;
    try {
      result = slot.current.step(actions.subarray(index * per, (index + 1) * per));
    } catch (error) {
      // An engine failure must not end a long training run. Drop the episode and start the next one.
      process.stderr.write(`env-server: episode ${slot.episode} in slot ${index} failed at tick ` +
        `${slot.current.observation().tick} and was dropped: ${error instanceof Error ? error.message : error}\n`);
      ended.push({ reward: 0, done: true, score: [0, 0], steps: slot.steps, failed: true, opponent: slot.opponentId });
      begin(slot);
      return slot.last;
    }
    slot.steps += 1;
    const done = result.terminated || result.truncated;
    ended.push({ reward: result.reward, done, score: result.info.score, steps: slot.steps, opponent: slot.opponentId });
    if (done) begin(slot);
    else slot.last = result;
    return slot.last;
  });
  return payload(results, ended);
}

let pending = Buffer.alloc(0);
function send(type, body) {
  const header = Buffer.alloc(5);
  header.writeUInt32LE(body.length + 1, 0);
  header.writeUInt8(type, 4);
  process.stdout.write(Buffer.concat([header, body]));
}

function handle(type, body) {
  if (type === INIT) send(129, Buffer.from(JSON.stringify(init(JSON.parse(body.toString())))));
  else if (type === RESET) send(130, reset(JSON.parse(body.toString() || "{}")));
  else if (type === STEP) {
    // Copy so the Int32Array is aligned, whatever offset the socket chunk had.
    const aligned = new Int32Array(body.buffer.slice(body.byteOffset, body.byteOffset + body.length));
    send(131, step(aligned));
  } else if (type === SET_POOL) {
    const entries = JSON.parse(body.toString());
    if (!Array.isArray(entries) || entries.length === 0) throw new RangeError("The opponent pool needs at least one entry");
    pool = entries;
  } else if (type === CLOSE) process.exit(0);
}

process.stdin.on("data", (chunk) => {
  pending = Buffer.concat([pending, chunk]);
  while (pending.length >= 5) {
    const length = pending.readUInt32LE(0);
    if (pending.length < 4 + length) break;
    const type = pending.readUInt8(4);
    const body = pending.subarray(5, 4 + length);
    pending = pending.subarray(4 + length);
    try {
      handle(type, body);
    } catch (error) {
      process.stderr.write(`env-server error: ${error instanceof Error ? error.stack : error}\n`);
      process.exit(1);
    }
  }
});
process.stdin.on("end", () => process.exit(0));
