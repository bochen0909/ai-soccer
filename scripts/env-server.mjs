// Hosts several SoccerTrainingEnv instances and serves them over stdio for a Python trainer.
// Options include drill ("shoot" or "attack") for short situation drills that end on a goal or dead ball.
// Frames, both directions: u32 little-endian length, u8 type, body.
//   1 INIT  body JSON  -> 129 JSON {envs, agents, obsSize, actCount, moveCount}
//   2 RESET body JSON  -> 130 binary step payload
//   3 STEP  body int32[envs * agents * 3] -> 131 binary step payload
//   4 CLOSE
// Step payload, each section padded to 4 bytes: obs f32[E*A*D], mask u8[E*A*ACT], agentMask u8[E*A],
// reward f32[E], done u8[E], score i32[E*2] (own, opponent; valid when done), steps i32[E] (length of the ended episode,
// or -1 when the engine failed and the episode was dropped).
// An environment that ends is reset at once, so the returned observation starts the next episode.
import {
  createArcadeConfig, createDrill, createElevenAsideConfig, profileTeamOptions, SoccerTrainingEnv,
} from "../dist/index.js";

const INIT = 1, RESET = 2, STEP = 3, CLOSE = 4;
let slots = [];
let layout;

function makeEnv(options, learnerTeamId) {
  const config = options.size === 11 ? createElevenAsideConfig() : createArcadeConfig();
  if (options.periodTicks) config.periodTicks = [options.periodTicks, options.periodTicks];
  return new SoccerTrainingEnv({
    config, learnerTeamId, opponent: profileTeamOptions(options.opponent ?? "default"),
    ticksPerStep: options.ticksPerStep, maxSteps: options.maxSteps, reward: options.reward,
    maxPlayersPerTeam: options.maxPlayersPerTeam, endOnDeadBall: options.drill !== undefined || options.endOnDeadBall,
  });
}

function init(options) {
  slots = Array.from({ length: options.envs }, (_, index) => ({
    index, episode: 0, side: 0, steps: 0, envs: [undefined, undefined], current: undefined, last: undefined,
  }));
  const probe = makeEnv(options, "home");
  layout = {
    envs: options.envs, agents: probe.agentCount, obsSize: probe.layout.size,
    actCount: probe.actionSpace.actCount, moveCount: probe.actionSpace.moveCount,
  };
  slots.options = options;
  return layout;
}

function seedFor(slot) {
  const { seedBase = 0 } = slots.options;
  return (seedBase + slot.index * 7919 + slot.episode * 104729) >>> 0;
}

/** Starts the next episode in a slot. The learner alternates between the home and away team. */
function begin(slot, seed) {
  slot.side = slot.episode % 2;
  slot.envs[slot.side] ??= makeEnv(slots.options, slot.side === 0 ? "home" : "away");
  slot.current = slot.envs[slot.side];
  slot.steps = 0;
  const { drill } = slots.options;
  for (let attempt = 0; ; attempt += 1) {
    const episodeSeed = (seed ?? seedFor(slot)) + attempt * 15_485_863 >>> 0;
    try {
      // Drills start from a built situation. Without one, the match starts from kickoff.
      const scenario = drill ? createDrill(drill, slot.current.config, slot.side, episodeSeed) : undefined;
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

function payload(results, ended) {
  const { envs, agents, obsSize, actCount } = layout;
  const pad = (bytes) => (bytes + 3) & ~3;
  const sizes = {
    obs: envs * agents * obsSize * 4, mask: pad(envs * agents * actCount), agentMask: pad(envs * agents),
    reward: envs * 4, done: pad(envs), score: envs * 8, steps: envs * 4,
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
  return payload(slots.map((slot) => slot.last), slots.map(() => ({ reward: 0, done: false, score: [0, 0], steps: 0 })));
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
      ended.push({ reward: 0, done: true, score: [0, 0], steps: slot.steps, failed: true });
      begin(slot);
      return slot.last;
    }
    slot.steps += 1;
    const done = result.terminated || result.truncated;
    ended.push({ reward: result.reward, done, score: result.info.score, steps: slot.steps });
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
