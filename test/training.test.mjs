import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ACT_NONE, ACT_PASS_BASE, ACT_SHOOT_CENTER, ACT_TACKLE, ACTION_FIELDS, createArcadeConfig,
  createDrill, createElevenAsideConfig, DRILL_KINDS, SoccerTrainingEnv,
} from '../dist/index.js';

function shortConfig(base = createArcadeConfig()) {
  base.periodTicks = [120, 120];
  Object.assign(base.rules, { restartPlacementTicks: 2, restartInputTicks: 12, goalPauseTicks: 3, halftimeTicks: 3 });
  return base;
}

function liveScenario(config, ball = { x: 0, y: 0 }, owner = 'home-3') {
  return {
    version: 1, id: 'training-live', period: 1, phase: { kind: 'live' },
    players: config.teams.flatMap((team, index) => team.players.map((player) => ({
      playerId: player.id,
      position: { x: player.formation.x * (index ? -1 : 1), y: player.formation.y * (index ? -1 : 1) },
      facingRad: index ? Math.PI : 0,
    }))),
    ball: {
      position: { x: ball.x, y: ball.y, z: config.ball.radiusM }, velocity: { x: 0, y: 0, z: 0 },
      lastTouchPlayerId: owner, control: { kind: 'free' },
    },
  };
}

const idle = (env) => new Array(env.agentCount * ACTION_FIELDS).fill(0);
const row = (env, result, agent) => result.observation.subarray(agent * env.layout.size, (agent + 1) * env.layout.size);
const mask = (env, result, agent) => result.actionMask.subarray(agent * env.actionSpace.actCount, (agent + 1) * env.actionSpace.actCount);

test('observations have the documented size, finite values and sensible scale', () => {
  const env = new SoccerTrainingEnv({ config: shortConfig() });
  const result = env.reset(3, liveScenario(env.config ?? shortConfig()));
  assert.equal(result.observation.length, env.agentCount * env.layout.size);
  assert.equal(env.layout.size, 63 + 21 * 5 + 23 + 5);
  for (const value of result.observation) assert.ok(Number.isFinite(value) && Math.abs(value) <= 5, `value ${value}`);
  assert.deepEqual([...result.agentMask], [1, 1, 1, 1, 1]);
});

test('both teams see themselves attacking toward positive x', () => {
  const config = shortConfig();
  const home = new SoccerTrainingEnv({ config, learnerTeamId: 'home' });
  const away = new SoccerTrainingEnv({ config, learnerTeamId: 'away' });
  const scenario = liveScenario(config);
  const a = home.reset(5, scenario);
  const b = away.reset(5, scenario);
  // Roster slot 0 is the keeper: it stands near its own goal, which is negative x in both frames.
  assert.ok(row(home, a, 0)[0] < -0.8);
  assert.ok(row(away, b, 0)[0] < -0.8);
  assert.deepEqual([...row(home, a, 0).subarray(0, 2)], [...row(away, b, 0).subarray(0, 2)]);
});

test('padded rosters mask absent agents and slots', () => {
  const env = new SoccerTrainingEnv({ config: shortConfig(), maxPlayersPerTeam: 11 });
  const result = env.reset(1, liveScenario(shortConfig()));
  assert.equal(result.agentMask.reduce((sum, value) => sum + value, 0), 5);
  assert.ok(row(env, result, 7).every((value) => value === 0));
  assert.deepEqual([...mask(env, result, 7)].slice(1), new Array(env.actionSpace.actCount - 1).fill(0));
});

test('only the player on the ball may kick, and passing skips the passer', () => {
  const config = shortConfig();
  const env = new SoccerTrainingEnv({ config });
  const start = liveScenario(config);
  const carrier = start.players.find(({ playerId }) => playerId === 'home-3');
  start.ball.position = { x: carrier.position.x + 0.4, y: carrier.position.y, z: config.ball.radiusM };
  const result = env.reset(9, start);
  const slot = config.teams[0].players.findIndex(({ id }) => id === 'home-3');
  const carrierMask = mask(env, result, slot);
  assert.equal(carrierMask[ACT_SHOOT_CENTER], 1);
  assert.equal(carrierMask[ACT_PASS_BASE + slot], 0);
  assert.equal(carrierMask[ACT_PASS_BASE + 0], 1);
  const other = mask(env, result, 1);
  assert.equal(other[ACT_SHOOT_CENTER], 0);
  assert.equal(other[ACT_TACKLE], 0);
  assert.equal(mask(env, result, 0)[ACT_NONE], 1);
});

test('a pass action kicks the ball toward the chosen teammate', () => {
  const config = shortConfig();
  const env = new SoccerTrainingEnv({ config });
  const start = liveScenario(config);
  const carrier = start.players.find(({ playerId }) => playerId === 'home-3');
  start.ball.position = { x: carrier.position.x + 0.4, y: carrier.position.y, z: config.ball.radiusM };
  const first = env.reset(9, start);
  const slot = config.teams[0].players.findIndex(({ id }) => id === 'home-3');
  const target = config.teams[0].players.findIndex(({ id }) => id === 'home-2');
  assert.equal(mask(env, first, slot)[ACT_PASS_BASE + target], 1);
  const actions = idle(env);
  actions[slot * ACTION_FIELDS + 2] = ACT_PASS_BASE + target;
  let result = env.step(actions);
  assert.equal(result.info.illegalActions, 0);
  for (let i = 0; i < 3; i += 1) result = env.step(idle(env));
  const receiver = start.players.find(({ playerId }) => playerId === 'home-2');
  const snapshot = env.observation();
  const speed = Math.hypot(snapshot.ball.velocity.x, snapshot.ball.velocity.y);
  assert.ok(speed > 2, `ball speed ${speed}`);
  const toward = (receiver.position.x - snapshot.ball.position.x) * snapshot.ball.velocity.x +
    (receiver.position.y - snapshot.ball.position.y) * snapshot.ball.velocity.y;
  assert.ok(toward > 0);
});

test('illegal acts are replaced by none and counted', () => {
  const config = shortConfig();
  const env = new SoccerTrainingEnv({ config });
  env.reset(2, liveScenario(config));
  const actions = idle(env);
  actions[2] = ACT_SHOOT_CENTER; // The keeper is nowhere near the ball.
  const result = env.step(actions);
  assert.equal(result.info.illegalActions, 1);
});

test('invalid action values are rejected', () => {
  const env = new SoccerTrainingEnv({ config: shortConfig() });
  env.reset(2);
  assert.throws(() => env.step([0, 0]), /Expected/);
  const actions = idle(env);
  actions[0] = 99;
  assert.throws(() => env.step(actions), /move/);
});

test('identical seeds and actions reproduce the match', () => {
  const play = () => {
    const env = new SoccerTrainingEnv({ config: shortConfig() });
    let result = env.reset(11);
    let total = 0;
    for (let i = 0; i < 40 && !result.terminated; i += 1) {
      const actions = idle(env);
      for (let agent = 0; agent < env.agentCount; agent += 1) {
        actions[agent * ACTION_FIELDS] = 1 + ((i + agent) % 8);
        actions[agent * ACTION_FIELDS + 1] = i % 2;
      }
      result = env.step(actions);
      total += result.reward;
    }
    return { hash: env.hashState(), total, tick: result.info.tick };
  };
  assert.deepEqual(play(), play());
});

test('a match terminates, and maxSteps truncates', () => {
  const env = new SoccerTrainingEnv({ config: shortConfig() });
  let result = env.reset(4);
  let steps = 0;
  while (!result.terminated) {
    result = env.step(idle(env));
    steps += 1;
    assert.ok(steps < 200);
  }
  assert.throws(() => env.step(idle(env)), /Reset/);

  const limited = new SoccerTrainingEnv({ config: shortConfig(), maxSteps: 3 });
  limited.reset(4);
  limited.step(idle(limited));
  limited.step(idle(limited));
  const last = limited.step(idle(limited));
  assert.equal(last.truncated, true);
  assert.equal(last.terminated, false);
});

test('scoring pays the goal reward with the learner as the reference side', () => {
  const config = shortConfig();
  const env = new SoccerTrainingEnv({ config, reward: { ballProgress: 0, possession: 0 } });
  const start = liveScenario(config, { x: 26, y: 0 }, 'home-4');
  // Put the striker on the ball in front of an open goal, with the opposing keeper moved aside.
  for (const player of start.players) {
    if (player.playerId === 'home-4') player.position = { x: 25.6, y: 0 };
    if (player.playerId === 'away-keeper') player.position = { x: 27, y: 12 };
  }
  const first = env.reset(6, start);
  const slot = config.teams[0].players.findIndex(({ id }) => id === 'home-4');
  assert.equal(mask(env, first, slot)[ACT_SHOOT_CENTER], 1);
  const actions = idle(env);
  actions[slot * ACTION_FIELDS + 2] = ACT_SHOOT_CENTER;
  let total = 0;
  let result = env.step(actions);
  total += result.reward;
  for (let i = 0; i < 6 && total === 0; i += 1) {
    result = env.step(idle(env));
    total += result.reward;
  }
  assert.equal(total, 1);
  assert.deepEqual(result.info.score, [1, 0]);
});

test('eleven-a-side matches work with a padded policy interface', () => {
  const env = new SoccerTrainingEnv({ config: shortConfig(createElevenAsideConfig()) });
  const result = env.reset(8);
  assert.equal(env.agentCount, 11);
  assert.equal(env.layout.size, 63 + 21 * 11 + 23 + 11);
  assert.equal(result.agentMask.reduce((sum, value) => sum + value, 0), 11);
  env.step(idle(env));
});

test('players that all run into a wall or corner pile up without crashing the physics', () => {
  // Each case drove the whole team the same way and used to hit the physics event limit:
  // contacts between players pinned against a wall passed ever smaller impulses back and forth.
  for (const [move, side, seed] of [[6, 'away', 6], [4, 'home', 4], [4, 'away', 4]]) {
    const config = shortConfig();
    config.periodTicks = [1800, 1800];
    const env = new SoccerTrainingEnv({ config, learnerTeamId: side, maxSteps: 220 });
    let result = env.reset(seed);
    while (!result.terminated && !result.truncated) {
      const actions = new Array(env.agentCount * ACTION_FIELDS).fill(0);
      for (let agent = 0; agent < env.agentCount; agent += 1) {
        actions[agent * ACTION_FIELDS] = move;
        actions[agent * ACTION_FIELDS + 1] = 1;
      }
      result = env.step(actions);
    }
    assert.ok(result.info.tick > 1000, `move ${move} as ${side} stopped at tick ${result.info.tick}`);
  }
});

test('drills build valid situations that play out the same from either side', () => {
  const play = (kind, side) => {
    const config = createArcadeConfig();
    const env = new SoccerTrainingEnv({ config, learnerTeamId: side, endOnDeadBall: true, maxSteps: 40 });
    const outcomes = [];
    for (let seed = 1; seed <= 6; seed += 1) {
      let result = env.reset(seed, createDrill(kind, config, side === 'home' ? 0 : 1, seed));
      const [startHash, startBall] = [env.hashState(), env.observation().ball.position];
      assert.ok(Math.abs(startBall.x) < config.pitch.lengthM / 2);
      while (!result.terminated && !result.truncated) {
        const actions = new Array(env.agentCount * ACTION_FIELDS).fill(0);
        actions[(env.agentCount - 1) * ACTION_FIELDS] = 1;
        result = env.step(actions);
      }
      outcomes.push([result.info.step, result.info.score[0], result.reward.toFixed(4)]);
      assert.notEqual(env.hashState(), startHash);
    }
    return outcomes;
  };
  // The pass drill starts with a defender in contact range, and the engine breaks contact ties by player
  // order, which differs between the sides. Its rewards can differ by rounding, so compare steps and scores.
  const coarse = (outcomes) => outcomes.map(([steps, goals]) => [steps, goals]);
  for (const kind of DRILL_KINDS) {
    const [home, away] = [play(kind, 'home'), play(kind, 'away')];
    const loose = kind !== 'shoot' && kind !== 'attack';
    // The give-and-go starts with a presser in contact range, so its length can differ by side; it must still play out.
    if (kind !== 'giveandgo') assert.deepEqual(loose ? coarse(home) : home, loose ? coarse(away) : away);
  }
});

test('an episode can end on a goal or when the ball goes out', () => {
  const config = shortConfig();
  const env = new SoccerTrainingEnv({ config, endOnDeadBall: true });
  const start = liveScenario(config, { x: 26, y: 0 }, 'home-4');
  for (const player of start.players) {
    if (player.playerId === 'home-4') player.position = { x: 25.6, y: 0 };
    if (player.playerId === 'away-keeper') player.position = { x: 27, y: 12 };
  }
  const first = env.reset(6, start);
  const slot = config.teams[0].players.findIndex(({ id }) => id === 'home-4');
  const actions = idle(env);
  actions[slot * ACTION_FIELDS + 2] = ACT_SHOOT_CENTER;
  let result = env.step(actions);
  for (let i = 0; i < 8 && !result.terminated; i += 1) result = env.step(idle(env));
  assert.equal(result.terminated, true);
  assert.deepEqual(result.info.score, [1, 0]);
  assert.ok(first.agentMask[0]);
});

test('drills never place players on top of each other, whatever the seed', () => {
  const config = createArcadeConfig();
  const env = new SoccerTrainingEnv({ config, endOnDeadBall: true });
  for (const kind of DRILL_KINDS) {
    for (let seed = 1; seed <= 400; seed += 1) {
      // Scenario validation rejects overlapping bodies or an illegal ball, so a successful reset is the check.
      env.reset(seed, createDrill(kind, config, seed % 2 === 0 ? 0 : 1, seed));
    }
  }
});
