import assert from 'node:assert/strict';
import test from 'node:test';
import { createArcadeConfig, SoccerEngine } from '../dist/index.js';

function createTestConfig() {
  const config = createArcadeConfig();
  config.rules.randomKickoff = false; // most tests rely on the home team kicking off first
  config.ball.maxSpeedMps = 120;
  config.rules.restartPlacementTicks = 2;
  config.rules.restartInputTicks = 3;
  config.rules.goalPauseTicks = 3;
  config.rules.halftimeTicks = 3;
  config.periodTicks = [180, 180];
  return config;
}

function liveScenario(config) {
  return {
    version: 1,
    id: 'engine-test',
    period: 1,
    phase: { kind: 'live' },
    players: config.teams.flatMap((team, teamIndex) =>
      team.players.map((player, index) => ({
        playerId: player.id,
        position: {
          x: (teamIndex === 0 ? -1 : 1) * (config.pitch.lengthM * 0.2 + index),
          y: (index - 2) * config.pitch.widthM * 0.12,
        },
        facingRad: teamIndex === 0 ? 0 : Math.PI,
      })),
    ),
    ball: {
      position: { x: 0, y: 0, z: config.ball.radiusM },
      velocity: { x: 0, y: 0, z: 0 },
      lastTouchPlayerId: 'home-1',
      control: { kind: 'free' },
    },
  };
}

function step(engine, actions = []) {
  return engine.step({ tick: engine.snapshot().tick, actions });
}

function findPlayer(engine, playerId) {
  const player = engine.snapshot().players.find((candidate) => candidate.playerId === playerId);
  assert.ok(player, `expected ${playerId} in the snapshot`);
  return player;
}

function movement(playerId, x, y = 0) {
  return { playerId, movement: { x, y }, sprint: false };
}

test('scenario reset preserves authored state, applies defaults, and copies inputs and observations', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  scenario.period = 2;
  scenario.remainingPeriodTicks = 91;
  scenario.score = [1, 2];
  const authored = scenario.players.find((player) => player.playerId === 'home-1');
  authored.velocity = { x: 1, y: -0.5 };
  authored.stamina = 0.75;

  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);
  const snapshot = engine.snapshot();
  assert.equal(snapshot.tick, 0);
  assert.equal(snapshot.period, 2);
  assert.equal(snapshot.remainingPeriodTicks, 91);
  assert.deepEqual(snapshot.score, [1, 2]);
  assert.deepEqual(snapshot.attackDirections, [-1, 1]);
  assert.deepEqual(snapshot.ball.position, scenario.ball.position);
  const player = findPlayer(engine, 'home-1');
  assert.deepEqual(player.position, authored.position);
  assert.deepEqual(player.velocity, authored.velocity);
  assert.equal(player.stamina, 0.75);
  assert.equal(player.balance, 1);
  assert.deepEqual(findPlayer(engine, 'away-1').velocity, { x: 0, y: 0 });
  assert.deepEqual(engine.observe('home').events, []);

  const hash = engine.hashState();
  authored.position.x = 999;
  config.pitch.lengthM = 999;
  Reflect.set(snapshot.ball.position, 'x', 999);
  const observation = engine.observe('away');
  Reflect.set(observation.players[0].definition.abilities.physical, 'maxSpeedMps', 999);
  Reflect.set(observation.pitch, 'lengthM', 999);
  assert.equal(engine.hashState(), hash);
  assert.deepEqual(engine.snapshot().ball.position, { x: 0, y: 0, z: config.ball.radiusM });
  assert.deepEqual(findPlayer(engine, 'home-1').position, player.position);
  assert.notEqual(engine.observe('away').pitch.lengthM, 999);
  assert.notEqual(engine.observe('away').players[0].definition.abilities.physical.maxSpeedMps, 999);
});

test('invalid config and scenario resets preserve the running match', async (t) => {
  const config = createTestConfig();
  const engine = new SoccerEngine();
  engine.reset(config, 17, liveScenario(config));
  step(engine, [movement('home-1', 1)]);
  const snapshot = engine.snapshot();
  const hash = engine.hashState();

  const attempts = [
    ['invalid seed', (candidateConfig, candidateScenario) => engine.reset(candidateConfig, -1, candidateScenario)],
    ['invalid pitch', (candidateConfig, candidateScenario) => {
      candidateConfig.pitch.lengthM = 0;
      engine.reset(candidateConfig, 18, candidateScenario);
    }],
    ['unknown player', (candidateConfig, candidateScenario) => {
      candidateScenario.players[0].playerId = 'missing';
      engine.reset(candidateConfig, 18, candidateScenario);
    }],
    ['overlap', (candidateConfig, candidateScenario) => {
      candidateScenario.players[0].position = candidateScenario.players[1].position;
      engine.reset(candidateConfig, 18, candidateScenario);
    }],
    ['out of reach dribble', (candidateConfig, candidateScenario) => {
      candidateScenario.ball.control = { kind: 'dribble', playerId: 'home-1' };
      engine.reset(candidateConfig, 18, candidateScenario);
    }],
    ['invalid remaining period time', (candidateConfig, candidateScenario) => {
      candidateScenario.remainingPeriodTicks = 0;
      engine.reset(candidateConfig, 18, candidateScenario);
    }],
  ];

  for (const [name, attempt] of attempts) {
    await t.test(name, () => {
      assert.throws(() => attempt(createTestConfig(), liveScenario(config)));
      assert.deepEqual(engine.snapshot(), snapshot);
      assert.equal(engine.hashState(), hash);
    });
  }
});

test('malformed and stale action frames leave the state untouched', () => {
  const config = createTestConfig();
  const engine = new SoccerEngine();
  engine.reset(config, 17, liveScenario(config));
  const valid = movement('home-1', 1);
  const initialHash = engine.hashState();
  for (const frame of [
    { tick: 0, actions: [valid, valid] },
    { tick: 0, actions: [valid, movement('away-1', NaN)] },
    { tick: 0, actions: [valid, movement('missing', 1)] },
    { tick: 1, actions: [valid] },
  ]) {
    assert.throws(() => engine.step(frame));
    assert.equal(engine.hashState(), initialHash);
    assert.equal(engine.snapshot().tick, 0);
  }
  assert.equal(engine.step({ tick: 0, actions: [valid] }).tick, 1);
  const advancedHash = engine.hashState();
  assert.throws(() => engine.step({ tick: 0, actions: [valid] }));
  assert.equal(engine.hashState(), advancedHash);
});

test('movement accelerates, respects the speed cap, and brakes under neutral input', () => {
  const config = createTestConfig();
  const engine = new SoccerEngine();
  engine.reset(config, 17, liveScenario(config));
  const playerId = 'home-1';
  const start = findPlayer(engine, playerId);
  const { maxSpeedMps, accelerationMps2 } = start.definition.abilities.physical;
  step(engine, [movement(playerId, 100)]);
  const afterOneTick = findPlayer(engine, playerId);
  assert.ok(afterOneTick.position.x > start.position.x);
  assert.ok(afterOneTick.velocity.x > 0);
  assert.ok(afterOneTick.velocity.x <= accelerationMps2 / 60 + 1e-9);
  for (let tick = 0; tick < 25; tick += 1) {
    step(engine, [movement(playerId, 100)]);
    const { velocity } = findPlayer(engine, playerId);
    assert.ok(Math.hypot(velocity.x, velocity.y) <= maxSpeedMps + 1e-9);
  }
  const moving = findPlayer(engine, playerId);
  step(engine);
  const braking = findPlayer(engine, playerId);
  assert.ok(Math.abs(braking.velocity.x) < Math.abs(moving.velocity.x));
});

test('a goal requires the whole ball to cross and is recorded once', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  const goalLine = config.pitch.lengthM / 2;
  const radius = config.ball.radiusM;
  scenario.ball.position = { x: goalLine - 1.2 * radius, y: 0, z: config.pitch.goalHeightM / 2 };
  scenario.ball.velocity = { x: 1.5 * radius * 60, y: 0, z: 0 };
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);

  const partialCrossing = step(engine);
  assert.ok(engine.snapshot().ball.position.x > goalLine);
  assert.equal(partialCrossing.events.some((event) => event.kind === 'goal'), false);
  assert.deepEqual(engine.snapshot().score, [0, 0]);

  const crossing = step(engine);
  const goals = crossing.events.filter((event) => event.kind === 'goal');
  assert.equal(goals.length, 1);
  assert.equal(goals[0].teamId, 'home');
  assert.ok(goals[0].position.x >= goalLine + radius - 1e-8);
  assert.deepEqual(engine.snapshot().score, [1, 0]);
  assert.equal(engine.snapshot().phase.kind, 'goalPause');
  assert.equal(step(engine).events.some((event) => event.kind === 'goal'), false);
  assert.deepEqual(engine.snapshot().score, [1, 0]);
});

test('a swept post contact can prevent a fast ball from crossing the goal line', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  const goalLine = config.pitch.lengthM / 2;
  scenario.ball.position = {
    x: goalLine - 0.5,
    y: config.pitch.goalWidthM / 2,
    z: config.pitch.goalHeightM / 2,
  };
  scenario.ball.velocity.x = 100;
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);

  const result = step(engine);
  assert.ok(engine.snapshot().ball.velocity.x < 0);
  assert.ok(engine.snapshot().ball.position.x < goalLine);
  assert.equal(result.events.some((event) => event.kind === 'goal' || event.kind === 'ballExit'), false);
  assert.deepEqual(engine.snapshot().score, [0, 0]);
});

test('a goalkeeper body contact before the crossing keeps a fast ball out', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  const goalLine = config.pitch.lengthM / 2;
  const keeper = scenario.players.find((player) => player.playerId === 'away-keeper');
  keeper.position = { x: goalLine - 0.4, y: 0 };
  scenario.ball.position = { x: goalLine - 1.1, y: 0, z: 0.65 };
  scenario.ball.velocity.x = 100;
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);

  const result = step(engine);
  assert.equal(result.events.some((event) => event.kind === 'goal'), false);
  assert.deepEqual(engine.snapshot().score, [0, 0]);
  assert.equal(engine.snapshot().ball.lastTouchPlayerId, 'away-keeper');
  assert.ok(engine.snapshot().ball.velocity.x < 0);
});

test('a whole-ball goal takes precedence over a later post contact in the same tick', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  const radius = config.ball.radiusM;
  const goalLine = config.pitch.lengthM / 2;
  const startX = goalLine - 1.2 * radius;
  const velocity = { x: 18, y: 80, z: 0 };
  const crossingTime = (goalLine + radius - startX) / velocity.x;
  const crossingY = config.pitch.goalWidthM / 2 - radius - 0.005;
  // The ball fits inside the opening at the crossing and then heads into the post.
  scenario.ball.position = {
    x: startX,
    y: crossingY - velocity.y * crossingTime,
    z: config.pitch.goalHeightM / 2,
  };
  scenario.ball.velocity = velocity;
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);

  const result = step(engine);
  const goals = result.events.filter((event) => event.kind === 'goal');
  assert.equal(goals.length, 1);
  assert.ok(Math.abs(goals[0].position.y - crossingY) < 1e-6);
  assert.deepEqual(engine.snapshot().score, [1, 0]);
  assert.equal(result.events.some((event) => event.kind === 'ballExit'), false);
});

test('a fast sideline exit awards a throw-in against the last-touch team', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  scenario.ball.position = { x: 0, y: config.pitch.widthM / 2 - 0.5, z: 0.5 };
  scenario.ball.velocity.y = 100;
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);

  const result = step(engine);
  const exits = result.events.filter((event) => event.kind === 'ballExit');
  assert.equal(exits.length, 1);
  assert.equal(exits[0].restart, 'throwIn');
  assert.equal(exits[0].teamId, 'away');
  assert.ok(exits[0].position.y >= config.pitch.widthM / 2 + config.ball.radiusM - 1e-8);
  assert.equal(engine.snapshot().phase.kind, 'restartSetup');
  assert.equal(engine.snapshot().remainingPeriodTicks, config.periodTicks[0] - 1);
  step(engine);
  assert.equal(engine.snapshot().remainingPeriodTicks, config.periodTicks[0] - 1);

  let automaticThrow;
  for (let tick = 0; tick < 30 && automaticThrow === undefined; tick += 1) {
    automaticThrow = step(engine).events.find((event) => event.kind === 'actionContact');
  }
  assert.equal(automaticThrow?.outcome, 'thrown');
  assert.equal(automaticThrow?.action.source, 'rule');
  assert.equal(engine.snapshot().phase.kind, 'live');
});

test('a goal on the last live tick counts before the transition to halftime', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  scenario.remainingPeriodTicks = 1;
  scenario.ball.position = { x: config.pitch.lengthM / 2 - 0.5, y: 0, z: 0.5 };
  scenario.ball.velocity.x = 100;
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);

  const result = step(engine);
  assert.equal(result.events.filter((event) => event.kind === 'goal').length, 1);
  const transitions = result.events.filter((event) => event.kind === 'phaseChanged');
  assert.deepEqual(transitions.map((event) => event.phase.kind), ['halftime']);
  assert.equal(engine.snapshot().remainingPeriodTicks, 0);
  assert.deepEqual(engine.snapshot().score, [1, 0]);
});

test('neutral input advances both kickoffs and periods through full time', () => {
  const config = createTestConfig();
  config.periodTicks = [3, 3];
  const engine = new SoccerEngine();
  engine.reset(config, 17);
  const initial = engine.snapshot();
  assert.equal(initial.phase.kind, 'kickoffSetup');
  assert.equal(initial.phase.teamId, 'home');
  assert.deepEqual(initial.attackDirections, [1, -1]);

  const observedPhases = [];
  let result;
  for (let tick = 0; tick < 100; tick += 1) {
    result = step(engine);
    observedPhases.push(...result.events.filter((event) => event.kind === 'phaseChanged').map((event) => event.phase));
    if (result.status === 'finished') break;
  }
  assert.equal(result?.status, 'finished');
  assert.deepEqual(observedPhases.map((phase) => phase.kind), ['live', 'halftime', 'kickoffSetup', 'live', 'fullTime']);
  assert.equal(observedPhases[2].teamId, 'away');
  assert.deepEqual(engine.snapshot().attackDirections, [-1, 1]);
  assert.equal(engine.snapshot().remainingPeriodTicks, 0);
  const finalHash = engine.hashState();
  assert.throws(() => step(engine));
  assert.equal(engine.hashState(), finalHash);
});

test('recorded frames replay the same events and state hashes after a seeded reset', () => {
  const config = createTestConfig();
  const engine = new SoccerEngine();
  const frames = Array.from({ length: 60 }, (_, tick) => ({
    tick,
    actions: [movement('home-2', tick < 35 ? 1 : -1, 0.25)],
  }));
  function run() {
    engine.reset(config, 0x1badb002);
    return frames.map((frame) => ({ result: engine.step(frame), hash: engine.hashState() }));
  }
  const recorded = run();
  assert.ok(recorded.some(({ result }) => result.events.some((event) => event.kind === 'actionContact')));
  assert.ok(new Set(recorded.map(({ hash }) => hash)).size > 1);
  assert.deepEqual(run(), recorded);
});

test('the seed affects launch variation while engine instances remain independent', () => {
  const config = createTestConfig();
  const first = new SoccerEngine();
  const second = new SoccerEngine();
  first.reset(config, 1);
  second.reset(config, 2);
  const untouchedHash = second.hashState();

  for (let tick = 0; tick < 30 && first.snapshot().phase.kind !== 'live'; tick += 1) step(first);
  assert.equal(first.snapshot().phase.kind, 'live');
  assert.equal(second.hashState(), untouchedHash);
  assert.equal(second.snapshot().tick, 0);
  for (let tick = 0; tick < 30 && second.snapshot().phase.kind !== 'live'; tick += 1) step(second);
  assert.equal(second.snapshot().phase.kind, 'live');
  assert.equal(first.snapshot().tick, second.snapshot().tick);
  assert.notDeepEqual(first.snapshot().ball.velocity, second.snapshot().ball.velocity);
});

test('interrupted kicks retain recovery and produce one terminal result', () => {
  const config = createTestConfig();
  const scenario = liveScenario(config);
  const player = scenario.players.find(({ playerId }) => playerId === 'home-1');
  player.position = { x: -0.8, y: 0 };
  scenario.ball.velocity.x = 40;
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);
  const kick = (id) => ({
    ...movement(player.playerId, 0),
    request: { id, kind: 'kick', kickType: 'pass', direction: { x: 1, y: 0 }, power: 1, loft: 0 },
  });

  const initial = step(engine, [kick(1)]);
  assert.equal(initial.actionResults[0]?.status, 'accepted');
  const terminal = step(engine);
  assert.equal(terminal.actionResults[0]?.status, 'interrupted');
  assert.equal(terminal.actionResults[0]?.reason, 'ballUnavailable');
  const busy = step(engine, [kick(2)]);
  assert.equal(busy.actionResults[0]?.reason, 'busy');
  const duplicate = step(engine, [kick(2)]);
  assert.equal(duplicate.diagnostics[0]?.kind, 'duplicateRequest');
  assert.equal(duplicate.actionResults.length, 0);
  for (let index = 0; index < 9; index += 1) {
    const result = step(engine);
    assert.ok(result.actionResults.every(({ action }) => action.requestId !== 1));
  }
  assert.equal(findPlayer(engine, player.playerId).action, undefined);
});

test('a keeper-held scenario automatically releases the ball on its deadline', () => {
  const config = createTestConfig();
  config.rules.keeperHoldTicks = 3;
  const scenario = liveScenario(config);
  const keeper = config.teams[0].players[0];
  const player = scenario.players.find(({ playerId }) => playerId === keeper.id);
  player.position = { x: -config.pitch.lengthM / 2 + 3, y: 0 };
  scenario.ball = {
    position: {
      x: player.position.x + keeper.radiusM + config.ball.radiusM,
      y: 0,
      z: keeper.bodyHeightM * 0.72,
    },
    velocity: { x: 0, y: 0, z: 0 },
    lastTouchPlayerId: keeper.id,
    control: { kind: 'keeperHeld', playerId: keeper.id },
  };
  const engine = new SoccerEngine();
  engine.reset(config, 17, scenario);
  for (let tick = 0; tick < config.rules.keeperHoldTicks; tick += 1) {
    assert.equal(step(engine).actionResults.length, 0);
  }
  const accepted = step(engine).actionResults[0];
  assert.equal(accepted?.action.source, 'rule');
  assert.equal(accepted?.status, 'accepted');
  let contact;
  for (let tick = 0; tick < 10 && !contact; tick += 1) {
    contact = step(engine).events.find(({ kind }) => kind === 'actionContact');
  }
  assert.equal(contact?.outcome, 'released');
  assert.deepEqual(engine.snapshot().ball.control, { kind: 'free' });
  assert.equal(engine.snapshot().ball.lastTouchPlayerId, keeper.id);
  assert.equal(engine.snapshot().remainingPeriodTicks, config.periodTicks[0] - engine.snapshot().tick);
});

function kickoffAfterGoal(config, seed) {
  const scenario = liveScenario(config);
  const goalLine = config.pitch.lengthM / 2;
  const radius = config.ball.radiusM;
  scenario.ball.position = { x: goalLine - 1.2 * radius, y: 0, z: config.pitch.goalHeightM / 2 };
  scenario.ball.velocity = { x: 1.5 * radius * 60, y: 0, z: 0 };
  const engine = new SoccerEngine();
  engine.reset(config, seed, scenario);
  step(engine);
  step(engine);
  assert.equal(engine.snapshot().phase.kind, 'goalPause');
  return engine.snapshot().phase.teamId;
}

test('without random kickoffs the opening kickoff goes to the first team and the team that conceded restarts', () => {
  const config = createTestConfig();
  for (let seed = 1; seed <= 12; seed += 1) {
    const engine = new SoccerEngine();
    engine.reset(config, seed);
    assert.equal(engine.snapshot().phase.teamId, 'home');
    // Home scored, so away takes the kickoff.
    assert.equal(kickoffAfterGoal(config, seed), 'away');
  }
});

test('with random kickoffs both teams get the opening kickoff and the one after a goal', () => {
  const config = createTestConfig();
  config.rules.randomKickoff = true;
  const opening = { home: 0, away: 0 };
  const afterGoal = { home: 0, away: 0 };
  for (let seed = 1; seed <= 60; seed += 1) {
    const engine = new SoccerEngine();
    engine.reset(config, seed);
    opening[engine.snapshot().phase.teamId] += 1;
    afterGoal[kickoffAfterGoal(config, seed)] += 1;
  }
  for (const counts of [opening, afterGoal]) {
    assert.ok(counts.home >= 15 && counts.away >= 15, `kickoffs were not shared: ${JSON.stringify(counts)}`);
  }
});

test('random kickoffs are chosen from the match seed, so a seed always gives the same kickoffs', () => {
  const config = createTestConfig();
  config.rules.randomKickoff = true;
  config.periodTicks = [3, 3];
  const play = (seed) => {
    const engine = new SoccerEngine();
    engine.reset(config, seed);
    const kickoffs = [engine.snapshot().phase.teamId];
    for (let tick = 0; tick < 40 && engine.snapshot().phase.kind !== 'fullTime'; tick += 1) {
      step(engine);
      const { phase } = engine.snapshot();
      if (phase.kind === 'kickoffSetup' && kickoffs.length < 2 && engine.snapshot().period === 2) kickoffs.push(phase.teamId);
    }
    return [kickoffs, engine.hashState()];
  };
  assert.deepEqual(play(5), play(5));
  const seconds = new Set([1, 2, 3, 4, 5, 6, 7, 8].map((seed) => play(seed)[0][1]));
  assert.equal(seconds.size, 2, 'the second-half kickoff should vary with the seed');
});
