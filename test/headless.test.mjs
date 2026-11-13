import assert from 'node:assert/strict';
import test from 'node:test';
import {
  benchmarkMatches, createArcadeConfig, evaluateMatch, TeamAIRegistry, verifyReplay,
} from '../dist/index.js';

function shortMatch() {
  const config = createArcadeConfig();
  config.periodTicks = [48, 48];
  Object.assign(config.rules, {
    restartPlacementTicks: 2,
    restartInputTicks: 12,
    goalPauseTicks: 3,
    halftimeTicks: 3,
  });
  return config;
}

function liveScenario(config) {
  return {
    version: 1,
    id: 'headless-live',
    period: 1,
    phase: { kind: 'live' },
    players: config.teams.flatMap((team, index) => team.players.map((player) => ({
      playerId: player.id,
      position: { x: player.formation.x * (index ? -1 : 1), y: player.formation.y },
      facingRad: index ? Math.PI : 0,
    }))),
    ball: {
      position: { x: 0, y: 0, z: config.ball.radiusM },
      velocity: { x: 0, y: 0, z: 0 },
      lastTouchPlayerId: 'home-1',
      control: { kind: 'free' },
    },
  };
}

function replaySummary({ status, ticks, score, finalHash }) {
  return { status, ticks, score, finalHash };
}

function pluginFixture(methods) {
  const pluginId = 'evaluation-test';
  const version = 'test-1';
  const registry = new TeamAIRegistry().register(pluginId, () => ({
    id: pluginId,
    version,
    reset() {},
    decide() { return []; },
    dispose() {},
    ...methods,
  }));
  const teams = { home: { pluginId }, away: { pluginId } };
  return { registry, teams, pluginId, version };
}

test('seeded AI evaluations are independent and serialize to verifiable full-match replays', () => {
  const config = shortMatch();
  const options = { config, seed: 17, recordReplay: true };
  const first = evaluateMatch(options);
  const other = evaluateMatch({ ...options, seed: 19 });
  const repeat = evaluateMatch(options);

  assert.equal(first.status, 'finished');
  assert.equal(other.status, 'finished');
  assert.notEqual(first.finalHash, other.finalHash);
  assert.deepEqual(repeat, first);
  assert.equal(first.metadata.seed, 17);
  assert.deepEqual(first.metadata.config, config);
  assert.equal(first.metadata.initial.tick, 0);
  assert.equal(first.metadata.initial.phase.kind, 'kickoffSetup');
  assert.equal(first.replay.frames.length, first.ticks);
  assert.deepEqual(first.replay.frames.map(({ tick }) => tick), Array.from({ length: first.ticks }, (_, tick) => tick));
  assert.deepEqual(verifyReplay(JSON.parse(JSON.stringify(first.replay))), replaySummary(first));
});

test('replay verifies a tick-limited authored scenario and rejects changed action input', () => {
  const config = shortMatch();
  const scenario = liveScenario(config);
  const evaluation = evaluateMatch({ config, scenario, seed: 17, maxTicks: 12, recordReplay: true });
  assert.equal(evaluation.status, 'tickLimit');
  assert.equal(evaluation.ticks, 12);
  assert.deepEqual(evaluation.metadata.scenario, scenario);
  assert.equal(evaluation.metadata.initial.phase.kind, 'live');
  assert.deepEqual(verifyReplay(JSON.parse(JSON.stringify(evaluation.replay))), replaySummary(evaluation));

  const modified = structuredClone(evaluation.replay);
  const lastFrame = modified.frames.at(-1);
  const command = lastFrame.actions.find(({ playerId }) => playerId === 'home-4');
  assert.ok(command);
  command.movement = { x: command.movement.x > 0 ? -1 : 1, y: 0 };
  assert.throws(() => verifyReplay(modified), /final state hash/i);
});

test('replay rejects a missing frame, incompatible version, or inconsistent initial state', () => {
  const { replay } = evaluateMatch({ config: shortMatch(), seed: 17, recordReplay: true });
  const truncated = { ...replay, frames: replay.frames.slice(0, -1) };
  assert.throws(() => verifyReplay(truncated), /tick|frame/i);
  const incompatible = { ...replay, header: { ...replay.header, formatVersion: 999 } };
  assert.throws(() => verifyReplay(incompatible), /format|version/i);
  const changedSeed = { ...replay, header: { ...replay.header, seed: 18 } };
  assert.throws(() => verifyReplay(changedSeed), /initial state hash/i);
});

test('a replay preserves raw commands and verifies without the recorded controller', () => {
  const config = shortMatch();
  const { registry, teams } = pluginFixture({
    decide(_world, assignment) {
      return assignment.playerIds.map((playerId) => ({ playerId, movement: { x: 2, y: -3 }, sprint: true }));
    },
  });
  const evaluation = evaluateMatch({ config, seed: 17, registry, teams, recordReplay: true });
  const recording = JSON.parse(JSON.stringify(evaluation.replay));
  assert.deepEqual(recording.frames[0].actions[0].movement, { x: 2, y: -3 });
  assert.deepEqual(verifyReplay(recording), replaySummary(evaluation));
});

test('controller failures invalidate the match and identify the affected plugin', () => {
  let disposed = 0;
  const { pluginId, registry, teams } = pluginFixture({
    decide() { throw new Error('decision failed deliberately'); },
    dispose() { disposed += 1; },
  });
  assert.throws(() => evaluateMatch({ config: shortMatch(), seed: 17, registry, teams }), (error) => {
    assert.match(error.message, new RegExp(pluginId));
    assert.match(error.message, /decision failed deliberately/);
    return true;
  });
  assert.equal(disposed, 2);
});

test('throughput measures completed runs separately from warmup', () => {
  const resets = [];
  const { pluginId, version, registry, teams } = pluginFixture({
    reset(context) { resets.push(context.teamId); },
  });
  const config = shortMatch();
  const evaluation = evaluateMatch({ config, seed: 17, registry, teams });
  assert.equal(evaluation.replay, undefined);
  assert.equal(evaluation.metadata.controllers.length, 2);
  assert.ok(evaluation.metadata.controllers.every((controller) =>
    controller.pluginId === pluginId && controller.pluginVersion === version));
  resets.length = 0;

  const measured = benchmarkMatches({ config, seed: 17, matches: 2, warmup: 1, registry, teams });
  assert.equal(resets.length, 6);
  assert.equal(measured.attemptedMatches, 2);
  assert.equal(measured.completedMatches, 2);
  assert.equal(measured.truncatedMatches, 0);
  assert.deepEqual(measured.runs.map(({ seed }) => seed), [17, 18]);
  assert.equal(measured.ticks, measured.runs.reduce((total, run) => total + run.ticks, 0));
  assert.ok(measured.elapsedSeconds > 0);
  assert.ok(Math.abs(measured.ticksPerSecond * measured.elapsedSeconds - measured.ticks) < 1e-8);
  assert.ok(Math.abs(measured.matchesPerSecond * measured.elapsedSeconds - measured.completedMatches) < 1e-8);
  for (const seconds of [measured.simulationSeconds, measured.controllerSeconds, measured.coordinatorSeconds]) {
    assert.ok(Number.isFinite(seconds) && seconds >= 0);
  }
});

test('a tick-limited benchmark includes throughput without counting a completed match', () => {
  const measured = benchmarkMatches({ config: shortMatch(), seed: 17, matches: 2, warmup: 0, maxTicks: 5 });
  assert.equal(measured.attemptedMatches, 2);
  assert.equal(measured.completedMatches, 0);
  assert.equal(measured.truncatedMatches, 2);
  assert.equal(measured.ticks, 10);
  assert.equal(measured.matchesPerSecond, 0);
  assert.ok(measured.ticksPerSecond > 0);
});

test('a pile of players pressed against the end line settles instead of crashing the physics', () => {
  // Seed 100050 jams home-4, away-keeper and away-1 in a row against the end line at tick 2959.
  // Before the settle fallback the contacts traded zero-time impulses until the event limit threw.
  const result = evaluateMatch({
    seed: 100050,
    teams: { home: { pluginId: 'baseline' }, away: { pluginId: 'baseline' } },
    maxTicks: 3600,
  });
  assert.ok(result.ticks > 2959);
});
