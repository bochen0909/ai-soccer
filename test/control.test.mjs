import assert from 'node:assert/strict';
import test from 'node:test';
import { createArcadeConfig, MatchSession, TeamAIRegistry } from '../dist/index.js';

const east = { x: 1, y: 0 };
const north = { x: 0, y: 1 };
const neutral = { x: 0, y: 0 };

function fixture() {
  const config = createArcadeConfig();
  config.periodTicks = [300, 300];
  const scenario = {
    id: 'control-test',
    version: 1,
    period: 1,
    phase: { kind: 'live' },
    players: config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
      playerId: player.id,
      position: { x: (teamIndex === 0 ? -1 : 1) * (12 + index), y: (index - 2) * 5 },
      facingRad: teamIndex === 0 ? 0 : Math.PI,
    }))),
    ball: {
      position: { x: 0, y: 0, z: config.ball.radiusM },
      velocity: { x: 0, y: 0, z: 0 },
      lastTouchPlayerId: 'home-1',
      control: { kind: 'free' },
    },
  };

  function player(playerId) {
    return scenario.players.find((entry) => entry.playerId === playerId);
  }

  function definition(playerId) {
    return config.teams.flatMap((team) => team.players).find((entry) => entry.id === playerId);
  }

  return { config, scenario, player, definition };
}

function passFixture() {
  const setup = fixture();
  setup.player('home-1').position = { x: -2.5, y: 0 };
  setup.player('home-2').position = { x: 0, y: 0 };
  setup.player('home-2').facingRad = Math.PI;
  setup.definition('home-1').abilities.technical.passing = 1;
  setup.definition('home-2').abilities.technical.firstTouch = 1;
  setup.scenario.ball.position.x = -1.95;
  setup.scenario.ball.control = { kind: 'dribble', playerId: 'home-1' };
  return setup;
}

function controlled(setup, {
  decide = () => [],
  reset = () => {},
  dispose = () => {},
  decisionIntervalTicks,
  human,
  passIntentionTicks,
} = {}) {
  const registry = new TeamAIRegistry();
  registry.register('test-team', () => {
    let teamId;
    return {
      id: 'test-team',
      version: '1.0.0',
      reset(context) {
        teamId = context.teamId;
        reset(context);
      },
      decide(observation, assignment, context) {
        return decide(teamId, observation, assignment, context);
      },
      dispose() {
        dispose(teamId);
      },
    };
  });
  const session = new MatchSession(registry);
  const settings = { pluginId: 'test-team', decisionIntervalTicks };
  const options = { teams: { home: settings, away: settings }, human, passIntentionTicks };
  session.reset(setup.config, 17, options, setup.scenario);
  return { session, options };
}

function command(result, playerId) {
  const actions = result.frame.actions.filter((action) => action.playerId === playerId);
  assert.equal(actions.length, 1, `exactly one command for ${playerId}`);
  return actions[0];
}

function playerIntent(playerId, movement, request) {
  return { playerId, movement, sprint: false, request };
}

const passRequest = { kind: 'kick', kickType: 'pass', direction: east, power: 0.65, loft: 0 };
const humanPass = { request: passRequest, pass: { receiverId: 'home-2', target: neutral } };

function advanceToTouch(session, playerId) {
  for (let index = 0; index < 35; index += 1) {
    const result = session.step();
    const touch = result.events.find((event) =>
      event.kind === 'ballTouch' && event.playerId === playerId && event.outcome === 'received');
    if (touch) return touch;
  }
  assert.fail(`no reception for ${playerId}`);
}

test('both AI plugins see the same pre-tick world and held movement continues between decisions', () => {
  const calls = [];
  const setups = [];
  const setup = passFixture();
  const { session } = controlled(setup, {
    decisionIntervalTicks: 6,
    reset: (context) => setups.push(context),
    decide(teamId, observation, _assignment, context) {
      calls.push({
        teamId,
        tick: observation.tick,
        positions: observation.players.map(({ playerId, position }) => ({ playerId, position: { ...position } })),
        actionResults: context.actionResults,
      });
      if (teamId === 'home') observation.players[0].position.x = 999;
      return teamId === 'home' && observation.tick === 0
        ? [playerIntent('home-1', neutral, passRequest), playerIntent('home-3', north)]
        : [];
    },
  });
  assert.deepEqual(setups.map(({ teamId }) => teamId).sort(), ['away', 'home']);
  assert.notEqual(setups[0].seed, setups[1].seed);

  const start = session.snapshot();
  const results = Array.from({ length: 7 }, () => session.step());
  assert.equal(calls.length, 4);
  for (const tick of [0, 6]) {
    const world = calls.filter((call) => call.tick === tick);
    assert.deepEqual(world.map(({ teamId }) => teamId).sort(), ['away', 'home']);
    assert.deepEqual(world[0].positions, world[1].positions);
  }
  assert.deepEqual(calls[0].positions, start.players.map(({ playerId, position }) => ({ playerId, position })));

  assert.ok(command(results[0], 'home-1').request);
  const accepted = results[0].actionResults.find(({ status }) => status === 'accepted');
  assert.ok(accepted);
  const homeNextDecision = calls.find(({ teamId, tick }) => teamId === 'home' && tick === 6);
  const awayNextDecision = calls.find(({ teamId, tick }) => teamId === 'away' && tick === 6);
  assert.deepEqual(homeNextDecision.actionResults, [{ ...accepted, decisionTick: 0 }]);
  assert.deepEqual(awayNextDecision.actionResults, []);
  for (const result of results.slice(0, 6)) {
    assert.equal(result.frame.actions.length, start.players.length);
    assert.deepEqual(command(result, 'home-3').movement, north);
  }
  for (const result of results.slice(1)) {
    assert.equal(command(result, 'home-1').request, undefined);
    assert.equal(result.diagnostics.length, 0);
  }
  assert.deepEqual(command(results[6], 'home-3').movement, neutral);
});

test('manual selection refreshes AI assignment and clears the previous human input', () => {
  const homeAssignments = [];
  const { session } = controlled(fixture(), {
    human: { teamId: 'home', playerId: 'home-1' },
    decisionIntervalTicks: 60,
    decide(teamId, observation, assignment) {
      if (teamId === 'away') return [];
      homeAssignments.push({ tick: observation.tick, ...assignment });
      return assignment.playerIds.map((id) => playerIntent(id, north));
    },
  });

  const before = session.step({ movement: east, sprint: true });
  const held = session.step();
  for (const result of [before, held]) {
    assert.deepEqual(command(result, 'home-1').movement, east);
    assert.equal(command(result, 'home-1').sprint, true);
    assert.deepEqual(command(result, 'home-2').movement, north);
  }

  const changed = session.step({ selectPlayerId: 'home-2' });
  assert.equal(session.snapshot().selectedPlayerId, 'home-2');
  assert.deepEqual(command(changed, 'home-1').movement, north);
  assert.deepEqual(command(changed, 'home-2').movement, neutral);
  assert.equal(command(changed, 'home-2').sprint, false);
  assert.deepEqual(homeAssignments.map(({ tick }) => tick), [0, 2]);
  assert.ok(homeAssignments[1].revision > homeAssignments[0].revision);
  assert.ok(homeAssignments[1].playerIds.includes('home-1'));
  assert.ok(!homeAssignments[1].playerIds.includes('home-2'));
});

test('the AI receives a pass proposal immediately and human control switches on the tick after reception', () => {
  const homeDecisions = [];
  const awayDecisions = [];
  const { session } = controlled(passFixture(), {
    human: { teamId: 'home', playerId: 'home-1' },
    decisionIntervalTicks: 60,
    decide(teamId, observation, assignment, context) {
      const entry = { tick: observation.tick, context, assignment, events: observation.events };
      (teamId === 'home' ? homeDecisions : awayDecisions).push(entry);
      return [];
    },
  });

  session.step();
  const result = session.step(humanPass);
  const initiating = result.actionResults.find(({ status }) => status === 'accepted')?.action;
  assert.ok(initiating);
  assert.deepEqual(homeDecisions.map(({ tick }) => tick), [0, 1]);
  const proposal = homeDecisions[1].context.passIntention;
  assert.equal(proposal.status, 'proposed');
  assert.equal(proposal.receiverId, 'home-2');
  assert.deepEqual(proposal.target, neutral);
  assert.deepEqual(proposal.action, initiating);
  assert.ok(awayDecisions.every(({ context }) => !context.passIntention));

  const touch = advanceToTouch(session, 'home-2');
  assert.deepEqual(touch.flightAction, initiating);
  assert.equal(session.snapshot().selectedPlayerId, 'home-1');
  const following = session.step({ movement: east });
  assert.equal(session.snapshot().selectedPlayerId, 'home-2');
  assert.deepEqual(command(following, 'home-2').movement, east);
  assert.deepEqual(command(following, 'home-1').movement, neutral);
  const after = homeDecisions.at(-1);
  assert.ok(after.events.some(({ kind, ordinal, tick }) =>
    kind === 'ballTouch' && ordinal === touch.ordinal && tick === touch.tick));
  assert.ok(!after.context.passIntention);
});

test('an opponent interception cancels the pending handoff', () => {
  const setup = passFixture();
  setup.player('away-1').position = { x: -0.82, y: 0 };
  setup.player('away-1').facingRad = Math.PI;
  setup.definition('away-1').abilities.technical.firstTouch = 1;
  const { session } = controlled(setup, { human: { teamId: 'home', playerId: 'home-1' } });

  const initiating = session.step(humanPass).actionResults.find(({ status }) => status === 'accepted')?.action;
  assert.ok(initiating);
  const touch = advanceToTouch(session, 'away-1');
  assert.deepEqual(touch.flightAction, initiating);
  session.step();
  assert.equal(session.snapshot().selectedPlayerId, 'home-1');
});

test('a rejected pass clears its proposed receiver before the next AI decision', () => {
  const decisions = [];
  const { session } = controlled(fixture(), {
    human: { teamId: 'home', playerId: 'home-1' },
    decisionIntervalTicks: 60,
    decide(teamId, observation, _assignment, context) {
      if (teamId === 'home') decisions.push({ tick: observation.tick, ...context });
      return [];
    },
  });

  const rejected = session.step(humanPass).actionResults.find(({ status }) => status === 'rejected');
  assert.ok(rejected);
  assert.equal(decisions[0].passIntention.status, 'proposed');
  assert.deepEqual(decisions[0].passIntention.action, rejected.action);
  session.step();
  assert.deepEqual(decisions.map(({ tick }) => tick), [0, 1]);
  assert.equal(decisions[1].passIntention, null);
  assert.deepEqual(decisions[1].actionResults, [{ ...rejected, decisionTick: 0 }]);
  assert.equal(session.snapshot().selectedPlayerId, 'home-1');
});

test('a reception after pass intention expiry keeps human control on the passer', () => {
  const { session } = controlled(passFixture(), {
    human: { teamId: 'home', playerId: 'home-1' },
    passIntentionTicks: 3,
  });
  const initiating = session.step(humanPass).actionResults.find(({ status }) => status === 'accepted')?.action;
  assert.ok(initiating);

  const touch = advanceToTouch(session, 'home-2');
  assert.deepEqual(touch.flightAction, initiating);
  assert.ok(touch.tick > 3);
  session.step();
  assert.equal(session.snapshot().selectedPlayerId, 'home-1');
});

test('manual selection takes precedence when a confirmed reception is pending', () => {
  const { session } = controlled(passFixture(), { human: { teamId: 'home', playerId: 'home-1' } });
  session.step(humanPass);
  advanceToTouch(session, 'home-2');
  const selected = session.step({ selectPlayerId: 'home-3', movement: north });
  assert.equal(session.snapshot().selectedPlayerId, 'home-3');
  assert.deepEqual(command(selected, 'home-3').movement, north);
  session.step();
  assert.equal(session.snapshot().selectedPlayerId, 'home-3');
});

test('failed AI decisions discard held commands and leave the other team in control', () => {
  const { session } = controlled(fixture(), {
    decisionIntervalTicks: 1,
    decide(teamId, observation) {
      if (teamId === 'away') return [playerIntent('away-1', east)];
      if (observation.tick === 0) return [playerIntent('home-1', east)];
      if (observation.tick === 1) return [playerIntent('home-1', east), playerIntent('away-1', north)];
      throw new Error('test controller failure');
    },
  });
  assert.deepEqual(command(session.step(), 'home-1').movement, east);
  for (let index = 0; index < 2; index += 1) {
    const result = session.step();
    assert.equal(result.controllerFailures.length, 1);
    assert.equal(result.controllerFailures[0].teamId, 'home');
    assert.equal(result.controllerFailures[0].stage, 'decide');
    assert.deepEqual(command(result, 'home-1').movement, neutral);
    assert.deepEqual(command(result, 'away-1').movement, east);
  }
});

test('reset clears held input, pass state and request counters', () => {
  const setup = passFixture();
  const resets = [];
  const disposals = [];
  const proposals = [];
  const { session, options } = controlled(setup, {
    human: { teamId: 'home', playerId: 'home-1' },
    reset: ({ teamId, seed }) => resets.push({ teamId, seed }),
    dispose: (teamId) => disposals.push(teamId),
    decide(teamId, _observation, _assignment, context) {
      if (teamId === 'home') proposals.push(context.passIntention);
      return [];
    },
  });
  const first = session.step({ ...humanPass, movement: east, sprint: true });
  const firstRequest = command(first, 'home-1').request;
  assert.ok(firstRequest);
  assert.ok(proposals[0]);
  session.step({ selectPlayerId: 'home-3' });

  session.reset(setup.config, 17, options, setup.scenario);
  assert.deepEqual(disposals.sort(), ['away', 'home']);
  assert.deepEqual(resets.slice(2), resets.slice(0, 2));
  const initial = session.snapshot();
  assert.equal(initial.tick, 0);
  assert.equal(initial.selectedPlayerId, 'home-1');
  const cleared = session.step();
  assert.equal(proposals.at(-1), null);
  assert.deepEqual(command(cleared, 'home-1').movement, neutral);
  assert.equal(command(cleared, 'home-1').sprint, false);
  assert.equal(command(cleared, 'home-1').request, undefined);
  const another = session.step(humanPass);
  assert.equal(command(another, 'home-1').request.id, firstRequest.id);
});

test('the default AI plays through full time and reproduces the same seeded match', () => {
  const config = createArcadeConfig();
  config.periodTicks = [300, 300];
  config.rules.restartPlacementTicks = 2;
  config.rules.restartInputTicks = 30;
  config.rules.goalPauseTicks = 3;
  config.rules.halftimeTicks = 3;
  const session = new MatchSession();

  function play() {
    session.reset(config, 17);
    const results = [];
    for (let index = 0; index < 950; index += 1) {
      const result = session.step();
      assert.deepEqual(result.controllerFailures, []);
      results.push(result);
      if (result.status === 'finished') break;
    }
    assert.equal(results.at(-1).status, 'finished');
    const final = session.snapshot();
    assert.equal(final.selectedPlayerId, null);
    assert.equal(final.phase.kind, 'fullTime');
    assert.ok(results.some(({ events }) => events.some(({ kind }) => kind === 'actionContact')));
    assert.ok(results.some(({ frame }) => frame.actions.some(({ movement }) => Math.hypot(movement.x, movement.y) > 0)));
    return { final, results };
  }

  assert.deepEqual(play(), play());
});

test('a controller gets a decision on the exact tick a short restart window opens', () => {
  const setup = fixture();
  setup.config.rules.restartPlacementTicks = 2;
  setup.config.rules.restartInputTicks = 1;
  setup.scenario.phase = { kind: 'kickoffSetup', teamId: 'home' };
  const decisions = [];
  const { session } = controlled(setup, {
    decisionIntervalTicks: 60,
    decide(teamId, { tick, phase }, assignment) {
      if (teamId !== 'home') return [];
      decisions.push(tick);
      if (phase.kind !== 'kickoffSetup' || tick < phase.readyTick) return [];
      assert.ok(assignment.playerIds.includes(phase.takerId));
      return [playerIntent(phase.takerId, neutral, passRequest)];
    },
  });
  session.step();
  session.step();
  const window = session.step();
  assert.deepEqual(decisions, [0, 2]);
  assert.ok(window.actionResults.some(({ status, action }) => status === 'accepted' && action.source === 'controller'));
});

test('a failed plugin reset is reported and leaves its players neutral', () => {
  const setup = fixture();
  const registry = new TeamAIRegistry().register('fragile', () => ({
    id: 'fragile',
    version: '1',
    reset(context) {
      if (context.teamId === 'home') throw new Error('cannot initialize');
    },
    decide(_observation, assignment) {
      return assignment.playerIds.map((id) => playerIntent(id, east));
    },
    dispose() {},
  }));
  const session = new MatchSession(registry);
  const control = { pluginId: 'fragile' };
  const failures = session.reset(setup.config, 17, { teams: { home: control, away: control } }, setup.scenario);
  assert.deepEqual(failures, [{
    teamId: 'home', pluginId: 'fragile', tick: 0, stage: 'reset', message: 'cannot initialize',
  }]);
  const result = session.step();
  assert.deepEqual(command(result, 'home-1').movement, neutral);
  assert.deepEqual(command(result, 'away-1').movement, east);
});

test('the baseline keeper reads a reachable shot, catches it, and releases it', () => {
  const setup = fixture();
  setup.player('away-keeper').position = { x: 28, y: 0 };
  setup.scenario.ball.position = { x: 26.6, y: 0.78, z: 1.2 };
  setup.scenario.ball.velocity = { x: 12, y: 0, z: 0 };
  const session = new MatchSession();
  session.reset(setup.config, 17, {}, setup.scenario);
  const results = Array.from({ length: 35 }, () => session.step());
  const events = results.flatMap((result) => result.events);
  assert.ok(events.some(({ kind, outcome, action }) =>
    kind === 'actionContact' && outcome === 'caught' && action.playerId === 'away-keeper'));
  assert.ok(events.some(({ kind, outcome, action }) =>
    kind === 'actionContact' && outcome === 'released' && action.playerId === 'away-keeper'));
  assert.deepEqual(session.snapshot().score, [0, 0]);
});

test('the baseline keeper parries a reachable ground shot and concedes a wider one', () => {
  function playShot(y) {
    const setup = fixture();
    setup.player('away-keeper').position = { x: 28, y: 0 };
    setup.scenario.ball.position = { x: 26.4, y, z: setup.config.ball.radiusM };
    setup.scenario.ball.velocity = { x: 20, y: 0, z: 0 };
    const session = new MatchSession();
    session.reset(setup.config, 17, {}, setup.scenario);
    const events = Array.from({ length: 25 }, () => session.step()).flatMap((result) => result.events);
    return {
      parried: events.some((event) => event.kind === 'actionContact' &&
        event.action.playerId === 'away-keeper' && event.outcome === 'parried'),
      goals: session.snapshot().score[0],
    };
  }

  assert.deepEqual(playShot(0.55), { parried: true, goals: 0 });
  assert.deepEqual(playShot(0.8), { parried: false, goals: 1 });
});
