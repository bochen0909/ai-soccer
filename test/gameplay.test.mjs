import assert from 'node:assert/strict';
import test from 'node:test';
import { createArcadeConfig, createPlayerAbilities, SoccerEngine } from '../dist/index.js';

function fixture(id) {
  const config = createArcadeConfig();
  config.periodTicks = [300, 300];
  const scenario = {
    version: 1,
    id,
    period: 1,
    phase: { kind: 'live' },
    players: config.teams.flatMap((team, teamIndex) =>
      team.players.map((player, index) => ({
        playerId: player.id,
        position: { x: (teamIndex === 0 ? -1 : 1) * (12 + index), y: (index - 2) * 5 },
        facingRad: teamIndex === 0 ? 0 : Math.PI,
      }))),
    ball: {
      position: { x: 0, y: 0, z: config.ball.radiusM },
      velocity: { x: 0, y: 0, z: 0 },
      lastTouchPlayerId: 'away-1',
      control: { kind: 'free' },
    },
  };
  const place = (id, x, y, facingRad = 0) => {
    const player = scenario.players.find(({ playerId }) => playerId === id);
    Object.assign(player, { position: { x, y }, facingRad });
    return player;
  };
  const definition = (id) => config.teams.flatMap(({ players }) => players).find((player) => player.id === id);
  return { config, scenario, place, definition };
}

function engineFor({ config, scenario }, seed = 17) {
  const engine = new SoccerEngine();
  engine.reset(config, seed, scenario);
  return engine;
}

function step(engine, actions = []) {
  return engine.step({ tick: engine.snapshot().tick, actions });
}

function advance(engine, ticks, actions = () => []) {
  const events = [];
  const results = [];
  for (let tick = 0; tick < ticks; tick += 1) {
    const result = step(engine, actions(tick));
    events.push(...result.events);
    results.push(...result.actionResults);
  }
  return { events, results };
}

function action(playerId, movement = { x: 0, y: 0 }, request) {
  return { playerId, movement, sprint: false, request };
}

const groundPass = (id, power = 0.4) => ({
  id, kind: 'kick', kickType: 'pass', direction: { x: 1, y: 0 }, power, loft: 0,
});

test('player presets separate physical capability from editable technical skill', () => {
  const youth = createPlayerAbilities('youth');
  const teen = createPlayerAbilities('teen');
  const adult = createPlayerAbilities('adult');
  assert.ok(youth.physical.maxKickSpeedMps < teen.physical.maxKickSpeedMps);
  assert.ok(teen.physical.maxKickSpeedMps < adult.physical.maxKickSpeedMps);
  youth.technical.passing = 1;
  assert.notEqual(createPlayerAbilities('youth').technical.passing, youth.technical.passing);
  assert.notStrictEqual(youth.physical, createPlayerAbilities('youth').physical);

  const power = (profile) => {
    const setup = fixture('technique-limit');
    setup.place('home-1', 0, 0);
    setup.definition('home-1').abilities = profile;
    setup.scenario.ball.position.x = 0.5;
    setup.scenario.ball.lastTouchPlayerId = 'home-1';
    setup.scenario.ball.control = { kind: 'dribble', playerId: 'home-1' };
    const engine = engineFor(setup);
    const result = advance(engine, 7, (tick) => tick === 0 ? [action('home-1', undefined, groundPass(1, 1))] : []);
    assert.equal(result.events.filter(({ kind, outcome }) => kind === 'actionContact' && outcome === 'kicked').length, 1);
    return Math.hypot(...Object.values(engine.snapshot().ball.velocity));
  };
  const youthSpeed = power(youth);
  const adultSpeed = power(adult);
  assert.ok(youthSpeed <= youth.physical.maxKickSpeedMps * 0.55 + 1e-9);
  assert.ok(adultSpeed <= adult.physical.maxKickSpeedMps * 0.55 + 1e-9);
  assert.ok(youthSpeed < adultSpeed);
});

test('an authored dribbler advances the ball with timed touches and physical travel', () => {
  const setup = fixture('dribble');
  setup.place('home-1', 0, 0);
  setup.definition('home-1').abilities.technical.dribbling = 1;
  setup.scenario.ball.position.x = 0.55;
  setup.scenario.ball.lastTouchPlayerId = 'home-1';
  setup.scenario.ball.control = { kind: 'dribble', playerId: 'home-1' };
  const engine = engineFor(setup);
  const idle = engineFor(setup);
  const movement = [action('home-1', { x: 1, y: 0 })];

  const dribble = advance(engine, 42, () => movement);
  advance(idle, 42);
  const touches = dribble.events.filter(({ kind, outcome }) => kind === 'ballTouch' && outcome === 'dribbled');
  assert.ok(touches.length >= 2, 'moving player makes repeated dribble touches');
  assert.ok(touches.length < 42, 'touch cadence allows the ball to roll between contacts');
  assert.ok(engine.snapshot().ball.position.x > idle.snapshot().ball.position.x + 0.5);
  assert.deepEqual(engine.snapshot().ball.control, { kind: 'dribble', playerId: 'home-1' });
  assert.notDeepEqual(engine.snapshot().ball.velocity, engine.snapshot().players.find(({ playerId }) => playerId === 'home-1').velocity);
});

test('a swept first touch uses the receiver rating to absorb an incoming ground ball', () => {
  function receive(skill) {
    const setup = fixture('first-touch');
    setup.place('home-1', 0, 0, Math.PI);
    setup.definition('home-1').abilities.technical.firstTouch = skill;
    setup.scenario.ball.position.x = -1.2;
    setup.scenario.ball.velocity.x = 10;
    const engine = engineFor(setup);
    for (let tick = 0; tick < 16; tick += 1) {
      const event = step(engine).events.find(({ kind, playerId }) => kind === 'ballTouch' && playerId === 'home-1');
      if (event) return { event, ball: engine.snapshot().ball };
    }
    assert.fail('the ball crosses the receiver’s foot reach');
  }

  const clean = receive(1);
  const poor = receive(0);
  assert.equal(clean.event.outcome, 'received');
  assert.deepEqual(clean.ball.control, { kind: 'dribble', playerId: 'home-1' });
  assert.equal(poor.event.outcome, 'miscontrolled');
  assert.deepEqual(poor.ball.control, { kind: 'free' });
  assert.ok(Math.hypot(...Object.values(clean.ball.velocity)) < Math.hypot(...Object.values(poor.ball.velocity)));
});

test('a received pass reports its originating action and replays exactly', () => {
  const setup = fixture('seeded-pass');
  setup.place('home-1', -2.5, 0);
  setup.place('home-2', 0, 0, Math.PI);
  setup.definition('home-1').abilities.technical.passing = 1;
  setup.definition('home-2').abilities.technical.firstTouch = 1;
  setup.scenario.ball.position.x = -1.95;
  setup.scenario.ball.lastTouchPlayerId = 'home-1';
  setup.scenario.ball.control = { kind: 'dribble', playerId: 'home-1' };
  const engine = engineFor(setup);

  const replay = () => {
    engine.reset(setup.config, 17, setup.scenario);
    return Array.from({ length: 30 }, (_, tick) => ({
      result: engine.step({
        tick,
        actions: tick === 0 ? [action('home-1', undefined, groundPass(1, 0.65))] : [],
      }),
      hash: engine.hashState(),
    }));
  };
  const recorded = replay();
  const events = recorded.flatMap(({ result }) => result.events);
  const contact = events.find(({ kind, outcome }) => kind === 'actionContact' && outcome === 'kicked');
  const reception = events.find(({ kind, playerId, outcome }) =>
    kind === 'ballTouch' && playerId === 'home-2' && outcome === 'received');
  assert.ok(contact);
  assert.ok(reception);
  assert.deepEqual(reception.flightAction, contact.action);
  assert.ok(reception.tick > contact.tick);
  assert.deepEqual(replay(), recorded);
});

test('a standing tackle wins possession and interrupts the opposing shot exactly once', () => {
  const setup = fixture('tackle-interrupts');
  setup.place('home-1', 0, 0);
  setup.place('away-1', 1.15, 0, Math.PI);
  setup.definition('away-1').abilities.technical.tackling = 1;
  setup.scenario.ball.position.x = 0.55;
  setup.scenario.ball.lastTouchPlayerId = 'home-1';
  setup.scenario.ball.control = { kind: 'dribble', playerId: 'home-1' };
  const engine = engineFor(setup);
  const shot = { ...groundPass(1, 1), kickType: 'shot' };
  const tackle = { id: 1, kind: 'tackle', direction: { x: -1, y: 0 } };
  const opener = step(engine, [action('home-1', undefined, shot), action('away-1', undefined, tackle)]);
  assert.equal(opener.actionResults.filter(({ status }) => status === 'accepted').length, 2);

  const busy = step(engine, [action('away-1', undefined, { ...tackle, id: 2 })]);
  assert.ok(busy.actionResults.some(({ status, reason }) => status === 'rejected' && reason === 'busy'));
  const throughContact = advance(engine, 6);
  const contacts = throughContact.events.filter(({ kind }) => kind === 'actionContact');
  assert.equal(contacts.filter(({ outcome }) => outcome === 'wonBall').length, 1);
  assert.equal(contacts.find(({ outcome }) => outcome === 'wonBall')?.action.playerId, 'away-1');
  assert.deepEqual(engine.snapshot().ball.control, { kind: 'dribble', playerId: 'away-1' });
  const interruption = throughContact.results.find(({ action: key, status }) =>
    key.playerId === 'home-1' && key.requestId === 1 && status === 'interrupted');
  assert.ok(interruption);
  assert.equal(interruption.tick, contacts.find(({ outcome }) => outcome === 'wonBall').tick);
  const recovery = advance(engine, 30);
  assert.equal(recovery.results.filter(({ action: key }) => key.playerId === 'home-1' && key.requestId === 1).length, 0);
  assert.equal(recovery.events.filter(({ kind, outcome }) => kind === 'actionContact' && outcome === 'kicked').length, 0);
  assert.equal(recovery.results.filter(({ action: key, status }) =>
    key.playerId === 'away-1' && key.requestId === 1 && status === 'completed').length, 1);
});

function shotAtKeeper(skill, { keeperX = 28, ballX = 26.6, speed = 12 } = {}) {
  const setup = fixture('keeper-save');
  setup.place('away-keeper', keeperX, 0, Math.PI);
  setup.definition('away-keeper').abilities.technical.goalkeeping = skill;
  setup.scenario.ball.position = { x: ballX, y: 0.85, z: 1.2 };
  setup.scenario.ball.velocity.x = speed;
  return setup;
}

const save = { id: 1, kind: 'keeperSave', direction: { x: 0, y: 1 } };

test('a reachable goalkeeper save catches or parries according to goalkeeping skill', () => {
  const unsaved = advance(engineFor(shotAtKeeper(1)), 28);
  assert.equal(unsaved.events.filter(({ kind }) => kind === 'goal').length, 1);
  const run = (skill) => {
    const engine = engineFor(shotAtKeeper(skill));
    const activity = advance(engine, 14, (tick) => tick === 0 ? [action('away-keeper', undefined, save)] : []);
    const contact = activity.events.find(({ kind, action: key, outcome }) =>
      kind === 'actionContact' && key.playerId === 'away-keeper' && outcome !== 'missed');
    assert.ok(contact, 'keeper reaches the shot during the active save');
    assert.ok(activity.events.every(({ kind }) => kind !== 'goal'));
    return { contact, snapshot: engine.snapshot() };
  };
  const high = run(1);
  const low = run(0);
  assert.equal(high.contact.outcome, 'caught');
  assert.deepEqual(high.snapshot.ball.control, { kind: 'keeperHeld', playerId: 'away-keeper' });
  assert.equal(low.contact.outcome, 'parried');
  assert.deepEqual(low.snapshot.ball.control, { kind: 'free' });
  assert.ok(low.snapshot.ball.velocity.x < 0, 'a central parry pushes the ball away from goal');
});

test('save requests require a goalkeeper in its own handling area', () => {
  const setup = fixture('illegal-save');
  setup.place('away-keeper', 0, -4, Math.PI);
  const engine = engineFor(setup);
  const result = step(engine, [action('away-keeper', undefined, save), action('home-1', undefined, save)]);
  assert.equal(result.actionResults.length, 2);
  assert.ok(result.actionResults.every(({ status }) => status === 'rejected'));
  assert.ok(result.events.every(({ kind }) => kind !== 'actionContact'));
});

test('the whole-ball goal crossing ends play before a later active save contact', () => {
  const setup = shotAtKeeper(1, { keeperX: 29.65, ballX: 28.5 });
  setup.scenario.ball.position.y = 0.96;
  setup.scenario.ball.velocity.y = -4;
  const engine = engineFor(setup);
  const reachBack = { ...save, direction: { x: 1, y: 0 } };
  const approach = advance(engine, 8, (tick) => tick === 0 ? [action('away-keeper', undefined, reachBack)] : []);
  assert.equal(approach.events.length, 0);
  assert.equal(engine.snapshot().players.find(({ playerId }) => playerId === 'away-keeper').action?.phase, 'active');
  const result = step(engine);
  assert.equal(result.events.filter(({ kind }) => kind === 'goal').length, 1);
  assert.ok(result.events.every(({ kind }) => kind !== 'actionContact'));
  assert.deepEqual(engine.snapshot().score, [1, 0]);
  assert.equal(engine.snapshot().phase.kind, 'goalPause');
});

test('higher passing skill reduces angular error across the same launch seeds', () => {
  const averageError = (passing) => {
    const setup = fixture('passing-accuracy');
    setup.place('home-1', 0, 0);
    setup.definition('home-1').abilities.technical.passing = passing;
    setup.scenario.ball.position.x = 0.6;
    setup.scenario.ball.lastTouchPlayerId = 'home-1';
    setup.scenario.ball.control = { kind: 'dribble', playerId: 'home-1' };
    let total = 0;
    for (let seed = 0; seed < 12; seed += 1) {
      const engine = engineFor(setup, seed);
      advance(engine, 7, (tick) => tick === 0 ? [action('home-1', undefined, groundPass(1, 0.8))] : []);
      const { x, y } = engine.snapshot().ball.velocity;
      assert.ok(x > 0);
      total += Math.abs(Math.atan2(y, x));
    }
    return total / 12;
  };
  assert.ok(averageError(1) < averageError(0) / 2);
});

test('a low ball squeezed between converging players still advances the tick', () => {
  const setup = fixture('pinched-ball');
  const first = setup.place('away-4', -0.377, 0.013);
  const second = setup.place('home-3', 0.375, 0.031);
  first.velocity = { x: 4.35, y: 0.13 };
  second.velocity = { x: -4.98, y: -0.106 };
  setup.scenario.ball.position.z = 0.1107;
  setup.scenario.ball.velocity = { x: 12.05, y: -3.21, z: 0.396 };
  const engine = engineFor(setup);
  const result = step(engine, [action('away-4', { x: 0.999, y: -0.033 }),
    action('home-3', { x: -0.997, y: -0.081 })]);
  assert.equal(result.tick, 1);
  assert.ok(result.events.some(({ kind, outcome }) => kind === 'ballTouch' && outcome === 'deflected'));
  const ball = engine.snapshot().ball;
  assert.ok(Object.values(ball.position).every(Number.isFinite));
  assert.ok(Math.hypot(ball.velocity.x, ball.velocity.y, ball.velocity.z) <= setup.config.ball.maxSpeedMps + 1e-9);
});

test('a save starts a hold deadline and schedules release even with a short hold clock', () => {
  const setup = shotAtKeeper(1);
  setup.config.rules.keeperHoldTicks = 2;
  const engine = engineFor(setup);
  const activity = advance(engine, 24, (tick) => tick === 0 ? [action('away-keeper', undefined, save)] : []);
  const caught = activity.events.find(({ kind, outcome }) => kind === 'actionContact' && outcome === 'caught');
  const accepted = activity.results.find(({ action: key, status }) => key.source === 'rule' && status === 'accepted');
  const released = activity.events.find(({ kind, outcome }) => kind === 'actionContact' && outcome === 'released');
  assert.ok(caught);
  assert.ok(accepted);
  assert.equal(accepted.tick, caught.tick + setup.config.rules.keeperHoldTicks + 1);
  assert.deepEqual(released?.action, accepted.action);
  assert.equal(engine.snapshot().ball.control.kind, 'free');
});
