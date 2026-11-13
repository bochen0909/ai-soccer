import assert from 'node:assert/strict';
import test from 'node:test';
import { createArcadeConfig, PHYSICS_TICK_SECONDS, SoccerEngine } from '../dist/index.js';
import { FixedStepClock } from '../dist/runners/fixed-step.js';

const tickMs = PHYSICS_TICK_SECONDS * 1_000;

function simulation() {
  const config = createArcadeConfig();
  const engine = new SoccerEngine();
  engine.reset(config, 17, {
    version: 1,
    id: 'timing-test',
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
  });
  let tick = 0;
  return {
    engine,
    step() {
      const sprint = tick < 40;
      engine.step({
        tick,
        actions: [{ playerId: 'home-1', movement: { x: sprint ? 1 : 0, y: sprint ? 0 : 1 }, sprint }],
      });
      tick += 1;
    },
  };
}

test('30, 60, and 144 Hz presentation schedules reproduce a headless simulation', () => {
  const headless = simulation();
  for (let tick = 0; tick < 120; tick += 1) headless.step();

  for (const refreshRate of [30, 60, 144]) {
    const browser = simulation();
    const clock = new FixedStepClock(browser.engine.snapshot());
    const stepsPerFrame = new Set();
    for (let frameIndex = 0; frameIndex <= 2 * refreshRate; frameIndex += 1) {
      const frame = clock.advance(1_000 * frameIndex / refreshRate, () => {
        browser.step();
        return { snapshot: browser.engine.snapshot() };
      });
      assert.equal(frame.pauseReason, null);
      assert.ok(frame.alpha >= 0 && frame.alpha < 1);
      assert.equal(frame.previous.tick, Math.max(0, frame.current.tick - 1));
      if (frameIndex > 0) stepsPerFrame.add(frame.steps);
    }
    assert.equal(clock.frame().current.tick, 120);
    assert.equal(browser.engine.hashState(), headless.engine.hashState());
    assert.ok(stepsPerFrame.has(refreshRate < 60 ? 2 : 1));
    if (refreshRate > 60) assert.ok(stepsPerFrame.has(0));
  }
});

test('a scheduling gap aligns the latest state and resumes without catching up paused time', () => {
  const clock = new FixedStepClock(0);
  let tick = 0;
  const step = () => ({ snapshot: ++tick });
  clock.advance(0, step);
  assert.ok(Math.abs(clock.advance(tickMs * 1.5, step).alpha - 0.5) < 1e-10);

  const paused = clock.advance(1_000, step);
  assert.deepEqual(paused, {
    previous: 1, current: 1, alpha: 0, steps: 0, pauseReason: 'gap', finished: false,
  });
  assert.equal(clock.advance(3_000, step).steps, 0);
  assert.equal(clock.advance(3_000 + tickMs, step).current, 2);
});

test('catch-up work stops at the frame limit and reports backlog', () => {
  const clock = new FixedStepClock(0, { maxStepsPerFrame: 2 });
  let tick = 0;
  const step = () => ({ snapshot: ++tick });
  clock.advance(0, step);

  assert.deepEqual(clock.advance(4 * tickMs, step), {
    previous: 2, current: 2, alpha: 0, steps: 2, pauseReason: 'backlog', finished: false,
  });
  assert.equal(clock.advance(5 * tickMs, step).steps, 0);
  assert.equal(clock.advance(6 * tickMs, step).current, 3);
});

test('placement, terminal, and reset snapshots align instead of interpolating', () => {
  const clock = new FixedStepClock('old');
  clock.advance(0, () => assert.fail('the first timestamp must only establish an origin'));
  const placed = clock.advance(tickMs, () => ({ snapshot: 'placed', align: true }));
  assert.equal(placed.previous, 'placed');
  assert.equal(placed.current, 'placed');

  const final = clock.advance(4 * tickMs, () => ({ snapshot: 'full time', finished: true }));
  assert.deepEqual(final, {
    previous: 'full time', current: 'full time', alpha: 0, steps: 1, pauseReason: null, finished: true,
  });
  clock.advance(1_000, () => assert.fail('full time must not advance'));
  clock.align();
  assert.equal(clock.frame().finished, true);

  clock.align('new match');
  assert.deepEqual(clock.frame(), {
    previous: 'new match', current: 'new match', alpha: 0, steps: 0, pauseReason: null, finished: false,
  });
  clock.advance(2_000, () => assert.fail('reset must clear the previous timestamp'));
  assert.equal(clock.advance(2_000 + tickMs, () => ({ snapshot: 'kickoff' })).current, 'kickoff');
});
