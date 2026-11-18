import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, SoccerEngine } from "../dist/index.js";
import { ballPositionAfter, flightLaunchSpeed, rollDistance, rollSpeedAfter } from "../dist/engine/ball-flight.js";

function engineWith(ball, place = {}) {
  const config = createArcadeConfig();
  const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
    playerId: player.id, facingRad: teamIndex === 0 ? 0 : Math.PI,
    position: index === 0 ? { x: teamIndex === 0 ? -29 : 29, y: 0 }
      : { x: (teamIndex === 0 ? -1 : 1) * (14 + index), y: (teamIndex === 0 ? 1 : -1) * 18 },
  })));
  for (const [id, entry] of Object.entries(place)) Object.assign(players.find((player) => player.playerId === id), entry);
  const engine = new SoccerEngine();
  engine.reset(config, 1, { id: "motion", version: 1, period: 1, phase: { kind: "live" }, players,
    ball: { velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "home-1", control: { kind: "free" }, ...ball } });
  return { engine, config };
}

test("a rolling ball slows as the shared prediction says: fast at first, then a long slow roll", () => {
  const { engine, config } = engineWith({ position: { x: -25, y: 0, z: 0.11 }, velocity: { x: 12, y: 0, z: 0 } });
  let tick = 0, speedAt20;
  for (; tick < 600 && speedAt20 === undefined; tick++) {
    engine.step({ tick, actions: [] });
    const { position, velocity } = engine.snapshot().ball;
    if (position.x + 25 >= 20) speedAt20 = velocity.x;
  }
  assert.ok(Math.abs(speedAt20 - rollSpeedAfter(12, 20, config.ball)) < 0.15, `${speedAt20}`);
  // Where the prediction puts the ball after one second matches the engine.
  const fresh = engineWith({ position: { x: -25, y: 0, z: 0.11 }, velocity: { x: 12, y: 0, z: 0 } });
  for (let t = 0; t < 60; t++) fresh.engine.step({ tick: t, actions: [] });
  const predicted = ballPositionAfter({ x: -25, y: 0, z: 0.11 }, { x: 12, y: 0, z: 0 }, 1, config.ball);
  assert.ok(Math.abs(fresh.engine.snapshot().ball.position.x - predicted.x) < 0.1);
  // A soft tap rolls several metres, a hard pass loses a third of its pace in 20 m.
  assert.ok(rollDistance(3, 30, config.ball) > 4.5);
  assert.ok(rollSpeedAfter(15, 20, config.ball) < 11);
});

test("air drag shortens a lofted ball, and a bounce takes ground speed off it", () => {
  const { config } = engineWith({ position: { x: 0, y: 0, z: 0.11 } });
  const elevation = Math.PI / 4;
  // Without drag 30 m at 45 degrees needs about 17 m/s.
  assert.ok(flightLaunchSpeed(30, elevation, config.ball) > 18.5);
  const { engine } = engineWith({ position: { x: -25, y: 0, z: 2 }, velocity: { x: 10, y: 0, z: -2 } });
  let before, after;
  for (let tick = 0; tick < 60 && after === undefined; tick++) {
    const { velocity } = engine.snapshot().ball;
    engine.step({ tick, actions: [] });
    const next = engine.snapshot().ball.velocity;
    if (velocity.z < 0 && next.z > 0) { before = velocity.x; after = next.x; }
  }
  assert.ok(after < 0.85 * before, `${before} -> ${after}`);
  assert.ok(after >= 0.55 * before, "a bounce cannot slow the ball below rolling");
});

test("a sprinter accelerates hardest from a standstill, and runs slower with the ball", () => {
  const top = (carry) => {
    const { engine } = engineWith(carry ? { position: { x: -19.45, y: 0, z: 0.11 }, control: { kind: "dribble", playerId: "home-1" } }
      : { position: { x: 0, y: 10, z: 0.11 } }, { "home-1": { position: { x: -20, y: 0 } } });
    const speeds = [];
    for (let tick = 0; tick < 150; tick++) {
      engine.step({ tick, actions: [{ playerId: "home-1", movement: { x: 1, y: 0 }, sprint: true }] });
      speeds.push(engine.snapshot().players.find((player) => player.playerId === "home-1").velocity.x);
    }
    return speeds;
  };
  const free = top(false);
  assert.ok(free[9] - free[0] > free[69] - free[60], "early acceleration is larger than late");
  assert.ok(Math.max(...top(true)) < 0.92 * Math.max(...free));
});
