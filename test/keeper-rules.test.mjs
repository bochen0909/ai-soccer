import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, SoccerEngine } from "../dist/index.js";

function engineFor(place, ball, seed = 1) {
  const config = createArcadeConfig();
  const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
    playerId: player.id, facingRad: teamIndex === 0 ? 0 : Math.PI,
    position: index === 0 ? { x: teamIndex === 0 ? -29 : 29, y: 0 }
      : { x: (teamIndex === 0 ? -1 : 1) * (14 + index), y: (teamIndex === 0 ? 1 : -1) * 18 },
  })));
  for (const [id, entry] of Object.entries(place)) Object.assign(players.find((player) => player.playerId === id), entry);
  const engine = new SoccerEngine();
  engine.reset(config, seed, { id: "keeper-rules", version: 1, period: 1, phase: { kind: "live" }, players,
    ball: { velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "away-1", control: { kind: "free" }, ...ball } });
  return engine;
}

const keeperEvents = (events) => events.filter((event) =>
  (event.kind === "actionContact" && event.action.playerId === "away-keeper") ||
  (event.kind === "ballTouch" && event.playerId === "away-keeper")).map((event) => event.outcome);

test("the keeper cannot take the ball off a teammate's feet with a save", () => {
  const engine = engineFor({ "away-keeper": { position: { x: 27, y: 0 } }, "away-1": { position: { x: 25.8, y: 0.5 } } },
    { position: { x: 26.35, y: 0.5, z: 0.11 }, control: { kind: "dribble", playerId: "away-1" } });
  const events = [];
  for (let tick = 0; tick < 30; tick++) {
    events.push(...engine.step({ tick, actions: [{ playerId: "away-keeper", movement: { x: 0, y: 0 }, sprint: false,
      request: tick === 0 ? { id: 1, kind: "keeperSave", direction: { x: -0.6, y: 0.5 } } : undefined }] }).events);
  }
  assert.deepEqual(keeperEvents(events), ["missed"]);
  assert.deepEqual(engine.snapshot().ball.control, { kind: "dribble", playerId: "away-1" });
});

test("the back-pass rule: a teammate's pass cannot be handled, so the keeper plays it with their feet", () => {
  const engine = engineFor({ "away-keeper": { position: { x: 27, y: 0 } }, "away-1": { position: { x: 19.45, y: 0 }, facingRad: 0 } },
    { position: { x: 20, y: 0, z: 0.11 }, control: { kind: "dribble", playerId: "away-1" } });
  const events = [];
  let requestId = 10;
  for (let tick = 0; tick < 90; tick++) {
    const { ball, players } = engine.snapshot();
    const keeper = players.find((player) => player.playerId === "away-keeper");
    const near = Math.hypot(ball.position.x - keeper.position.x, ball.position.y - keeper.position.y) < 2.5 && ball.velocity.x > 0.5;
    events.push(...engine.step({ tick, actions: [
      { playerId: "away-1", movement: { x: 0, y: 0 }, sprint: false,
        request: tick === 0 ? { id: 1, kind: "kick", kickType: "pass", direction: { x: 1, y: 0 }, power: 0.45, loft: 0 } : undefined },
      { playerId: "away-keeper", movement: { x: 0, y: 0 }, sprint: false,
        request: near && !keeper.action ? { id: requestId++, kind: "keeperSave", direction: { x: -1, y: 0 } } : undefined },
    ] }).events);
  }
  const outcomes = keeperEvents(events);
  assert.ok(!outcomes.includes("caught"), outcomes.join());
  assert.ok(outcomes.includes("received"), outcomes.join());
});

test("a diving parry goes wide on the side of the dive, not back into the middle", () => {
  let wide = 0;
  for (let seed = 1; seed <= 10; seed++) {
    // A hard shot toward the keeper's right; the keeper (facing the field, -x) dives that way.
    const engine = engineFor({ "away-keeper": { position: { x: 28.6, y: 0 } } },
      { position: { x: 22, y: 1.2, z: 0.11 }, velocity: { x: 20, y: 0.2, z: 0 }, lastTouchPlayerId: "home-1" }, seed);
    let parried;
    for (let tick = 0; tick < 30 && !parried; tick++) {
      const result = engine.step({ tick, actions: [{ playerId: "away-keeper", movement: { x: 0, y: 0 }, sprint: false,
        request: tick === 0 ? { id: 1, kind: "keeperSave", direction: { x: -0.2, y: 0.98 } } : undefined }] });
      if (result.events.some((event) => event.kind === "actionContact" && event.outcome === "parried")) parried = engine.snapshot().ball;
    }
    assert.ok(parried, `seed ${seed}: no parry`);
    if (parried.velocity.y > 2 * Math.abs(parried.velocity.x)) wide += 1;
  }
  assert.equal(wide, 10);
});
