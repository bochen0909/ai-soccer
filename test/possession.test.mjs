import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, SoccerEngine } from "../dist/index.js";

// Everyone not named stands well away from the ball.
function situation(place, ball, edit = () => {}) {
  const config = createArcadeConfig();
  edit(config);
  const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
    playerId: player.id,
    facingRad: teamIndex === 0 ? 0 : Math.PI,
    position: index === 0 ? { x: teamIndex === 0 ? -29 : 29, y: 0 }
      : { x: (teamIndex === 0 ? -1 : 1) * (14 + index), y: (teamIndex === 0 ? 1 : -1) * 18 },
  })));
  for (const [id, entry] of Object.entries(place)) Object.assign(players.find((player) => player.playerId === id), entry);
  return { config, scenario: { id: "possession", version: 1, period: 1, phase: { kind: "live" }, players,
    ball: { velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "home-1", control: { kind: "free" }, ...ball } } };
}

function play({ config, scenario }, seed, ticks, actions) {
  const engine = new SoccerEngine();
  engine.reset(config, seed, scenario);
  const events = [];
  for (let tick = 0; tick < ticks; tick++) events.push(...engine.step({ tick, actions: actions(tick) }).events);
  return { events, ball: engine.snapshot().ball };
}

const tackle = { id: 1, kind: "tackle", direction: { x: -1, y: 0 } };

test("a skilled dribbler can ride a poor tackle and keep the ball", () => {
  const setup = () => situation({ "home-1": { position: { x: 0, y: 0 } }, "away-1": { position: { x: 1.3, y: 0 } } },
    { position: { x: 0.55, y: 0, z: 0.11 }, control: { kind: "dribble", playerId: "home-1" } }, (config) => {
      config.teams[0].players[1].abilities.technical.dribbling = 1;
      config.teams[1].players[1].abilities.technical.tackling = 0;
    });
  const outcomes = new Set();
  for (let seed = 1; seed <= 12; seed++) {
    const { events, ball } = play(setup(), seed, 12, (tick) => [{ playerId: "away-1", movement: { x: 0, y: 0 }, sprint: false,
      request: tick === 0 ? tackle : undefined }]);
    const contact = events.find((event) => event.kind === "actionContact" && event.action.playerId === "away-1");
    outcomes.add(contact?.outcome);
    if (contact?.outcome === "missed") assert.deepEqual(ball.control, { kind: "dribble", playerId: "home-1" });
  }
  assert.ok(outcomes.has("missed"));
});

test("a tackle at a loose ball pokes it away instead of collecting it", () => {
  const { events, ball } = play(situation({ "away-1": { position: { x: 1.3, y: 0 } } }, { position: { x: 0.6, y: 0, z: 0.11 } }),
    1, 12, (tick) => [{ playerId: "away-1", movement: { x: 0, y: 0 }, sprint: false, request: tick === 0 ? tackle : undefined }]);
  assert.ok(events.some((event) => event.kind === "actionContact" && event.outcome === "poked"));
  assert.equal(ball.control.kind, "free");
  assert.ok(ball.velocity.x < -1);
});

test("a ball arriving at chest height is cushioned and kept", () => {
  const speed = 8, flight = 2 / speed;
  for (let seed = 1; seed <= 6; seed++) {
    const { events, ball } = play(situation({ "home-2": { position: { x: 0, y: 0 }, facingRad: Math.PI } },
      { position: { x: -2, y: 0.05, z: 0.8 + 0.5 * 9.81 * flight * flight }, velocity: { x: speed, y: 0, z: 0 } }),
    seed, 60, () => [{ playerId: "home-2", movement: { x: 0, y: 0 }, sprint: false }]);
    const touch = events.find((event) => event.kind === "ballTouch" && event.playerId === "home-2");
    assert.equal(touch?.outcome, "received");
    assert.ok(touch.position.z > 0.5);
    assert.deepEqual(ball.control, { kind: "dribble", playerId: "home-2" });
  }
});

test("a kick struck behind the kicker is weaker than one struck ahead", () => {
  const speedOf = (direction) => {
    const engine = new SoccerEngine();
    const { config, scenario } = situation({ "home-1": { position: { x: 0, y: 0 }, facingRad: 0 } },
      { position: { x: 0.55, y: 0, z: 0.11 }, control: { kind: "dribble", playerId: "home-1" } });
    engine.reset(config, 5, scenario);
    let fastest = 0;
    for (let tick = 0; tick < 12; tick++) {
      engine.step({ tick, actions: [{ playerId: "home-1", movement: { x: 0, y: 0 }, sprint: false,
        request: tick === 0 ? { id: 1, kind: "kick", kickType: "pass", direction, power: 0.6, loft: 0 } : undefined }] });
      const { velocity } = engine.snapshot().ball;
      fastest = Math.max(fastest, Math.hypot(velocity.x, velocity.y));
    }
    return fastest;
  };
  assert.ok(speedOf({ x: -1, y: 0 }) < 0.8 * speedOf({ x: 1, y: 0 }));
});
