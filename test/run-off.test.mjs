import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, SoccerEngine } from "../dist/index.js";

function engineWith(place) {
  const config = createArcadeConfig();
  const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
    playerId: player.id, facingRad: teamIndex === 0 ? 0 : Math.PI,
    position: index === 0 ? { x: teamIndex === 0 ? -29 : 29, y: 0 }
      : { x: (teamIndex === 0 ? -1 : 1) * (8 + index), y: (teamIndex === 0 ? 1 : -1) * 4 },
  })));
  for (const [id, entry] of Object.entries(place)) Object.assign(players.find((player) => player.playerId === id), entry);
  const engine = new SoccerEngine();
  engine.reset(config, 1, { id: "run-off", version: 1, period: 1, phase: { kind: "live" }, players,
    ball: { position: { x: 0, y: 0, z: 0.11 }, velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "home-1", control: { kind: "free" } } });
  return { engine, config };
}

function runTo(engine, playerId, movement, ticks) {
  for (let tick = 0; tick < ticks; tick++) engine.step({ tick, actions: [{ playerId, movement, sprint: true }] });
  return engine.snapshot().players.find((player) => player.playerId === playerId).position;
}

test("players run on past the touchline onto the run-off strip, and stop at its edge", () => {
  const { engine, config } = engineWith({ "home-1": { position: { x: 0, y: 17 } } });
  const { y } = runTo(engine, "home-1", { x: 0, y: 1 }, 120);
  const radius = config.teams[0].players[1].radiusM;
  assert.ok(y > config.pitch.widthM / 2, `${y}`);
  assert.ok(Math.abs(y - (config.pitch.widthM / 2 + config.pitch.runOffM - radius)) < 1e-6, `${y}`);
});

test("the goal is solid: players cannot run into it from the front or from the side", () => {
  const front = engineWith({ "away-2": { position: { x: 26, y: 0.5 } } });
  const fromFront = runTo(front.engine, "away-2", { x: 1, y: 0 }, 90);
  const radius = front.config.teams[1].players[2].radiusM;
  assert.ok(Math.abs(fromFront.x - (front.config.pitch.lengthM / 2 - radius)) < 1e-6, `${fromFront.x}`);

  // Behind the goal line, beside the post, running across toward the middle of the goal.
  const side = engineWith({ "away-2": { position: { x: 31, y: 6 } } });
  const fromSide = runTo(side.engine, "away-2", { x: 0, y: -1 }, 120);
  const { goalWidthM, postRadiusM } = side.config.pitch;
  assert.ok(fromSide.y >= goalWidthM / 2 + postRadiusM + radius - 1e-6, `${fromSide.y}`);
});
