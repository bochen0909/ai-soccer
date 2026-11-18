import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, SoccerEngine } from "../dist/index.js";

function dribbleThenShoot(sprint) {
  const config = createArcadeConfig();
  const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
    playerId: player.id,
    facingRad: teamIndex === 0 ? 0 : Math.PI,
    position: index === 0 ? { x: teamIndex === 0 ? -29 : 29, y: 0 }
      : { x: (teamIndex === 0 ? -1 : 1) * (14 + index), y: (teamIndex === 0 ? 1 : -1) * 18 },
  })));
  players.find((player) => player.playerId === "home-1").position = { x: -10, y: 0 };
  const engine = new SoccerEngine();
  engine.reset(config, 3, {
    id: "kick-windup", version: 1, period: 1, phase: { kind: "live" }, players,
    ball: { position: { x: -9.45, y: 0, z: 0.11 }, velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "home-1",
      control: { kind: "dribble", playerId: "home-1" } },
  });
  const events = [];
  for (let tick = 0; tick < 100; tick++) {
    const request = tick === 60
      ? { id: 1, kind: "kick", kickType: "shot", direction: { x: 1, y: 0 }, power: 1, loft: 0.2 } : undefined;
    const result = engine.step({ tick, actions: [{ playerId: "home-1", movement: { x: 1, y: 0 }, sprint, request }] });
    events.push(...result.events, ...result.actionResults);
  }
  return events;
}

test("a full-power shot on the run goes off: the ball stays at the kicker's feet through the windup", () => {
  for (const sprint of [false, true]) {
    const events = dribbleThenShoot(sprint);
    assert.ok(events.some((event) => event.kind === "actionContact" && event.outcome === "kicked"), `sprint ${sprint}`);
    assert.ok(!events.some((event) => event.status === "interrupted"), `sprint ${sprint}`);
  }
});
