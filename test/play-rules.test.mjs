import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, SoccerEngine } from "../dist/index.js";

function scenarioPlayers(config, place) {
  const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
    playerId: player.id, facingRad: teamIndex === 0 ? 0 : Math.PI,
    position: index === 0 ? { x: teamIndex === 0 ? -29 : 29, y: 0 }
      : { x: (teamIndex === 0 ? -1 : 1) * (14 + index), y: (teamIndex === 0 ? 1 : -1) * 18 },
  })));
  for (const [id, entry] of Object.entries(place)) Object.assign(players.find((player) => player.playerId === id), entry);
  return players;
}

function liveEngine(place, ball, seed = 1) {
  const config = createArcadeConfig();
  const engine = new SoccerEngine();
  engine.reset(config, seed, { id: "play-rules", version: 1, period: 1, phase: { kind: "live" },
    players: scenarioPlayers(config, place),
    ball: { velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "away-1", control: { kind: "free" }, ...ball } });
  return engine;
}

const idle = (playerId, extra = {}) => ({ playerId, movement: { x: 0, y: 0 }, sprint: false, ...extra });

test("a first-time shot can be shaped up before a rolling ball arrives", () => {
  let kicked = 0;
  for (let seed = 1; seed <= 10; seed++) {
    // A 10 m/s ball rolls across 0.6 m in front of home-1, who starts a hard shot while it is still 2.6 m away.
    const engine = liveEngine({ "home-1": { position: { x: 10, y: 0 }, facingRad: 0 } },
      { position: { x: 10.6, y: -2.6, z: 0.11 }, velocity: { x: 0, y: 10, z: 0 } }, seed);
    const events = [];
    for (let tick = 0; tick < 30; tick++) {
      events.push(...engine.step({ tick, actions: [idle("home-1", tick === 0
        ? { request: { id: 1, kind: "kick", kickType: "shot", direction: { x: 1, y: 0 }, power: 0.8, loft: 0.1 } } : {})] }).events);
    }
    kicked += Number(events.some((event) => event.kind === "actionContact" && event.outcome === "kicked"));
  }
  assert.equal(kicked, 10);
});

test("a kick cannot start on a ball that will not be in reach when the foot comes through", () => {
  const engine = liveEngine({ "home-1": { position: { x: 10, y: 0 }, facingRad: 0 } },
    { position: { x: 10.6, y: -6, z: 0.11 }, velocity: { x: 0, y: 6, z: 0 } });
  const result = engine.step({ tick: 0, actions: [idle("home-1",
    { request: { id: 1, kind: "kick", kickType: "pass", direction: { x: 1, y: 0 }, power: 0.5, loft: 0 } })] });
  assert.equal(result.actionResults[0]?.reason, "ballUnavailable");
});

test("a keeper who lets the ball go cannot pick it up again until another player touches it", () => {
  const config = createArcadeConfig();
  const keeper = config.teams[0].players[0];
  const engine = new SoccerEngine();
  engine.reset(config, 1, { id: "release", version: 1, period: 1, phase: { kind: "live" },
    players: scenarioPlayers(config, { "home-keeper": { position: { x: -27, y: 0 }, facingRad: 0 } }),
    ball: { position: { x: -27 + keeper.radiusM + config.ball.radiusM, y: 0, z: keeper.bodyHeightM * 0.72 },
      velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "home-keeper", control: { kind: "keeperHeld", playerId: "home-keeper" } } });
  const outcomes = [];
  for (let tick = 0; tick < 60; tick++) {
    const request = tick === 0 ? { id: 1, kind: "keeperRelease", technique: "roll", direction: { x: 1, y: 0 }, power: 0.15 }
      : tick === 18 ? { id: 2, kind: "keeperSave", direction: { x: 1, y: 0 } } : undefined;
    for (const event of engine.step({ tick, actions: [idle("home-keeper", { request })] }).events) {
      if (event.kind === "actionContact") outcomes.push(event.outcome);
    }
  }
  assert.deepEqual(outcomes, ["released", "missed"]);
});

test("a restart taker cannot play the ball again until another player touches it", () => {
  for (let seed = 1; seed <= 5; seed++) {
    const config = createArcadeConfig();
    const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => ({
      playerId: player.id, facingRad: 0,
      position: { x: teamIndex ? 25 - index : 20 - 2 * index, y: teamIndex ? -5 : 5 - index },
    })));
    const engine = new SoccerEngine();
    engine.reset(config, seed, { id: "corner", version: 1, period: 1,
      phase: { kind: "restartSetup", restart: "corner", teamId: "home", spot: { x: 30, y: 20 } }, players,
      ball: { position: { x: 30, y: 20, z: 0.11 }, velocity: { x: 0, y: 0, z: 0 }, lastTouchPlayerId: "home-1", control: { kind: "free" } } });
    const taker = engine.snapshot().phase.takerId;
    const touches = [];
    let requested = false;
    let seenLive = false;
    for (let step = 0; step < 300; step++) {
      const snapshot = engine.snapshot();
      if (snapshot.phase.kind === "live") seenLive = true;
      else if (seenLive) break;
      const me = snapshot.players.find((player) => player.playerId === taker);
      let request;
      if (snapshot.phase.kind === "restartSetup" && snapshot.tick >= snapshot.phase.readyTick && !requested) {
        requested = true;
        request = { id: 1, kind: "kick", kickType: "pass", direction: { x: -0.7, y: -0.7 }, power: 0.12, loft: 0 };
      }
      const live = snapshot.phase.kind === "live";
      const toBall = { x: snapshot.ball.position.x - me.position.x, y: snapshot.ball.position.y - me.position.y };
      const kick = live && Math.hypot(toBall.x, toBall.y) < 0.9
        ? { id: 10 + step, kind: "kick", kickType: "pass", direction: { x: -1, y: 0 }, power: 0.5, loft: 0 } : undefined;
      const result = engine.step({ tick: snapshot.tick, actions: [{ playerId: taker, movement: live ? toBall : { x: 0, y: 0 },
        sprint: false, request: request ?? kick }] });
      for (const event of result.events) {
        if ((event.kind === "ballTouch" && event.playerId === taker) ||
            (event.kind === "actionContact" && event.action.playerId === taker)) touches.push(event.outcome);
      }
    }
    assert.equal(touches[0], "kicked");
    assert.deepEqual(touches, ["kicked"]);
  }
});

test("stamina follows the pace run: a sprint drains it, a steady run about holds it, and standing recovers it", () => {
  const drain = (sprint, move) => {
    const engine = liveEngine({ "home-1": { position: { x: -10, y: -5 }, stamina: 0.8 } },
      { position: { x: 0, y: 15, z: 0.11 } });
    for (let tick = 0; tick < 180; tick++) {
      engine.step({ tick, actions: [{ playerId: "home-1", movement: { x: move, y: 0 }, sprint }] });
    }
    return engine.snapshot().players.find((player) => player.playerId === "home-1").stamina - 0.8;
  };
  const sprinting = drain(true, 1);
  const running = drain(false, 1);
  const standing = drain(false, 0);
  assert.ok(sprinting < -0.05, `${sprinting}`);
  assert.ok(Math.abs(running) < 0.03, `a run at 70% of top speed is close to the threshold: ${running}`);
  assert.ok(standing > 0.1, `${standing}`);
});

test("running into another player knocks both off balance", () => {
  const engine = liveEngine({
    "home-1": { position: { x: 0, y: 0 }, velocity: { x: 8, y: 0 } },
    "away-1": { position: { x: 1.2, y: 0 }, velocity: { x: 0, y: 0 } },
  }, { position: { x: 0, y: 15, z: 0.11 } });
  let lowest = { "home-1": 1, "away-1": 1 };
  for (let tick = 0; tick < 12; tick++) {
    engine.step({ tick, actions: [
      { playerId: "home-1", movement: { x: 1, y: 0 }, sprint: true }, idle("away-1"),
    ] });
    for (const player of engine.snapshot().players) {
      if (player.playerId in lowest) lowest[player.playerId] = Math.min(lowest[player.playerId], player.balance);
    }
  }
  assert.ok(lowest["home-1"] < 0.8 && lowest["away-1"] < 0.8, JSON.stringify(lowest));
});
