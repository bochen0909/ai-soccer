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

test("a ball pinched between a dribbler and a defender squirts out instead of jamming the physics", () => {
  // The defender stands still and tackles once the ball is within a metre; the dribbler runs on into them.
  for (let seed = 1; seed <= 40; seed++) {
    const setup = situation({ "home-1": { position: { x: -6, y: 0 } }, "away-1": { position: { x: 0, y: 0.2 * ((seed % 5) - 2) } } },
      { position: { x: -5.45, y: 0, z: 0.11 }, control: { kind: "dribble", playerId: "home-1" } });
    const engine = new SoccerEngine();
    engine.reset(setup.config, seed, setup.scenario);
    let asked = false;
    assert.doesNotThrow(() => {
      for (let tick = 0; tick < 200 && engine.snapshot().phase.kind === "live"; tick++) {
        const { ball, players } = engine.snapshot();
        const defender = players.find((player) => player.playerId === "away-1");
        const offset = { x: ball.position.x - defender.position.x, y: ball.position.y - defender.position.y };
        const request = !asked && Math.hypot(offset.x, offset.y) < 1 ? { id: 1, kind: "tackle", direction: offset } : undefined;
        asked ||= request !== undefined;
        engine.step({ tick, actions: [{ playerId: "home-1", movement: { x: 1, y: 0 }, sprint: false },
          { playerId: "away-1", movement: { x: 0, y: 0 }, sprint: false, request }] });
      }
    }, `seed ${seed}`);
  }
});

test("a scramble of every player converging on a loose ball settles pile-ups instead of throwing", () => {
  let state = 12345;
  const random = () => ((state = (state * 1103515245 + 12345) >>> 0) / 2 ** 32);
  for (let episode = 0; episode < 12; episode++) {
    const config = createArcadeConfig();
    const placed = [];
    const players = config.teams.flatMap((team, teamIndex) => team.players.map((player, index) => {
      let position;
      for (let attempt = 0; attempt < 200; attempt++) {
        position = index === 0 ? { x: (teamIndex === 0 ? -27 : 27) + (random() - 0.5) * 2, y: (random() - 0.5) * 6 }
          : { x: (random() - 0.5) * 4, y: (random() - 0.5) * 4 };
        if (placed.every((other) => Math.hypot(other.x - position.x, other.y - position.y) > 0.7) && Math.hypot(position.x, position.y) > 0.5) break;
      }
      placed.push(position);
      return { playerId: player.id, position, facingRad: random() * 6.28 };
    }));
    const engine = new SoccerEngine();
    try {
      engine.reset(config, episode + 1, { id: "scramble", version: 1, period: 1, phase: { kind: "live" }, players,
        ball: { position: { x: 0, y: 0, z: 0.11 }, velocity: { x: (random() - 0.5) * 6, y: (random() - 0.5) * 6, z: random() < 0.3 ? random() * 4 : 0 },
          lastTouchPlayerId: "home-1", control: { kind: "free" } } });
    } catch { continue; }
    const requestIds = {};
    assert.doesNotThrow(() => {
      for (let tick = 0; tick < 240 && engine.snapshot().phase.kind === "live"; tick++) {
        const snapshot = engine.snapshot();
        const actions = snapshot.players.map((player) => {
          const toBall = { x: snapshot.ball.position.x - player.position.x, y: snapshot.ball.position.y - player.position.y };
          const norm = Math.hypot(toBall.x, toBall.y) || 1;
          const movement = random() < 0.8 ? { x: toBall.x / norm, y: toBall.y / norm } : { x: random() * 2 - 1, y: random() * 2 - 1 };
          let request;
          if (random() < 0.05) {
            requestIds[player.playerId] = (requestIds[player.playerId] ?? 0) + 1;
            const direction = { x: random() * 2 - 1 || 1, y: random() * 2 - 1 };
            const pick = random();
            request = pick < 0.4 ? { id: requestIds[player.playerId], kind: "tackle", direction }
              : pick < 0.8 ? { id: requestIds[player.playerId], kind: "kick", kickType: ["pass", "shot", "loft"][Math.floor(random() * 3)], direction, power: random(), loft: random() }
              : { id: requestIds[player.playerId], kind: "keeperSave", direction };
          }
          return { playerId: player.playerId, movement: { x: Math.max(-1, Math.min(1, movement.x)), y: Math.max(-1, Math.min(1, movement.y)) },
            sprint: random() < 0.6, request };
        });
        engine.step({ tick: snapshot.tick, actions });
      }
    }, `episode ${episode}`);
  }
});

test("a standing tackle lunges past standing reach, and a miss leaves the tackler off balance", () => {
  const engine = new SoccerEngine();
  const { config, scenario } = situation({ "home-1": { position: { x: -0.6, y: 0 } }, "away-1": { position: { x: 0.9, y: 0 } } },
    { position: { x: 0, y: 0, z: 0.11 }, control: { kind: "dribble", playerId: "home-1" } }, (edit) => {
      edit.teams[0].players[1].abilities.technical.dribbling = 1;
      edit.teams[1].players[1].abilities.technical.tackling = 0.6;
    });
  const outcomes = [];
  for (let seed = 1; seed <= 16; seed++) {
    engine.reset(config, seed, scenario);
    let contact, balance = 1;
    for (let tick = 0; tick < 20; tick++) {
      const result = engine.step({ tick, actions: [{ playerId: "away-1", movement: { x: 0, y: 0 }, sprint: false,
        request: tick === 0 ? { id: 1, kind: "tackle", direction: { x: -1, y: 0 } } : undefined }] });
      contact ??= result.events.find((event) => event.kind === "actionContact");
      balance = Math.min(balance, engine.snapshot().players.find((player) => player.playerId === "away-1").balance);
    }
    outcomes.push(contact.outcome);
    if (contact.outcome === "missed") assert.ok(balance < 0.6, `balance ${balance}`);
  }
  // The ball sits 0.9 m from the tackler, beyond the 0.84 m a plain standing reach gives: the lunge gets there.
  assert.ok(outcomes.includes("wonBall"));
  assert.ok(outcomes.includes("missed"));
});

test("a dribbler shielding the ball with their body is much harder to rob", async () => {
  const { dispossessChance } = await import("../dist/engine/ball-control.js");
  const config = createArcadeConfig();
  const [owner, challenger] = [config.teams[0].players[1], config.teams[1].players[1]];
  const state = (ballX) => ({
    config, tick: 0,
    ball: { position: { x: ballX, y: 0, z: 0.11 } },
    players: [
      { playerId: owner.id, teamId: "home", definition: owner, position: { x: 0, y: 0 }, velocity: { x: 0, y: 0 }, stamina: 1, balance: 1, facingRad: 0 },
      { playerId: challenger.id, teamId: "away", definition: challenger, position: { x: 1, y: 0 }, velocity: { x: 0, y: 0 }, stamina: 1, balance: 1, facingRad: Math.PI },
    ],
  });
  const exposed = state(0.55), shielded = state(-0.55);
  const chance = (world) => dispossessChance(world, world.players[1], world.players[0], 0.45);
  assert.ok(chance(shielded) < chance(exposed) - 0.3);
});
