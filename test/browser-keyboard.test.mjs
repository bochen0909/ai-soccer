import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, createElevenAsideConfig, MatchSession, SoccerEngine } from "../dist/index.js";
import { KeyboardControls } from "../dist/runners/keyboard.js";

function setup() {
  const config = createArcadeConfig();
  const session = new MatchSession();
  session.reset(config, 17, { human: { teamId: "home", playerId: "home-keeper" } });
  const snapshot = { ...session.snapshot(), phase: { kind: "live" } };
  const surface = new EventTarget();
  surface.ownerDocument = { activeElement: surface };
  surface.contains = (target) => target === surface;
  const keyboard = new EventTarget();
  keyboard.performance = { now: () => 0 };
  const messages = [];
  const controls = new KeyboardControls(surface, keyboard, (message) => messages.push(message));
  function keyEvent(type, code, key, shiftKey = false) {
    const properties = { code, key, shiftKey };
    keyboard.dispatchEvent(Object.assign(new Event(type, { cancelable: true }), properties));
  }
  function press(code, key, shiftKey = false) {
    keyEvent("keydown", code, key, shiftKey);
    keyEvent("keyup", code, key, shiftKey);
  }
  return { config, snapshot, controls, messages, press, keyEvent };
}

test("J plays kickoff back to a teammate and a held direction can aim forward", () => {
  const { config, controls, press, keyEvent } = setup();
  config.rules.directionalKickoff = true;
  const session = new MatchSession();
  session.reset(config, 17, { human: { teamId: "home", playerId: "home-4" } });
  for (let tick = 0; tick < config.rules.restartPlacementTicks; tick += 1) session.step();
  const snapshot = session.snapshot();
  press("KeyJ", "j");
  const command = controls.sample(snapshot, config);
  assert.ok(command.pass?.target.x < 0);
  assert.ok(command.request.direction.x < 0);
  session.step(command);
  let launched = false;
  for (let tick = 0; tick < 12 && !launched; tick += 1) {
    const result = session.step();
    launched = result.events.some((event) => event.kind === "actionContact" && event.outcome === "kicked");
  }
  assert.ok(launched);
  assert.ok(session.snapshot().ball.velocity.x < 0, "the kick must clear the taker and travel backwards");
  for (let tick = 0; tick < 180 && session.snapshot().selectedPlayerId !== command.pass.receiverId; tick += 1) {
    session.step();
  }
  assert.equal(session.snapshot().selectedPlayerId, command.pass.receiverId);

  keyEvent("keydown", "ArrowRight", "ArrowRight");
  press("KeyJ", "j");
  const forward = controls.sample(snapshot, config);
  assert.ok(forward.request.direction.x > 0);
});

test("the full match previews the kickoff receiver and hands over control on reception", () => {
  const { controls, press } = setup();
  const config = createElevenAsideConfig();
  const session = new MatchSession();
  session.reset(config, 17, { human: { teamId: "home", playerId: "home-9" } });
  for (let tick = 0; tick < config.rules.restartPlacementTicks; tick += 1) session.step();
  const snapshot = session.snapshot();
  assert.equal(snapshot.players.length, 22);
  assert.equal(snapshot.phase.takerId, "home-9");
  const preview = controls.previewGroundPass(snapshot, config);
  assert.ok(preview);
  press("KeyJ", "j");
  const command = controls.sample(snapshot, config);
  assert.deepEqual(command.pass, preview);
  session.step(command);
  for (let tick = 0; tick < 200 && session.snapshot().selectedPlayerId !== preview.receiverId; tick += 1) {
    session.step();
  }
  assert.equal(session.snapshot().selectedPlayerId, preview.receiverId);
});

test("an open teammate 35 m away can receive a full-match ground pass", () => {
  const { controls, press } = setup();
  const config = createElevenAsideConfig();
  const passerId = "home-1";
  const receiverId = "home-2";
  const session = new MatchSession();
  session.reset(config, 17, { human: { teamId: "home", playerId: passerId } }, {
    version: 1, id: "long-ground-pass", period: 1, phase: { kind: "live" },
    players: config.teams.flatMap((team, side) => team.players.map((player, index) => ({
      playerId: player.id,
      facingRad: side ? Math.PI : 0,
      position: player.id === passerId ? { x: -16.5, y: 0 } : player.id === receiverId ? { x: 19, y: 0 }
        : { x: side ? 42 : -35, y: -30 + 6 * index },
    }))),
    ball: {
      position: { x: -16, y: 0, z: config.ball.radiusM },
      velocity: { x: 0, y: 0, z: 0 },
      control: { kind: "dribble", playerId: passerId }, lastTouchPlayerId: passerId,
    },
  });
  const snapshot = session.snapshot();
  assert.equal(controls.previewGroundPass(snapshot, config)?.receiverId, receiverId);
  press("KeyJ", "j");
  session.step(controls.sample(snapshot, config));
  for (let tick = 0; tick < 250 && session.snapshot().selectedPlayerId !== receiverId; tick += 1) session.step();
  assert.equal(session.snapshot().selectedPlayerId, receiverId);
});

test("a pass into empty space can roll to a stop before the touchline", () => {
  const { config, controls, press, keyEvent } = setup();
  const playerId = "home-4";
  const start = { x: 8, y: 13, z: config.ball.radiusM };
  const engine = new SoccerEngine();
  engine.reset(config, 17, {
    version: 1, id: "pass-into-space", period: 1, phase: { kind: "live" },
    players: config.teams.flatMap((team, side) => team.players.map((player, index) => ({
      playerId: player.id,
      position: player.id === playerId ? { x: start.x, y: start.y - 0.5 }
        : { x: side === 0 ? -24 : 24, y: 3 * index },
      facingRad: Math.PI / 2,
    }))),
    ball: {
      position: start, velocity: { x: 0, y: 0, z: 0 },
      control: { kind: "dribble", playerId }, lastTouchPlayerId: playerId,
    },
  });
  keyEvent("keydown", "ArrowDown", "ArrowDown");
  press("KeyJ", "j");
  const command = controls.sample({ ...engine.snapshot(), selectedPlayerId: playerId }, config);
  assert.equal(command.pass, undefined);
  const actions = [{ playerId, movement: { x: 0, y: 0 }, sprint: false, request: { ...command.request, id: 1 } }];
  for (let tick = 0; tick < 240; tick += 1) engine.step({ tick, actions: tick === 0 ? actions : [] });
  const { ball, phase } = engine.snapshot();
  const distance = Math.hypot(ball.position.x - start.x, ball.position.y - start.y);
  assert.equal(phase.kind, "live");
  assert.ok(distance > 4 && distance < 6, `pass rolled ${distance.toFixed(1)} m`);
  assert.equal(Math.hypot(ball.velocity.x, ball.velocity.y, ball.velocity.z), 0);
});

test("Q picks a teammate with possession or an outfielder, while Shift+Q can select the keeper", () => {
  const { config, snapshot, controls, press } = setup();
  press("KeyQ", "q");
  const nearest = controls.sample(snapshot, config).selectPlayerId;
  assert.equal(snapshot.players.find((player) => player.playerId === nearest).definition.role, "outfield");

  snapshot.selectedPlayerId = "home-4";
  snapshot.ball.control = { kind: "dribble", playerId: "home-1" };
  press("KeyQ", "q");
  assert.equal(controls.sample(snapshot, config).selectPlayerId, "home-1");

  press("KeyQ", "Q", true);
  const next = controls.sample(snapshot, config).selectPlayerId;
  assert.ok(next && next !== snapshot.selectedPlayerId);
  snapshot.selectedPlayerId = next;
  const visited = new Set([next]);
  for (let index = 0; index < config.teams[0].players.length - 1; index += 1) {
    press("KeyQ", "Q", true);
    snapshot.selectedPlayerId = controls.sample(snapshot, config).selectPlayerId;
    visited.add(snapshot.selectedPlayerId);
  }
  assert.ok(visited.has("home-keeper"));
  assert.equal(visited.size, config.teams[0].players.length);
});

test("an out-of-reach pass explains the failed input without issuing a kick", () => {
  const { config, snapshot, controls, messages, press } = setup();
  press("KeyJ", "j");
  assert.equal(controls.sample(snapshot, config).request, undefined);
  assert.deepEqual(messages, ["Ball out of reach"]);
  controls.sample(snapshot, config);
  assert.equal(messages.length, 1);
});
