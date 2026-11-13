import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, MatchSession } from "../dist/index.js";
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
