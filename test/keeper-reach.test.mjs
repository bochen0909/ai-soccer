import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig } from "../dist/index.js";
import { keeperSaveVolumes } from "../dist/engine/keeper-save.js";

// A keeper facing +y unless told otherwise.
const player = (facingRad = Math.PI / 2) => {
  const definition = createArcadeConfig().teams[0].players[0];
  return { definition, position: { x: 0, y: 0 }, stamina: 1, balance: 1, facingRad };
};
const length = (volume) => Math.hypot(volume.end.x - volume.start.x, volume.end.y - volume.start.y);

test("keeperReachScale lengthens the save, and leaves it as it was when unset", () => {
  const direction = { x: 1, y: 0 };
  const [base] = keeperSaveVolumes(player(), direction);
  const [same] = keeperSaveVolumes(player(), direction, undefined);
  const [longer] = keeperSaveVolumes(player(), direction, 1.5);
  assert.equal(length(same), length(base));
  assert.ok(length(longer) > length(base));
});

test("a save stretches into a dive over its active window", () => {
  const direction = { x: 0, y: 1 };
  const reachOf = (extension) => Math.max(...keeperSaveVolumes(player(), direction, 1, extension).map(length));
  assert.ok(reachOf(0.5) > reachOf(0));
  assert.ok(reachOf(1) > reachOf(0.5));
  assert.equal(reachOf(2), reachOf(1));
});

test("a save across or behind the keeper's body reaches less far than one in front", () => {
  const reachToward = (direction) => Math.max(...keeperSaveVolumes(player(0), direction, 1, 1).map(length));
  const front = reachToward({ x: 1, y: 0 }), side = reachToward({ x: 0, y: 1 }), behind = reachToward({ x: -1, y: 0 });
  assert.ok(side < front);
  assert.ok(behind < 0.6 * front);
});
