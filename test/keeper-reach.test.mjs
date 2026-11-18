import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig } from "../dist/index.js";
import { keeperSaveVolumes } from "../dist/engine/keeper-save.js";

const player = () => {
  const definition = createArcadeConfig().teams[0].players[0];
  return { definition, position: { x: 0, y: 0 }, stamina: 1, balance: 1 };
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
