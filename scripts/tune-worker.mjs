import { parentPort } from "node:worker_threads";
import { readFileSync } from "node:fs";
import {
  createArcadeConfig, createDefaultAIRegistry, createElevenAsideConfig, evaluateMatch, registerNeuralPolicy,
} from "../dist/index.js";

const registries = new Map();

/** A registry with the trained policy loaded from a file, built once per worker. */
function registryFor(model) {
  if (!registries.has(model)) {
    registries.set(model, registerNeuralPolicy(createDefaultAIRegistry(), JSON.parse(readFileSync(model, "utf8"))));
  }
  return registries.get(model);
}

/** One match per task. `candidate` plays `side` against `opponent`; the result is from the candidate's view. */
parentPort.on("message", ({ id, seed, size, side, candidate, opponent, model }) => {
  try {
    const config = size === 11 ? createElevenAsideConfig() : createArcadeConfig();
    const teams = {
      [side]: model ? { pluginId: "neural" } : { pluginId: "baseline", config: candidate },
      [side === "home" ? "away" : "home"]: { pluginId: "baseline", config: opponent },
    };
    const { score, status } = evaluateMatch({ seed, config, teams, registry: model ? registryFor(model) : undefined });
    const [home, away] = score;
    const [goalsFor, goalsAgainst] = side === "home" ? [home, away] : [away, home];
    parentPort.postMessage({ id, goalsFor, goalsAgainst, finished: status === "finished" });
  } catch (error) {
    // An engine failure on one seed (see AI_TRAINING_PLAN.md) must not stop a search. Count it as a draw.
    parentPort.postMessage({
      id, goalsFor: 0, goalsAgainst: 0, finished: false, crashed: true, seed,
      message: error instanceof Error ? error.message : String(error),
    });
  }
});
