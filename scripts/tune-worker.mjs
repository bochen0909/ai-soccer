import { parentPort } from "node:worker_threads";
import { createArcadeConfig, createElevenAsideConfig, evaluateMatch } from "../dist/index.js";

/** One match per task. `candidate` plays `side` against `opponent`; the result is from the candidate's view. */
parentPort.on("message", ({ id, seed, size, side, candidate, opponent }) => {
  try {
    const config = size === 11 ? createElevenAsideConfig() : createArcadeConfig();
    const teams = {
      [side]: { pluginId: "baseline", config: candidate },
      [side === "home" ? "away" : "home"]: { pluginId: "baseline", config: opponent },
    };
    const { score, status } = evaluateMatch({ seed, config, teams });
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
