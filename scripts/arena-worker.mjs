import { parentPort } from "node:worker_threads";
import { readFileSync } from "node:fs";
import {
  AI_PROFILES, createArcadeConfig, createDefaultAIRegistry, evaluateMatch, profileTeamOptions, registerNeuralNetwork,
  decodeWeights,
} from "../dist/index.js";

const networks = new Map();

function networkFor(path) {
  if (!networks.has(path)) networks.set(path, decodeWeights(JSON.parse(readFileSync(path, "utf8"))));
  return networks.get(path);
}

/** Team options for a player spec: "default", "tuned" or "model:<absolute path>". Models get their own plugin id. */
function team(spec, registry, id) {
  if (spec.startsWith("greedy:")) return { ...team(spec.slice(7), registry, id), config: { greedy: true } };
  if (spec.startsWith("model:")) {
    registerNeuralNetwork(registry, networkFor(spec.slice(6)), id);
    return { pluginId: id };
  }
  if (!Object.hasOwn(AI_PROFILES, spec)) throw new Error(`Unknown player ${spec}`);
  return profileTeamOptions(spec);
}

/** One match: `a` plays `side` against `b`. Results are from a's view. */
parentPort.on("message", ({ id, seed, a, b, side }) => {
  try {
    const registry = createDefaultAIRegistry();
    const mine = team(a, registry, "neural-a");
    const theirs = team(b, registry, "neural-b");
    const teams = side === "home" ? { home: mine, away: theirs } : { home: theirs, away: mine };
    const { score, status } = evaluateMatch({ seed, config: createArcadeConfig(), teams, registry });
    const [goalsFor, goalsAgainst] = side === "home" ? score : [score[1], score[0]];
    parentPort.postMessage({ id, goalsFor, goalsAgainst, finished: status === "finished" });
  } catch (error) {
    parentPort.postMessage({ id, goalsFor: 0, goalsAgainst: 0, finished: false, crashed: true, message: String(error) });
  }
});
