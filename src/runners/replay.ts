import type { ActionFrame, MatchConfig, RenderSnapshot, ScenarioDefinition, TeamId } from "../contracts/index.js";
import { SoccerEngine } from "../engine/engine.js";

export const REPLAY_VERSION = 1;
export const ENGINE_VERSION = "0.1.0";
export const MATCH_CONTRACT_VERSION = 1;

export interface ReplayController {
  teamId: TeamId;
  pluginId: string;
  pluginVersion: string;
  seed: number;
  decisionIntervalTicks: number;
  config?: unknown;
}

export interface ReplayHeader {
  formatVersion: typeof REPLAY_VERSION;
  engineVersion: typeof ENGINE_VERSION;
  contractVersion: typeof MATCH_CONTRACT_VERSION;
  seed: number;
  config: MatchConfig;
  scenario?: ScenarioDefinition;
  initial: RenderSnapshot;
  initialHash: string;
  controllers: readonly ReplayController[];
}

export interface ReplayResult {
  status: "finished" | "tickLimit";
  ticks: number;
  score: readonly [number, number];
  finalHash: string;
}

export interface MatchReplay {
  header: ReplayHeader;
  /** Frames submitted to the engine, before its single normalization pass. */
  frames: readonly ActionFrame[];
  result: ReplayResult;
}

function object(value: unknown, name: string): Record<string, unknown> {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`Replay ${name} must be an object`);
  }
  return value as Record<string, unknown>;
}

function isInteger(value: unknown, min: number, max = Number.MAX_SAFE_INTEGER): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= min && value <= max;
}

function isHash(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{16}$/.test(value);
}

function validateControllers(controllers: unknown, config: MatchConfig): void {
  if (!Array.isArray(controllers) || controllers.length !== config.teams.length) {
    throw new TypeError("Replay must identify one controller per team");
  }
  const teams = new Set(config.teams.map(({ id }) => id));
  for (const value of controllers as unknown[]) {
    const { teamId, pluginId, pluginVersion, seed, decisionIntervalTicks } = object(value, "controller");
    if (typeof teamId !== "string" || !teams.delete(teamId) ||
        typeof pluginId !== "string" || !pluginId.trim() ||
        typeof pluginVersion !== "string" || !pluginVersion.trim() ||
        !isInteger(seed, 0, 0xffffffff) || !isInteger(decisionIntervalTicks, 1)) {
      throw new TypeError("Replay controller metadata is invalid");
    }
  }
}

/** Compare JSON-visible data. Property order and optional undefined fields can change on disk. */
function sameJson(expected: unknown, actual: unknown): boolean {
  if (expected === actual) return true;
  if (Array.isArray(expected)) {
    return Array.isArray(actual) && expected.length === actual.length &&
      expected.every((entry, index) => sameJson(entry, actual[index]));
  }
  if (!expected || !actual || typeof expected !== "object" || typeof actual !== "object" || Array.isArray(actual)) {
    return false;
  }

  const left = expected as Record<string, unknown>;
  const right = actual as Record<string, unknown>;
  const keys = Object.keys(left).filter((key) => left[key] !== undefined);
  return keys.length === Object.keys(right).filter((key) => right[key] !== undefined).length &&
    keys.every((key) => Object.hasOwn(right, key) && sameJson(left[key], right[key]));
}

/** Reapply a recording through a fresh engine and verify its reset state and final result. */
export function verifyReplay(value: unknown): ReplayResult {
  const replay = object(value, "record");
  const header = object(replay.header, "header");
  if (header.formatVersion !== REPLAY_VERSION) throw new Error("Unsupported replay format version");
  if (header.engineVersion !== ENGINE_VERSION) throw new Error("Unsupported replay engine version");
  if (header.contractVersion !== MATCH_CONTRACT_VERSION) throw new Error("Unsupported replay contract version");
  if (!isInteger(header.seed, 0, 0xffffffff)) throw new TypeError("Replay seed must be a uint32");
  if (!isHash(header.initialHash)) throw new TypeError("Replay initial state hash is invalid");

  const result = object(replay.result, "result");
  const frames = replay.frames;
  if (!Array.isArray(frames)) throw new TypeError("Replay frames must be an array");
  if (!isInteger(result.ticks, 0) || result.ticks !== frames.length) {
    throw new Error("Replay tick count does not match its frame count");
  }
  if (result.status !== "finished" && result.status !== "tickLimit") throw new TypeError("Replay result status is invalid");
  if (!isHash(result.finalHash)) throw new TypeError("Replay final state hash is invalid");

  // Reset and step already validate the simulation contracts, including nested fields.
  const engine = new SoccerEngine();
  const config = header.config as MatchConfig;
  try {
    engine.reset(config, header.seed, header.scenario as ScenarioDefinition | undefined);
  } catch (cause) {
    throw new Error(`Invalid replay reset: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
  }
  validateControllers(header.controllers, config);
  if (engine.hashState() !== header.initialHash) throw new Error("Replay initial state hash does not match");
  if (!sameJson(engine.snapshot(), header.initial)) throw new Error("Replay initial snapshot does not match");

  let status: ReplayResult["status"] = "tickLimit";
  for (const [tick, value] of (frames as unknown[]).entries()) {
    const frame = object(value, `frame ${tick}`);
    if (frame.tick !== tick) throw new Error(`Replay frame ${tick} is out of sequence`);
    if (status === "finished") throw new Error(`Replay frame ${tick} is after full time`);
    try {
      if (engine.step(frame as unknown as ActionFrame).status === "finished") status = "finished";
    } catch (cause) {
      throw new Error(`Invalid replay frame ${tick}: ${cause instanceof Error ? cause.message : String(cause)}`, { cause });
    }
  }

  const { tick: ticks, score } = engine.snapshot();
  if (ticks !== result.ticks) throw new Error("Replay final tick does not match");
  if (status !== result.status) throw new Error("Replay result status does not match");
  if (!sameJson(score, result.score)) throw new Error("Replay final score does not match");
  const finalHash = engine.hashState();
  if (finalHash !== result.finalHash) throw new Error("Replay final state hash does not match");
  return { status, ticks, score, finalHash };
}
