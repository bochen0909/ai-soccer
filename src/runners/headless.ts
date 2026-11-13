import type { ActionFrame, MatchConfig, ScenarioDefinition } from "../contracts/index.js";
import { TeamAIRegistry } from "../controllers/registry.js";
import { SoccerEngine } from "../engine/engine.js";
import { PHYSICS_TICK_SECONDS } from "../engine/physics.js";
import { createArcadeConfig } from "../presets/arcade.js";
import type { ControllerFailure, SessionOptions } from "../session/contracts.js";
import { createDefaultAIRegistry, MatchSession } from "../session/match-session.js";
import { ENGINE_VERSION, MATCH_CONTRACT_VERSION, REPLAY_VERSION } from "./replay.js";
import type { MatchReplay, ReplayController, ReplayHeader, ReplayResult } from "./replay.js";

export interface EvaluationOptions {
  config?: MatchConfig;
  seed: number;
  scenario?: ScenarioDefinition;
  teams?: SessionOptions["teams"];
  maxTicks?: number;
  recordReplay?: boolean;
  registry?: TeamAIRegistry;
}

export interface EvaluationResult extends ReplayResult {
  metadata: ReplayHeader;
  replay?: MatchReplay;
}

export interface BenchmarkOptions extends Omit<EvaluationOptions, "recordReplay"> {
  matches?: number;
  warmup?: number;
}

export interface BenchmarkResult {
  mode: "serial";
  warmupMatches: number;
  attemptedMatches: number;
  completedMatches: number;
  truncatedMatches: number;
  ticks: number;
  elapsedSeconds: number;
  ticksPerSecond: number;
  matchesPerSecond: number;
  simulationSeconds: number;
  controllerSeconds: number;
  coordinatorSeconds: number;
  setupAndReportingSeconds: number;
  metadata: {
    engineVersion: ReplayHeader["engineVersion"];
    contractVersion: ReplayHeader["contractVersion"];
    config: MatchConfig;
    scenario?: ScenarioDefinition;
    controllers: readonly Omit<ReplayController, "seed">[];
    tickRate: number;
    maxTicks: number;
  };
  runs: readonly (ReplayResult & { seed: number })[];
}

interface Timings {
  simulationMs: number;
  controllerMs: number;
  steppingMs: number;
}

interface MatchOptions {
  config: MatchConfig;
  scenario?: ScenarioDefinition;
  teams?: SessionOptions["teams"];
  registry: TeamAIRegistry;
  maxTicks: number;
}

function integer(value: number, label: string, min: number, max = Number.MAX_SAFE_INTEGER): number {
  if (!Number.isSafeInteger(value) || value < min || value > max) {
    throw new RangeError(`${label} must be an integer from ${min} to ${max}`);
  }
  return value;
}

/** Evaluation config must survive writing and reading the replay as JSON. */
function jsonSettings(value: unknown, ancestors = new Set<object>()): void {
  if (value === null || typeof value === "boolean" || typeof value === "string" ||
      (typeof value === "number" && Number.isFinite(value))) return;
  if (typeof value !== "object" || value === null || ancestors.has(value) ||
      (!Array.isArray(value) && Object.getPrototypeOf(value) !== Object.prototype &&
        Object.getPrototypeOf(value) !== null)) {
    throw new TypeError("Controller config must contain only finite JSON data");
  }
  ancestors.add(value);
  for (const entry of Array.isArray(value) ? value : Object.values(value)) jsonSettings(entry, ancestors);
  ancestors.delete(value);
}

function resolveOptions(options: EvaluationOptions): MatchOptions {
  integer(options.seed, "Seed", 0, 0xffff_ffff);
  for (const team of Object.values(options.teams ?? {})) {
    if (team?.config !== undefined) jsonSettings(team.config);
  }
  return {
    config: options.config === undefined ? createArcadeConfig() : structuredClone(options.config),
    scenario: structuredClone(options.scenario),
    teams: structuredClone(options.teams),
    registry: options.registry ?? createDefaultAIRegistry(),
    maxTicks: integer(options.maxTicks ?? 120_000, "Tick limit", 1),
  };
}

function rejectFailures(failures: readonly ControllerFailure[]): void {
  if (failures.length === 0) return;
  const details = failures.map(({ teamId, pluginId, stage, tick, message }) =>
    `${pluginId} (${teamId}) ${stage} at tick ${tick}: ${message}`);
  throw new Error(`Controller failure: ${details.join(" | ")}`);
}

class MeasuredEngine extends SoccerEngine {
  constructor(private readonly timings: Timings) { super(); }

  override step(frame: ActionFrame) {
    const start = performance.now();
    try {
      return super.step(frame);
    } finally {
      this.timings.simulationMs += performance.now() - start;
    }
  }
}

function evaluate(
  options: MatchOptions, seed: number, recordReplay: boolean, timings?: Timings,
): EvaluationResult {
  const { config, scenario, teams } = options;
  const controllers: ReplayController[] = [];
  const registry = new TeamAIRegistry();
  for (const { id: teamId } of config.teams) {
    const pluginId = teams && Object.hasOwn(teams, teamId) ? teams[teamId].pluginId : "baseline";
    if (registry.has(pluginId)) continue;
    registry.register(pluginId, () => {
      const plugin = options.registry.create(pluginId);
      return {
        id: plugin.id,
        version: plugin.version,
        reset(context) {
          controllers.push({
            teamId: context.teamId,
            pluginId: plugin.id,
            pluginVersion: plugin.version,
            seed: context.seed,
            decisionIntervalTicks: teams && Object.hasOwn(teams, context.teamId)
              ? teams[context.teamId].decisionIntervalTicks ?? 6 : 6,
            config: structuredClone(context.config),
          });
          plugin.reset(context);
        },
        decide(observation, assignment, context) {
          if (!timings) return plugin.decide(observation, assignment, context);
          const start = performance.now();
          try {
            return plugin.decide(observation, assignment, context);
          } finally {
            timings.controllerMs += performance.now() - start;
          }
        },
        dispose() { plugin.dispose(); },
      };
    });
  }

  const engine = timings ? new MeasuredEngine(timings) : new SoccerEngine();
  const session = new MatchSession(registry, engine);
  let disposal: readonly ControllerFailure[];
  let evaluation: EvaluationResult;
  try {
    rejectFailures(session.reset(config, seed, { teams }, scenario));
    const metadata: ReplayHeader = {
      formatVersion: REPLAY_VERSION,
      engineVersion: ENGINE_VERSION,
      contractVersion: MATCH_CONTRACT_VERSION,
      seed,
      config: structuredClone(config),
      scenario: structuredClone(scenario),
      initial: engine.snapshot(),
      initialHash: engine.hashState(),
      controllers,
    };

    let status: ReplayResult["status"] = "tickLimit";
    const frames = recordReplay ? [] as ActionFrame[] : undefined;
    const started = timings ? performance.now() : 0;
    for (let tick = 0; tick < options.maxTicks; tick += 1) {
      const result = session.step();
      rejectFailures(result.controllerFailures);
      frames?.push(result.frame);
      if (result.status === "finished") {
        status = "finished";
        break;
      }
    }
    if (timings) timings.steppingMs += performance.now() - started;

    const { tick: ticks, score } = engine.snapshot();
    const result: ReplayResult = { status, ticks, score, finalHash: engine.hashState() };
    evaluation = { ...result, metadata, replay: frames && { header: metadata, frames, result } };
  } finally {
    disposal = session.dispose();
  }
  rejectFailures(disposal);
  return evaluation;
}

/** Run a fresh AI-versus-AI session without wall-clock pacing or browser rendering. */
export function evaluateMatch(options: EvaluationOptions): EvaluationResult {
  return evaluate(resolveOptions(options), options.seed, options.recordReplay ?? false);
}

/** Time serial matches after warmup, with no action recording or file I/O. */
export function benchmarkMatches(options: BenchmarkOptions): BenchmarkResult {
  const matches = integer(options.matches ?? 5, "Match count", 1);
  const warmup = integer(options.warmup ?? 1, "Warmup count", 0);
  const resolved = resolveOptions(options);
  for (let index = warmup; index > 0; index -= 1) {
    evaluate(resolved, (options.seed - index) >>> 0, false);
  }

  const timings: Timings = { simulationMs: 0, controllerMs: 0, steppingMs: 0 };
  const runs: (ReplayResult & { seed: number })[] = [];
  let ticks = 0;
  let completed = 0;
  let metadata: ReplayHeader | undefined;
  const start = performance.now();
  for (let index = 0; index < matches; index += 1) {
    const seed = (options.seed + index) >>> 0;
    const result = evaluate(resolved, seed, false, timings);
    metadata ??= result.metadata;
    ticks += result.ticks;
    if (result.status === "finished") completed += 1;
    runs.push({ seed, status: result.status, ticks: result.ticks, score: result.score, finalHash: result.finalHash });
  }
  const elapsedSeconds = (performance.now() - start) / 1000;
  if (!metadata || elapsedSeconds <= 0) throw new Error("Benchmark clock did not record a positive duration");
  const coordinatorMs = Math.max(0, timings.steppingMs - timings.simulationMs - timings.controllerMs);
  return {
    mode: "serial",
    warmupMatches: warmup,
    attemptedMatches: matches,
    completedMatches: completed,
    truncatedMatches: matches - completed,
    ticks,
    elapsedSeconds,
    ticksPerSecond: ticks / elapsedSeconds,
    matchesPerSecond: completed / elapsedSeconds,
    simulationSeconds: timings.simulationMs / 1000,
    controllerSeconds: timings.controllerMs / 1000,
    coordinatorSeconds: coordinatorMs / 1000,
    setupAndReportingSeconds: Math.max(0, elapsedSeconds - timings.steppingMs / 1000),
    metadata: {
      engineVersion: metadata.engineVersion,
      contractVersion: metadata.contractVersion,
      config: metadata.config,
      scenario: metadata.scenario,
      controllers: metadata.controllers.map(({ seed: _seed, ...controller }) => controller),
      tickRate: 1 / PHYSICS_TICK_SECONDS,
      maxTicks: resolved.maxTicks,
    },
    runs,
  };
}
