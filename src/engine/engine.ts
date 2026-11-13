import type {
  ActionFrame, ActionResult, EngineDiagnostic, MatchConfig, MatchEngine, MatchEvent, MatchPhase,
  RenderSnapshot, ScenarioDefinition, TeamId, TeamObservation, TickResult,
} from "../contracts/index.js";
import {
  interruptLaunches, normalizeFrame, processActions, restrictMovement,
} from "./actions.js";
import type { EventData, Launches } from "./actions.js";
import { advancePhysics } from "./physics.js";
import type { PhysicsOutcome } from "./physics.js";
import { buildResetState, placeRestart } from "./reset.js";
import type { EngineState } from "./reset.js";

function stageTick(state: EngineState): EngineState {
  return {
    ...state,
    score: [...state.score],
    attackDirections: [...state.attackDirections],
    ball: structuredClone(state.ball),
    players: state.players.map((player) => ({
      ...player,
      position: { ...player.position },
      velocity: { ...player.velocity },
    })),
    requestIds: new Map(state.requestIds),
    events: [],
  };
}

function recordCrossing(
  state: EngineState, crossing: NonNullable<PhysicsOutcome["crossing"]>, events: EventData[],
): () => void {
  const { config, ball } = state;
  const { pitch, teams } = config;
  const { axis, side, position } = crossing;

  if (crossing.kind === "goal") {
    const scorer = state.attackDirections[0] === side ? 0 : 1;
    state.score[scorer] += 1;
    events.push({ kind: "goal", teamId: teams[scorer].id, position, lastTouchPlayerId: ball.lastTouchPlayerId });
    return () => {
      state.phase = { kind: "goalPause", teamId: teams[1 - scorer].id, untilTick: state.tick + config.rules.goalPauseTicks };
    };
  }

  const lastTouch = state.players.find((player) => player.playerId === ball.lastTouchPlayerId);
  if (!lastTouch) throw new Error("A live ball must have a known last touch");
  const awarded = teams.find((team) => team.id !== lastTouch.teamId)!.id;
  if (axis === "y") {
    const spot = { x: Math.max(-pitch.lengthM / 2, Math.min(pitch.lengthM / 2, position.x)), y: side * pitch.widthM / 2 };
    events.push({ kind: "ballExit", restart: "throwIn", teamId: awarded, position });
    return () => placeRestart(state, awarded, "throwIn", spot);
  }

  const attacker = state.attackDirections[0] === side ? teams[0].id : teams[1].id;
  const restart = lastTouch.teamId === attacker ? "goalKick" : "corner";
  const spot = restart === "goalKick"
    ? { x: side * (pitch.lengthM / 2 - pitch.goalAreaDepthM / 2), y: 0 }
    : { x: side * pitch.lengthM / 2, y: position.y < 0 ? -pitch.widthM / 2 : pitch.widthM / 2 };
  events.push({ kind: "ballExit", restart, teamId: awarded, position });
  return () => placeRestart(state, awarded, restart, spot);
}

function endPeriod(state: EngineState): void {
  state.phase = state.period === 1
    ? { kind: "halftime", untilTick: state.tick + state.config.rules.halftimeTicks }
    : { kind: "fullTime" };
  state.ball.control = { kind: "free" };
  state.keeperReleaseTick = null;
}

function advancePause(state: EngineState, startingPhase: MatchPhase): void {
  const center = { x: 0, y: 0 };
  if (startingPhase.kind === "goalPause" && state.tick >= startingPhase.untilTick) {
    placeRestart(state, startingPhase.teamId, "kickoff", center);
  } else if (startingPhase.kind === "halftime" && state.tick >= startingPhase.untilTick) {
    state.period = 2;
    state.attackDirections = [-1, 1];
    state.remainingPeriodTicks = state.config.periodTicks[1];
    const secondKickoff = state.config.teams.find((team) => team.id !== state.config.firstKickoffTeamId)!;
    placeRestart(state, secondKickoff.id, "kickoff", center);
  }
}

export class SoccerEngine implements MatchEngine {
  private state?: EngineState;
  private launches: Launches = new Map();

  reset(config: MatchConfig, seed: number, scenario?: ScenarioDefinition): void {
    const initial = buildResetState(config, seed, scenario);
    this.state = initial;
    this.launches = new Map();
  }

  private requireState(): EngineState {
    if (!this.state) throw new Error("Reset the engine before using it");
    return this.state;
  }

  private view(state: EngineState): RenderSnapshot {
    const { tick, phase, period, remainingPeriodTicks, score, attackDirections, ball } = state;
    const players = state.players.map((player) => {
      const launch = this.launches.get(player.playerId);
      return {
        ...player,
        action: launch ? {
          key: launch.action, kind: launch.request.kind, phase: launch.phase,
          contactTick: launch.contactTick, readyTick: launch.readyTick,
        } : undefined,
      };
    });
    return { tick, phase, period, remainingPeriodTicks, score, attackDirections, ball, players };
  }

  snapshot(): RenderSnapshot {
    return structuredClone(this.view(this.requireState()));
  }

  observe(teamId: TeamId): TeamObservation {
    const state = this.requireState();
    if (!state.config.teams.some((team) => team.id === teamId)) {
      throw new RangeError(`Unknown team ${teamId}`);
    }
    return structuredClone({ ...this.view(state), teamId, pitch: state.config.pitch, events: state.events });
  }

  step(frame: ActionFrame): TickResult {
    const current = this.requireState();
    if (current.phase.kind === "fullTime") throw new Error("The match has finished");
    const actions = normalizeFrame(current, frame);
    const state = stageTick(current);
    const launches = new Map(this.launches);
    const events: EventData[] = [];
    const actionResults: ActionResult[] = [];
    const diagnostics: EngineDiagnostic[] = [];
    processActions(state, launches, actions, events, actionResults, diagnostics);

    let physics: PhysicsOutcome | undefined;
    if (state.phase.kind === "live") {
      restrictMovement(launches, actions);
      physics = advancePhysics(state, actions);
    }
    state.tick += 1;
    if (physics && physics.elapsedSeconds > 0) state.remainingPeriodTicks -= 1;

    const afterCrossing = physics?.crossing && recordCrossing(state, physics.crossing, events);
    if (state.remainingPeriodTicks === 0 && state.phase.kind === "live") {
      interruptLaunches(launches, state.tick, actionResults);
      endPeriod(state);
    } else if (afterCrossing) {
      interruptLaunches(launches, state.tick, actionResults);
      afterCrossing();
    } else {
      advancePause(state, current.phase);
    }
    if (state.phase !== current.phase) events.push({ kind: "phaseChanged", phase: state.phase });
    state.events = events.map((event, ordinal): MatchEvent => ({ ...event, tick: state.tick, ordinal }));

    this.state = state;
    this.launches = launches;
    return structuredClone({
      tick: state.tick, events: state.events, actionResults, diagnostics,
      status: state.phase.kind === "fullTime" ? "finished" : "running",
    });
  }

  /** Hashes simulation data, including RNG and pending actions, for replay checks. */
  hashState(): string {
    const state = this.requireState();
    const serialized = JSON.stringify({
      ...state,
      requestIds: state.players.map(({ playerId }) => state.requestIds.get(playerId) ?? 0),
      launches: state.players.map(({ playerId }) => this.launches.get(playerId) ?? null),
    });
    let hash = 0xcbf29ce484222325n;
    for (let index = 0; index < serialized.length; index += 1) {
      hash = BigInt.asUintN(64, (hash ^ BigInt(serialized.charCodeAt(index))) * 0x100000001b3n);
    }
    return hash.toString(16).padStart(16, "0");
  }
}
