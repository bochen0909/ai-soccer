import type {
  ActionFrame, ActionKey, ActionRequest, ActionResult, EngineDiagnostic, MatchEvent,
  PlayerAction, PlayerId, PlayerState, Vec2,
} from "../contracts/index.js";
import { playerCapsule, sweepCapsule } from "./physics-geometry.js";
import { randomUnit } from "./random.js";
import type { EngineState } from "./reset.js";

export type EventData<T = MatchEvent> = T extends MatchEvent ? Omit<T, "tick" | "ordinal"> : never;

type LaunchRequest = Extract<ActionRequest, { kind: "kick" | "throwIn" | "keeperRelease" }>;
type Setup = Extract<EngineState["phase"], { kind: "kickoffSetup" | "restartSetup" }>;

interface PendingLaunch {
  action: ActionKey;
  request: LaunchRequest;
  contactTick: number;
  readyTick: number;
  phase: "windup" | "recovery";
  interrupted?: boolean;
}

export type Launches = Map<PlayerId, PendingLaunch>;

const techniques = {
  pass: { speed: 0.55, minElevation: 0, maxElevation: 5, windup: 6, recovery: 8 },
  shot: { speed: 1, minElevation: 0, maxElevation: 25, windup: 8, recovery: 12 },
  loft: { speed: 0.75, minElevation: 15, maxElevation: 50, windup: 12, recovery: 12 },
};

export function isSetup(phase: EngineState["phase"]): phase is Setup {
  return phase.kind === "kickoffSetup" || phase.kind === "restartSetup";
}

function finiteVector(value: Vec2): boolean {
  return Boolean(value && Number.isFinite(value.x) && Number.isFinite(value.y));
}

function unit(value: Vec2): Vec2 {
  const scale = Math.max(Math.abs(value.x), Math.abs(value.y));
  if (scale === 0) return { x: 0, y: 0 };
  const x = value.x / scale;
  const y = value.y / scale;
  const length = Math.hypot(x, y);
  return { x: x / length, y: y / length };
}

const clampUnit = (value: number): number => Math.max(0, Math.min(1, value));

function normalizeRequest(request: ActionRequest): ActionRequest {
  if (!request || !Number.isSafeInteger(request.id) || request.id < 1 ||
      !finiteVector(request.direction) || (request.direction.x === 0 && request.direction.y === 0)) {
    throw new RangeError("Action requests require a positive integer ID and a nonzero finite direction");
  }
  const direction = unit(request.direction);
  switch (request.kind) {
    case "kick":
      if (!Object.hasOwn(techniques, request.kickType) || !Number.isFinite(request.loft) ||
          !Number.isFinite(request.power)) {
        throw new RangeError("Invalid kick parameters");
      }
      return { ...request, direction, power: clampUnit(request.power), loft: clampUnit(request.loft) };
    case "keeperRelease":
      if (request.technique !== "roll" && request.technique !== "throw") {
        throw new RangeError("Invalid goalkeeper release technique");
      }
      // Both hand releases have the same normalized power contract.
    case "throwIn":
      if (!Number.isFinite(request.power)) throw new RangeError("Invalid release power");
      return { ...request, direction, power: clampUnit(request.power) };
    case "tackle":
    case "keeperSave":
      return { ...request, direction };
    default:
      throw new RangeError("Unknown action kind");
  }
}

export function normalizeFrame(state: EngineState, frame: ActionFrame): Map<PlayerId, PlayerAction> {
  if (!frame || frame.tick !== state.tick || !Array.isArray(frame.actions)) {
    throw new RangeError(`Expected an action frame for tick ${state.tick}`);
  }
  const knownIds = new Set(state.players.map((player) => player.playerId));
  const actions = new Map<PlayerId, PlayerAction>();
  for (const action of frame.actions) {
    if (!action || !knownIds.has(action.playerId) || actions.has(action.playerId) ||
        !finiteVector(action.movement) || typeof action.sprint !== "boolean") {
      throw new RangeError("Invalid or duplicate player action");
    }
    const movement = Math.hypot(action.movement.x, action.movement.y) > 1
      ? unit(action.movement) : { ...action.movement };
    const request = action.request === undefined ? undefined : normalizeRequest(action.request);
    actions.set(action.playerId, { playerId: action.playerId, movement, sprint: action.sprint, request });
  }
  return actions;
}

function mayKick(state: EngineState, player: PlayerState): boolean {
  const { control, position } = state.ball;
  const reach = player.definition.abilities.physical.reachM;
  return control.kind !== "keeperHeld" &&
    (control.kind === "free" || control.playerId === player.playerId) &&
    Math.hypot(position.x - player.position.x, position.y - player.position.y, position.z) <= reach;
}

function pointsInfield(state: EngineState, spot: Vec2, direction: Vec2): boolean {
  const { lengthM, widthM } = state.config.pitch;
  return (Math.abs(spot.x) < lengthM / 2 || spot.x * direction.x < 0) &&
    (Math.abs(spot.y) < widthM / 2 || spot.y * direction.y < 0);
}

function rejection(state: EngineState, player: PlayerState, request: ActionRequest): string | undefined {
  const phase = state.phase;
  if (request.kind === "tackle" || request.kind === "keeperSave") return "unsupportedAction";
  if (isSetup(phase)) {
    if (phase.takerId !== player.playerId || state.tick < phase.readyTick ||
        state.tick >= phase.deadlineTick) return "illegalPhase";
    const restart = phase.kind === "kickoffSetup" ? "kickoff" : phase.restart;
    const allowed = restart === "throwIn" ? request.kind === "throwIn" :
      request.kind === "kick" && (request.kickType === "pass" ||
        (restart !== "kickoff" && request.kickType === "loft"));
    if (!allowed || !pointsInfield(state, phase.spot, request.direction)) return "illegalRestart";
    return undefined;
  }
  if (phase.kind !== "live" || request.kind === "throwIn") return "illegalPhase";
  if (request.kind === "keeperRelease") {
    return state.ball.control.kind === "keeperHeld" && state.ball.control.playerId === player.playerId
      ? undefined : "ballUnavailable";
  }
  return mayKick(state, player) ? undefined : "ballUnavailable";
}

function actionTiming(request: LaunchRequest): { windup: number; recovery: number } {
  if (request.kind === "throwIn") return { windup: 12, recovery: 12 };
  if (request.kind === "keeperRelease") return { windup: 6, recovery: 8 };
  const technique = techniques[request.kickType];
  return {
    windup: technique.windup + (request.kickType === "shot" ? Math.round(10 * request.power) : 0),
    recovery: technique.recovery,
  };
}

function accept(
  state: EngineState, launches: Launches, action: ActionKey, request: LaunchRequest,
  results: ActionResult[],
): void {
  const { windup, recovery } = actionTiming(request);
  const contactTick = state.tick + windup;
  launches.set(action.playerId, {
    action, request, contactTick, readyTick: contactTick + recovery, phase: "windup",
  });
  results.push({ action, tick: state.tick + 1, status: "accepted" });
}

function automaticDirection(state: EngineState, player: PlayerState): Vec2 {
  const origin = state.ball.position;
  let closestDistance = Infinity;
  let direction = { x: -origin.x, y: -origin.y };
  const kicking = isSetup(state.phase) &&
    (state.phase.kind === "kickoffSetup" || state.phase.restart !== "throwIn");
  for (const teammate of state.players) {
    if (teammate.teamId !== player.teamId || teammate.playerId === player.playerId) continue;
    const candidate = { x: teammate.position.x - origin.x, y: teammate.position.y - origin.y };
    const distance = Math.hypot(candidate.x, candidate.y);
    if (distance === 0 || distance >= closestDistance || !pointsInfield(state, origin, candidate)) continue;
    if (kicking && sweepCapsule(origin, { ...unit(candidate), z: 0 },
      state.config.ball.radiusM + 0.04, playerCapsule(player), distance)) continue;
    closestDistance = distance;
    direction = candidate;
  }
  if (direction.x === 0 && direction.y === 0) {
    direction = { x: state.attackDirections[state.config.teams[0].id === player.teamId ? 0 : 1], y: 0 };
  }
  return unit(direction);
}

function acceptAutomatic(state: EngineState, launches: Launches, results: ActionResult[]): void {
  const phase = state.phase;
  const setup = isSetup(phase) && state.tick >= phase.deadlineTick ? phase : undefined;
  const held = phase.kind === "live" && state.ball.control.kind === "keeperHeld" &&
    state.keeperReleaseTick !== null && state.tick >= state.keeperReleaseTick ? state.ball.control : undefined;
  const playerId = setup?.takerId ?? held?.playerId;
  if (playerId === undefined || launches.has(playerId)) return;

  const player = state.players.find((entry) => entry.playerId === playerId)!;
  const id = ++state.nextRuleRequestId;
  const direction = automaticDirection(state, player);
  const shared = { id, direction, power: 0.5 };
  let request: LaunchRequest;
  if (held) request = { ...shared, kind: "keeperRelease", technique: "roll" };
  else if (setup?.kind === "restartSetup" && setup.restart === "throwIn") {
    request = { ...shared, kind: "throwIn" };
  } else request = { ...shared, kind: "kick", kickType: "pass", loft: 0 };
  accept(state, launches, { source: "rule", playerId, requestId: id }, request, results);
}

function launch(state: EngineState, player: PlayerState, pending: PendingLaunch, events: EventData[]): void {
  const { request, action } = pending;
  const { physical, technical } = player.definition.abilities;
  const technique = request.kind === "kick" ? techniques[request.kickType] : undefined;
  const maxSpeed = request.kind === "kick" ? physical.maxKickSpeedMps * technique!.speed :
    request.kind === "throwIn" ? physical.maxThrowInSpeedMps :
    request.technique === "roll" ? physical.maxKeeperRollSpeedMps : physical.maxKeeperThrowSpeedMps;
  const skill = request.kind === "keeperRelease" ? technical.goalkeeping :
    request.kind === "kick" && request.kickType === "shot" ? technical.shooting : technical.passing;
  const spread = (0.015 + 0.22 * (1 - skill)) * (0.55 + 0.45 * request.power);
  const angle = Math.atan2(request.direction.y, request.direction.x) + (2 * randomUnit(state) - 1) * spread;
  let direction = { x: Math.cos(angle), y: Math.sin(angle) };
  const setup = isSetup(state.phase);
  if (setup && !pointsInfield(state, state.ball.position, direction)) direction = request.direction;

  const elevationDegrees = request.kind === "kick"
    ? technique!.minElevation + request.loft * (technique!.maxElevation - technique!.minElevation)
    : request.kind === "throwIn" ? 28 : request.technique === "roll" ? 0 : 22;
  const elevation = elevationDegrees * Math.PI / 180;
  const exertion = (0.65 + 0.35 * player.stamina) * (0.7 + 0.3 * player.balance);
  const speed = Math.min(state.config.ball.maxSpeedMps, maxSpeed * request.power * exertion);
  const radius = state.config.ball.radiusM;

  if (request.kind === "throwIn") {
    state.ball.position.z = player.definition.bodyHeightM + radius;
  } else if (request.kind === "keeperRelease") {
    const { lengthM, widthM } = state.config.pitch;
    const offset = player.definition.radiusM + radius + 0.02;
    const limitX = lengthM / 2 - radius;
    const limitY = widthM / 2 - radius;
    state.ball.position = {
      x: Math.max(-limitX, Math.min(limitX, player.position.x + direction.x * offset)),
      y: Math.max(-limitY, Math.min(limitY, player.position.y + direction.y * offset)),
      z: request.technique === "roll" ? radius : player.definition.bodyHeightM * 0.82,
    };
  }
  if (setup) {
    // Setup holds the ball at the restart spot until the taker makes contact.
    player.facingRad = Math.atan2(direction.y, direction.x);
    state.phase = { kind: "live" };
  }
  state.ball.velocity = {
    x: direction.x * speed * Math.cos(elevation),
    y: direction.y * speed * Math.cos(elevation),
    z: speed * Math.sin(elevation),
  };
  state.ball.control = { kind: "free" };
  state.ball.lastTouchPlayerId = player.playerId;
  state.keeperReleaseTick = null;
  events.push({
    kind: "actionContact", action,
    outcome: request.kind === "kick" ? "kicked" : request.kind === "throwIn" ? "thrown" : "released",
    position: { ...state.ball.position },
  });
}

export function processActions(
  state: EngineState, launches: Launches, actions: Map<PlayerId, PlayerAction>,
  events: EventData[], results: ActionResult[], diagnostics: EngineDiagnostic[],
): void {
  for (const player of state.players) {
    const playerId = player.playerId;
    const pending = launches.get(playerId);
    if (pending?.phase === "recovery" && state.tick >= pending.readyTick) {
      launches.delete(playerId);
      if (!pending.interrupted) {
        results.push({ action: pending.action, tick: state.tick + 1, status: "completed" });
      }
    }
    const request = actions.get(playerId)?.request;
    if (!request) continue;
    const action: ActionKey = { source: "controller", playerId, requestId: request.id };
    if (request.id <= (state.requestIds.get(playerId) ?? 0)) {
      diagnostics.push({ kind: "duplicateRequest", action });
      continue;
    }
    state.requestIds.set(playerId, request.id);
    const reason = launches.has(playerId) ? "busy" : rejection(state, player, request);
    if (reason) {
      results.push({ action, tick: state.tick + 1, status: "rejected", reason });
    } else if (request.kind !== "tackle" && request.kind !== "keeperSave") {
      accept(state, launches, action, request, results);
    }
  }
  acceptAutomatic(state, launches, results);

  for (const player of state.players) {
    const pending = launches.get(player.playerId);
    if (!pending || pending.phase !== "windup") continue;
    const request = pending.request;
    const legal = isSetup(state.phase) ||
      (request.kind === "kick" && mayKick(state, player)) ||
      (request.kind === "keeperRelease" && state.ball.control.kind === "keeperHeld" &&
        state.ball.control.playerId === player.playerId);
    if (!legal) {
      launches.set(player.playerId, {
        ...pending,
        phase: "recovery",
        readyTick: state.tick + actionTiming(request).recovery,
        interrupted: true,
      });
      results.push({ action: pending.action, tick: state.tick + 1, status: "interrupted", reason: "ballUnavailable" });
    } else if (state.tick >= pending.contactTick) {
      launch(state, player, pending, events);
      launches.set(player.playerId, { ...pending, phase: "recovery" });
    }
  }
}

export function restrictMovement(
  launches: Launches, actions: Map<PlayerId, PlayerAction>,
): void {
  for (const [id, pending] of launches) {
    const action = actions.get(id);
    if (!action) continue;
    const scale = pending.phase === "windup" ? 0.4 : 0.65;
    actions.set(id, {
      ...action, sprint: false,
      movement: { x: action.movement.x * scale, y: action.movement.y * scale },
    });
  }
}

export function interruptLaunches(
  launches: Launches, tick: number, results: ActionResult[],
): void {
  for (const pending of launches.values()) {
    if (!pending.interrupted) {
      results.push({ action: pending.action, tick, status: "interrupted", reason: "phaseChanged" });
    }
  }
  launches.clear();
}
