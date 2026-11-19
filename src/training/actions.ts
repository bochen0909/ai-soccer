import type { AttackDirection, PlayerId, TeamObservation, Vec2 } from "../contracts/index.js";
import type { ActionIntent } from "../controllers/contracts.js";
import {
  ACT_NONE, ACT_SAVE, ACT_SHOOT_CENTER, ACT_SHOOT_LEFT, ACT_SHOOT_RIGHT, ACT_TACKLE, MOVE_COUNT,
} from "./contracts.js";
import type { ActionSpace } from "./contracts.js";
import type { BallConfig } from "../contracts/index.js";
import { ballOf, keeperSaveAim, passingPower, SOFT_PASS_ARRIVAL_MPS, tackleAim } from "../controllers/baseline.js";
import { flightLaunchSpeed } from "../engine/ball-flight.js";
import { clamp, lead, toWorld, unitDirection } from "./geometry.js";

type PlayerState = TeamObservation["players"][number];

export interface DecodedAction {
  movement: Vec2;
  sprint: boolean;
  request?: ActionIntent;
}

export interface AgentContext {
  observation: TeamObservation;
  attack: AttackDirection;
  /** Learner players in roster order. Pass slot k means team[k]. */
  team: readonly PlayerState[];
  /** Opposing players in their roster order. */
  opponents: readonly PlayerState[];
  ego: PlayerState;
}

const SHOT_POWER = 0.85;
const SHOT_LOFT = 0.12;
const LOFT_PASS = 0.55;
const LOFT_ELEVATION_RAD = ((15 + 35 * LOFT_PASS) * Math.PI) / 180;

/** Movement in the team's attack frame. Choice 0 stops. */
export function decodeMove(choice: number, attack: AttackDirection): Vec2 {
  if (choice === 0) return { x: 0, y: 0 };
  const angle = ((choice - 1) * Math.PI) / 4;
  return toWorld({ x: Math.cos(angle), y: Math.sin(angle) }, attack);
}

function loftPower(range: number, speedLimit: number, ball: BallConfig): number {
  return clamp(flightLaunchSpeed(range, LOFT_ELEVATION_RAD, ball) / speedLimit, 0.25, 1);
}

const CROSS_ELEVATION_DEG = 24;
const LOB_ELEVATION_DEG = 45;
const HEAD_HEIGHT_M = 1.9;
const ELEVATION_SPAN_DEG = 35;

/**
 * Picks the loft's height and pace. A cross from the wing to a runner in the box is flat enough to arrive at head
 * height; a short loft goes high to clear a defender; anything else keeps the default arc.
 */
function loftFlight(
  context: AgentContext, origin: Vec2, target: Vec2, range: number, speedLimit: number, ball: BallConfig,
): { loft: number; power: number } {
  const { pitch } = context.observation;
  const goalX = context.attack * pitch.lengthM / 2;
  const wide = Math.abs(origin.y) > pitch.goalWidthM && Math.abs(goalX - origin.x) < 32;
  const inBox = Math.abs(goalX - target.x) < 16 && Math.abs(target.y) < pitch.goalWidthM * 1.4;
  if (wide && inBox && range > 8) {
    const theta = (CROSS_ELEVATION_DEG * Math.PI) / 180;
    const rise = range * Math.tan(theta) - HEAD_HEIGHT_M;
    if (rise > 1) {
      const speed = flightLaunchSpeed(range, theta, ball, HEAD_HEIGHT_M);
      return { loft: (CROSS_ELEVATION_DEG - 15) / ELEVATION_SPAN_DEG, power: clamp(speed / speedLimit, 0.3, 1) };
    }
  }
  if (range < 14) {
    const theta = (LOB_ELEVATION_DEG * Math.PI) / 180;
    const speed = flightLaunchSpeed(range, theta, ball);
    return { loft: (LOB_ELEVATION_DEG - 15) / ELEVATION_SPAN_DEG, power: clamp(speed / speedLimit, 0.25, 1) };
  }
  return { loft: LOFT_PASS, power: loftPower(range, speedLimit, ball) };
}

function onPitch(observation: TeamObservation, point: Vec2, margin = 0.7): Vec2 {
  const { lengthM, widthM } = observation.pitch;
  return {
    x: clamp(point.x, -(lengthM / 2 - margin), lengthM / 2 - margin),
    y: clamp(point.y, -(widthM / 2 - margin), widthM / 2 - margin),
  };
}

function passTarget(context: AgentContext, slot: number): Vec2 | undefined {
  const receiver = context.team[slot];
  if (!receiver || receiver.playerId === context.ego.playerId) return undefined;
  return onPitch(context.observation, lead(receiver.position, receiver.velocity, 0.32));
}

function keeperHeldBy(observation: TeamObservation, playerId: PlayerId): boolean {
  const { control } = observation.ball;
  return control.kind === "keeperHeld" && control.playerId === playerId;
}

function inKeeperArea(context: AgentContext): boolean {
  const { pitch } = context.observation;
  const depth = (context.ego.position.x * context.attack) + pitch.lengthM / 2;
  return depth >= -0.05 && depth <= pitch.keeperAreaDepthM + 0.05 &&
    Math.abs(context.ego.position.y) <= pitch.keeperAreaWidthM / 2 + 0.05;
}

/** Writes 1 for every act the current state permits. "None" is always legal. */
export function writeActionMask(context: AgentContext, space: ActionSpace, mask: Uint8Array, offset: number): void {
  const { observation, ego } = context;
  const { phase, ball } = observation;
  const physical = ego.definition.abilities.physical;
  mask.fill(0, offset, offset + space.actCount);
  mask[offset + ACT_NONE] = 1;
  if (ego.action) return;

  const toBall = Math.hypot(ball.position.x - ego.position.x, ball.position.y - ego.position.y, ball.position.z);
  // Hand releases and throw-ins have no soft variant, so only kicked ground passes offer it.
  const targets = (loft: boolean, soft = !loft) => {
    for (let slot = 0; slot < context.team.length; slot += 1) {
      if (!passTarget(context, slot)) continue;
      mask[offset + (loft ? space.loftBase : space.passBase) + slot] = 1;
      if (soft) mask[offset + space.softBase + slot] = 1;
    }
  };

  if (phase.kind === "kickoffSetup" || phase.kind === "restartSetup") {
    if (phase.takerId !== ego.playerId || observation.tick < phase.readyTick ||
        observation.tick >= phase.deadlineTick) return;
    targets(false, phase.kind === "kickoffSetup" || phase.restart !== "throwIn");
    // The engine rejects lofted kickoffs and every non-pass kick at a kickoff.
    if (phase.kind === "restartSetup" && phase.restart !== "throwIn") targets(true);
    return;
  }
  if (phase.kind !== "live") return;

  if (keeperHeldBy(observation, ego.playerId)) {
    targets(false, false);
    targets(true);
    return;
  }
  if (ball.control.kind === "keeperHeld") return;

  const ownDribble = ball.control.kind === "dribble" && ball.control.playerId === ego.playerId;
  const mayKick = toBall <= physical.reachM && (ball.control.kind === "free" || ownDribble);
  if (mayKick) {
    for (const act of [ACT_SHOOT_LEFT, ACT_SHOOT_CENTER, ACT_SHOOT_RIGHT]) mask[offset + act] = 1;
    targets(false);
    targets(true);
  }
  if (!ownDribble && toBall <= 3) mask[offset + ACT_TACKLE] = 1;
  // A save is legal only when one started now could reach the ball, so a dive is never wasted on bad timing.
  if (ego.definition.role === "goalkeeper" && inKeeperArea(context) &&
      keeperSaveAim(observation, context.attack, ego) !== undefined) mask[offset + ACT_SAVE] = 1;
}

/** Converts a legal choice to an engine intent. Returns undefined requests for "none". */
export function decodeAction(
  context: AgentContext, space: ActionSpace, move: number, sprint: number, act: number,
): DecodedAction {
  const { observation, attack, ego } = context;
  const movement = decodeMove(move, attack);
  const base = { movement, sprint: sprint === 1 };
  if (act === ACT_NONE) return base;

  const { ball, pitch, phase } = observation;
  const forward = { x: attack, y: 0 };
  const origin = { x: ball.position.x, y: ball.position.y };
  const physical = ego.definition.abilities.physical;
  const ballConfig = ballOf(observation);

  if (act >= ACT_SHOOT_LEFT && act <= ACT_SHOOT_RIGHT) {
    const side = act === ACT_SHOOT_LEFT ? 1 : act === ACT_SHOOT_RIGHT ? -1 : 0;
    const aim = unitDirection(origin, { x: attack * pitch.lengthM / 2, y: attack * side * pitch.goalWidthM * 0.29 }, forward);
    return { ...base, request: { kind: "kick", kickType: "shot", direction: aim, power: SHOT_POWER, loft: SHOT_LOFT } };
  }
  if (act === ACT_TACKLE) {
    // Lunge where the ball will be at contact, or straight at it when it will be out of reach.
    const aim = tackleAim(observation, ego) ?? unitDirection(ego.position, origin, forward);
    return { ...base, request: { kind: "tackle", direction: aim } };
  }
  if (act === ACT_SAVE) {
    const aim = keeperSaveAim(observation, attack, ego);
    const target = lead(origin, ball.velocity, 0.2);
    return { ...base, request: { kind: "keeperSave", direction: aim ?? unitDirection(ego.position, target, forward) } };
  }

  const soft = act >= space.softBase;
  const loft = !soft && act >= space.loftBase;
  const slot = act - (soft ? space.softBase : loft ? space.loftBase : space.passBase);
  const target = passTarget(context, slot);
  if (!target) return base;
  const range = Math.hypot(target.x - origin.x, target.y - origin.y);
  const aim = unitDirection(origin, target, forward);

  if (phase.kind === "restartSetup" && phase.restart === "throwIn") {
    return { ...base, request: { kind: "throwIn", direction: aim, power: clamp(range / 13, 0.28, 0.7) } };
  }
  if (keeperHeldBy(observation, ego.playerId)) {
    const roll = !loft && range < Math.min(11, physical.maxKeeperRollSpeedMps ** 2 * 0.22);
    const power = roll ? passingPower(range, physical.maxKeeperRollSpeedMps, ballConfig, SOFT_PASS_ARRIVAL_MPS)
      : clamp(range / physical.maxKeeperThrowSpeedMps, 0.32, 0.82);
    return { ...base, request: { kind: "keeperRelease", technique: roll ? "roll" : "throw", direction: aim, power } };
  }
  if (loft) {
    const flight = loftFlight(context, origin, target, range, physical.maxKickSpeedMps * 0.75, ballConfig);
    return { ...base, request: { kind: "kick", kickType: "loft", direction: aim, ...flight } };
  }
  return { ...base, request: {
    kind: "kick", kickType: "pass", direction: aim, loft: 0,
    power: passingPower(range, physical.maxKickSpeedMps * 0.55, ballConfig, soft ? SOFT_PASS_ARRIVAL_MPS : undefined),
  } };
}

export { MOVE_COUNT };
