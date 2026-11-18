import type { ActionRequest, PlayerState, Vec3 } from "../contracts/index.js";
import { executionFactors, footQuality } from "./abilities.js";
import { keeperSaveTiming, keeperSaveVolumes } from "./keeper-save.js";
import { along, capsuleDistanceSquared, sweepCapsule } from "./physics-geometry.js";
import type { Capsule } from "./physics-geometry.js";
import { attackDirection, keeperHoldPosition } from "./reset.js";
import type { EngineState } from "./reset.js";

export type DefenseRequest = Extract<ActionRequest, { kind: "tackle" | "keeperSave" }>;

export function isDefenseRequest(request: ActionRequest): request is DefenseRequest {
  return request.kind === "tackle" || request.kind === "keeperSave";
}

export function defenseTiming(player: PlayerState, request: DefenseRequest) {
  if (request.kind === "tackle") {
    const skill = player.definition.abilities.technical.tackling;
    return { windup: 7 - Math.round(3 * skill), active: 4, recovery: 13 - Math.round(5 * skill) };
  }
  return keeperSaveTiming(player.definition.abilities.technical.goalkeeping);
}

function tackleVolume(state: EngineState, player: PlayerState, request: DefenseRequest): Capsule {
  const { physical, technical } = player.definition.abilities;
  const { exertion } = executionFactors(state, player);
  const foot = footQuality(player, state.ball.position);
  const reach = physical.reachM * (0.72 + 0.28 * technical.tackling) * (0.85 + 0.15 * exertion * foot);
  // This volume tests the ball center, so its far edge is also the control limit.
  const radiusM = Math.min(player.definition.radiusM * 0.6, reach * 0.28);
  const footCenter = (distance: number): Vec3 => ({
    x: player.position.x + request.direction.x * distance,
    y: player.position.y + request.direction.y * distance,
    z: state.config.ball.radiusM,
  });
  return { start: footCenter(reach * 0.35), end: footCenter(reach - radiusM), radiusM };
}

/** Restricts a linear path to the times at which it intersects an interval. */
function clipAxis(
  times: [number, number], position: number, velocity: number, min: number, max: number,
): boolean {
  if (velocity === 0) return position >= min && position <= max;
  const a = (min - position) / velocity;
  const b = (max - position) / velocity;
  times[0] = Math.max(times[0], Math.min(a, b));
  times[1] = Math.min(times[1], Math.max(a, b));
  return times[0] <= times[1];
}

function saveInterval(state: EngineState, keeper: PlayerState, times: [number, number]): boolean {
  const { pitch, ball } = state.config;
  const ballState = state.ball;
  const forward = attackDirection(state, keeper.teamId);
  const fieldDepth = (x: number): number => forward * x + pitch.lengthM / 2;
  const halfWidth = pitch.keeperAreaWidthM / 2;
  return clipAxis(times, fieldDepth(keeper.position.x), forward * keeper.velocity.x, 0, pitch.keeperAreaDepthM) &&
    clipAxis(times, keeper.position.y, keeper.velocity.y, -halfWidth, halfWidth) &&
    clipAxis(times, fieldDepth(ballState.position.x), forward * ballState.velocity.x,
      -ball.radiusM, pitch.keeperAreaDepthM) &&
    clipAxis(times, ballState.position.y, ballState.velocity.y, -halfWidth, halfWidth);
}

export function defenseCandidate(
  state: EngineState, player: PlayerState, request: DefenseRequest, duration: number,
): { time: number } | undefined {
  const { ball, players } = state;
  const control = ball.control;
  if (control.kind === "keeperHeld" || (control.kind === "dribble" && control.playerId === player.playerId)) {
    return undefined;
  }
  const times: [number, number] = [0, duration];
  const ballRadius = state.config.ball.radiusM;
  let volumes: Capsule[];
  let sweepRadius = 0;
  if (request.kind === "keeperSave") {
    if (player.definition.role !== "goalkeeper" || !saveInterval(state, player, times)) return undefined;
    volumes = keeperSaveVolumes(player, request.direction, state.config.tuning.version === 2, state.config.tuning.keeperReachScale);
    sweepRadius = ballRadius;
  } else {
    if (control.kind === "dribble" &&
      players.some((owner) => owner.playerId === control.playerId && owner.teamId === player.teamId)) {
      return undefined;
    }
    // Standing tackles can only contact a ball low enough to be put under foot control.
    const top = ballRadius + player.definition.radiusM;
    if (!clipAxis(times, ball.position.z, ball.velocity.z, ballRadius, top)) return undefined;
    volumes = [tackleVolume(state, player, request)];
  }

  const relativeVelocity = {
    x: ball.velocity.x - player.velocity.x,
    y: ball.velocity.y - player.velocity.y,
    z: ball.velocity.z,
  };
  const relativePosition = along(ball.position, relativeVelocity, times[0]);
  let earliest = Infinity;
  for (const volume of volumes) {
    if (capsuleDistanceSquared(relativePosition, volume) <= (volume.radiusM + sweepRadius) ** 2) {
      return { time: times[0] };
    }
    const hit = sweepCapsule(relativePosition, relativeVelocity, sweepRadius, volume, times[1] - times[0]);
    if (hit) earliest = Math.min(earliest, hit.time);
  }
  return Number.isFinite(earliest) ? { time: times[0] + earliest } : undefined;
}

export function resolveDefense(
  state: EngineState, player: PlayerState, request: DefenseRequest,
): "wonBall" | "caught" | "parried" {
  const { ball } = state;
  const maxBallSpeed = state.config.ball.maxSpeedMps;
  ball.lastTouchPlayerId = player.playerId;
  delete ball.flightAction;

  if (request.kind === "tackle") {
    const skill = player.definition.abilities.technical.tackling;
    const distance = Math.hypot(ball.position.x - player.position.x, ball.position.y - player.position.y);
    const controlMargin = Math.max(0, player.definition.abilities.physical.reachM - distance);
    const speed = Math.min(maxBallSpeed, 0.65 + 0.65 * skill, controlMargin * 12);
    ball.velocity = { x: request.direction.x * speed, y: request.direction.y * speed, z: 0 };
    ball.control = { kind: "dribble", playerId: player.playerId };
    state.touchExclusion = null;
    state.nextDribbleTouchTick = state.tick + 3;
    return "wonBall";
  }

  const skill = player.definition.abilities.technical.goalkeeping;
  const { exertion, movement, pressure } = executionFactors(state, player);
  const relativeSpeed = Math.hypot(
    ball.velocity.x - player.velocity.x, ball.velocity.y - player.velocity.y, ball.velocity.z,
  );
  const catchLimit = (6 + 14 * skill) * exertion * movement * pressure;
  state.nextDribbleTouchTick = state.tick;
  if (relativeSpeed <= catchLimit) {
    ball.position = keeperHoldPosition(player, state.config.ball.radiusM, state.config.pitch);
    ball.velocity = { ...player.velocity, z: 0 };
    ball.control = { kind: "keeperHeld", playerId: player.playerId };
    state.keeperReleaseTick = state.tick + 1 + state.config.rules.keeperHoldTicks;
    state.touchExclusion = null;
    return "caught";
  }

  // Direct a parry into the field, with a lateral component from the reach direction.
  const direction = { x: attackDirection(state, player.teamId), y: 0.8 * request.direction.y, z: 0.14 };
  const speed = Math.min(maxBallSpeed, 2 + relativeSpeed * (0.44 - 0.12 * skill));
  const scale = speed / Math.hypot(direction.x, direction.y, direction.z);
  ball.velocity = { x: direction.x * scale, y: direction.y * scale, z: direction.z * scale };
  ball.control = { kind: "free" };
  state.touchExclusion = { playerId: player.playerId, untilTick: state.tick + 5 };
  return "parried";
}
