import type { ActionRequest, PlayerState, Vec3 } from "../contracts/index.js";
import { executionFactors, footQuality } from "./abilities.js";
import { dispossessChance } from "./ball-control.js";
import { DIVE_REACH_M, keeperDiveExtension, keeperSaveTiming, keeperSaveVolumes, saveStrain } from "./keeper-save.js";
import { along, capsuleDistanceSquared, sweepCapsule } from "./physics-geometry.js";
import type { Capsule } from "./physics-geometry.js";
import { attackDirection, keeperHoldPosition } from "./reset.js";
import { randomUnit } from "./random.js";
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

/** How far a standing tackle's lunge reaches, from the tackler's centre to the ball's centre, when fresh and on the strong foot. */
export function tackleReachM(player: PlayerState): number {
  const { physical, technical } = player.definition.abilities;
  return physical.reachM * (0.95 + 0.2 * technical.tackling);
}

function tackleVolume(state: EngineState, player: PlayerState, request: DefenseRequest): Capsule {
  const { exertion } = executionFactors(state, player);
  const foot = footQuality(player, state.ball.position);
  // A tackle is a lunge, so it reaches past the standing reach; fatigue and the weaker foot shorten it.
  const reach = tackleReachM(player) * (0.85 + 0.15 * exertion * foot);
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

/** How far a save that became active at `activeTick` has stretched into its dive. */
export function saveExtension(state: EngineState, activeTick: number): number {
  return keeperDiveExtension(state.tick - activeTick);
}

/**
 * The back-pass rule: a ball a teammate deliberately kicked or threw to the keeper cannot be handled, so a save does
 * not touch it and the keeper plays it with their feet. Deflections end the flight and are fair to handle.
 */
export function isBackPass(state: EngineState, keeper: PlayerState): boolean {
  const flight = state.ball.flightAction;
  if (!flight || flight.playerId === keeper.playerId) return false;
  return state.players.some((player) => player.playerId === flight.playerId && player.teamId === keeper.teamId);
}

export function defenseCandidate(
  state: EngineState, player: PlayerState, request: DefenseRequest, duration: number, activeTick: number,
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
  const teammateHasIt = control.kind === "dribble" &&
    players.some((owner) => owner.playerId === control.playerId && owner.teamId === player.teamId);
  if (teammateHasIt) return undefined;
  if (request.kind === "keeperSave") {
    if (isBackPass(state, player)) return undefined;
    if (player.definition.role !== "goalkeeper" || !saveInterval(state, player, times)) return undefined;
    volumes = keeperSaveVolumes(player, request.direction, state.config.tuning.keeperReachScale,
      saveExtension(state, activeTick));
    sweepRadius = ballRadius;
  } else {
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
): "wonBall" | "poked" | "caught" | "parried" | "missed" {
  const { ball } = state;
  const maxBallSpeed = state.config.ball.maxSpeedMps;

  if (request.kind === "tackle") {
    const skill = player.definition.abilities.technical.tackling;
    const control = ball.control;
    if (control.kind === "dribble") {
      // The dribbler can ride the challenge: the tackle then misses and the ball stays with them.
      const owner = state.players.find((entry) => entry.playerId === control.playerId)!;
      if (randomUnit(state) >= dispossessChance(state, player, owner, 0.45)) return "missed";
    }
    ball.lastTouchPlayerId = player.playerId;
    delete ball.flightAction;
    if (control.kind === "free") {
      // A tackle at a loose ball pokes it away; collecting a loose ball is a first touch.
      const speed = Math.min(maxBallSpeed, 2.5 + 3 * skill);
      ball.velocity = { x: request.direction.x * speed, y: request.direction.y * speed, z: 0 };
      state.touchExclusion = { playerId: player.playerId, untilTick: state.tick + 4 };
      return "poked";
    }
    const distance = Math.hypot(ball.position.x - player.position.x, ball.position.y - player.position.y);
    const controlMargin = Math.max(0, player.definition.abilities.physical.reachM - distance);
    const speed = Math.min(maxBallSpeed, 0.65 + 0.65 * skill, controlMargin * 12);
    ball.velocity = { x: request.direction.x * speed, y: request.direction.y * speed, z: 0 };
    ball.control = { kind: "dribble", playerId: player.playerId };
    state.touchExclusion = null;
    state.nextDribbleTouchTick = state.tick + 3;
    return "wonBall";
  }

  ball.lastTouchPlayerId = player.playerId;
  delete ball.flightAction;
  const skill = player.definition.abilities.technical.goalkeeping;
  const { exertion, movement, pressure } = executionFactors(state, player);
  const relativeSpeed = Math.hypot(
    ball.velocity.x - player.velocity.x, ball.velocity.y - player.velocity.y, ball.velocity.z,
  );
  // A ball met at full stretch is mostly parried; one within standing reach is as easy to hold as before.
  // Across or behind the body it is hard to hold at all.
  const away = Math.hypot(ball.position.x - player.position.x, ball.position.y - player.position.y);
  const stretch = 1 - 0.45 * Math.max(0, Math.min(1, (away - player.definition.abilities.physical.reachM) / DIVE_REACH_M));
  const strain = saveStrain(player, request.direction);
  const catchLimit = (6 + 14 * skill) * exertion * movement * pressure * stretch * (1 - 0.7 * strain);
  state.nextDribbleTouchTick = state.tick;
  if (relativeSpeed <= catchLimit) {
    ball.position = keeperHoldPosition(player, state.config.ball.radiusM, state.config.pitch);
    ball.velocity = { ...player.velocity, z: 0 };
    ball.control = { kind: "keeperHeld", playerId: player.playerId };
    state.keeperReleaseTick = state.tick + 1 + state.config.rules.keeperHoldTicks;
    state.touchExclusion = null;
    return "caught";
  }

  // A save at the body blocks the ball back out; a diving save pushes it wide on the side of the dive, as keepers do,
  // rather than back into the middle. A ball met out by a post is turned around it, behind for a corner.
  const forward = attackDirection(state, player.teamId);
  const lateral = Math.abs(request.direction.y);
  const side = Math.sign(request.direction.y) || Math.sign(ball.velocity.y) || 1;
  const wide = Math.min(1, Math.abs(ball.position.y) / (state.config.pitch.goalWidthM / 2));
  const direction = { x: forward * (1 - 0.6 * lateral - 0.7 * lateral * wide), y: side * (0.3 + lateral), z: 0.2 };
  const speed = Math.min(maxBallSpeed, 2 + relativeSpeed * (0.44 - 0.12 * skill));
  const scale = speed / Math.hypot(direction.x, direction.y, direction.z);
  ball.velocity = { x: direction.x * scale, y: direction.y * scale, z: direction.z * scale };
  ball.control = { kind: "free" };
  state.touchExclusion = { playerId: player.playerId, untilTick: state.tick + 5 };
  return "parried";
}
