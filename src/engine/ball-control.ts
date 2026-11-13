import type { PlayerAction, PlayerId, PlayerState, Vec2 } from "../contracts/index.js";
import type { EventData } from "./actions.js";
import { executionFactors, footQuality } from "./abilities.js";
import { sphereContactTime } from "./physics-geometry.js";
import { randomUnit } from "./random.js";
import { inDribbleReach } from "./reset.js";
import type { EngineState } from "./reset.js";

const MAX_RECEIVING_SPEED_MPS = 19;
const CONTROL_EPSILON = 1e-7;

function rotate(direction: Vec2, angle: number): Vec2 {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  return { x: direction.x * cos - direction.y * sin, y: direction.x * sin + direction.y * cos };
}

function outwardDirection(player: PlayerState, position: Vec2): Vec2 {
  const x = position.x - player.position.x;
  const y = position.y - player.position.y;
  const distance = Math.hypot(x, y);
  return distance > CONTROL_EPSILON
    ? { x: x / distance, y: y / distance }
    : { x: Math.cos(player.facingRad), y: Math.sin(player.facingRad) };
}

/** Sweeps the ball's ground-level control reach relative to the moving player. */
export function firstTouchCandidate(
  state: EngineState, player: PlayerState, action: PlayerAction | undefined, duration: number,
): { time: number; player: PlayerState } | undefined {
  const { ball, tick, touchExclusion } = state;
  if (ball.control.kind !== "free" ||
      (touchExclusion?.playerId === player.playerId && tick < touchExclusion.untilTick)) return undefined;

  const ballRadius = state.config.ball.radiusM;
  const { radiusM, abilities } = player.definition;
  const reach = Math.min(abilities.physical.reachM, radiusM + ballRadius + 0.35);
  const maxHeight = ballRadius + Math.min(radiusM, 0.3);
  const velocity = {
    x: ball.velocity.x - player.velocity.x,
    y: ball.velocity.y - player.velocity.y,
    z: ball.velocity.z,
  };
  if (Math.hypot(velocity.x, velocity.y, velocity.z) > MAX_RECEIVING_SPEED_MPS) return undefined;

  const entersHeight = ball.position.z > maxHeight && velocity.z < 0;
  const startTime = entersHeight ? (maxHeight - ball.position.z) / velocity.z : 0;
  if (startTime > duration || (ball.position.z > maxHeight && !entersHeight)) return undefined;

  const offset = {
    x: ball.position.x - player.position.x + velocity.x * startTime,
    y: ball.position.y - player.position.y + velocity.y * startTime,
    z: 0,
  };
  const distance = Math.hypot(offset.x, offset.y);
  const closing = offset.x * velocity.x + offset.y * velocity.y < -CONTROL_EPSILON;
  const intentionalPickup = Math.hypot(ball.velocity.x, ball.velocity.y, ball.velocity.z) < 0.5 &&
    action !== undefined && action.movement.x * offset.x + action.movement.y * offset.y > CONTROL_EPSILON;

  let time = startTime;
  if (distance <= reach + CONTROL_EPSILON) {
    if (!closing && !intentionalPickup && velocity.z >= -CONTROL_EPSILON) return undefined;
  } else {
    const entry = sphereContactTime(offset, { ...velocity, z: 0 }, reach);
    if (entry === undefined) return undefined;
    time += entry;
  }
  if (time > duration || ball.position.z + velocity.z * time > maxHeight + CONTROL_EPSILON) return undefined;
  return { time, player };
}

function touchInterval(skill: number, sprint: boolean): number {
  return Math.round(10 - 5 * skill) + (sprint ? 3 : 0);
}

export function resolveFirstTouch(
  state: EngineState, player: PlayerState, action: PlayerAction | undefined, events: EventData[],
): void {
  const { ball } = state;
  const factors = executionFactors(state, player);
  const skill = player.definition.abilities.technical.firstTouch;
  const foot = footQuality(player, ball.position);
  const cleanSpeed = (4.5 + 12 * skill) * factors.exertion * factors.movement * factors.pressure * foot;
  const incomingSpeed = Math.hypot(ball.velocity.x - player.velocity.x,
    ball.velocity.y - player.velocity.y, ball.velocity.z);
  const received = incomingSpeed <= cleanSpeed;
  const angleError = (received ? 0.12 : 0.38) * (1 - 0.7 * skill);
  const direction = rotate(outwardDirection(player, ball.position), (2 * randomUnit(state) - 1) * angleError);
  const speed = received ? Math.min(1, incomingSpeed * (0.03 + 0.06 * (1 - skill)))
    : Math.min(8, 1.5 + (incomingSpeed - cleanSpeed) * 0.65);
  const velocity = { x: player.velocity.x + direction.x * speed, y: player.velocity.y + direction.y * speed };
  const scale = Math.min(1, state.config.ball.maxSpeedMps / Math.max(CONTROL_EPSILON, Math.hypot(velocity.x, velocity.y)));
  const flightAction = ball.flightAction;
  ball.velocity = { x: velocity.x * scale, y: velocity.y * scale, z: 0 };
  ball.lastTouchPlayerId = player.playerId;
  delete ball.flightAction;
  ball.control = received ? { kind: "dribble", playerId: player.playerId } : { kind: "free" };
  if (received) {
    state.nextDribbleTouchTick = state.tick + touchInterval(player.definition.abilities.technical.dribbling, action?.sprint === true);
  } else {
    state.touchExclusion = { playerId: player.playerId, untilTick: state.tick + 6 };
  }
  events.push({
    kind: "ballTouch", playerId: player.playerId, outcome: received ? "received" : "miscontrolled",
    position: { ...ball.position }, flightAction,
  });
}

/** Applies a foot touch at most once per cadence, then lets the ball roll. */
export function applyDribbleTouch(
  state: EngineState, actions: ReadonlyMap<PlayerId, PlayerAction>, events: EventData[],
): void {
  const { ball } = state;
  if (ball.control.kind !== "dribble" || state.tick < state.nextDribbleTouchTick) return;
  const holderId = ball.control.playerId;
  const player = state.players.find((entry) => entry.playerId === holderId)!;
  const ballRadius = state.config.ball.radiusM;
  if (!inDribbleReach(player, ball.position, ballRadius)) return;

  const action = actions.get(holderId);
  const movement = action?.movement ?? { x: 0, y: 0 };
  const input = Math.hypot(movement.x, movement.y);
  const moving = input > CONTROL_EPSILON;
  const rolling = Math.hypot(ball.velocity.x, ball.velocity.y) > 0.3;
  if (!moving && !rolling) return;

  const sprint = action?.sprint === true;
  const factors = executionFactors(state, player);
  const foot = footQuality(player, ball.position);
  const skill = player.definition.abilities.technical.dribbling * factors.exertion * factors.pressure * foot;
  const interval = touchInterval(skill, sprint);
  const outward = outwardDirection(player, ball.position);
  const direction = moving ? { x: movement.x / input, y: movement.y / input } : outward;
  const { radiusM, abilities } = player.definition;
  const pocket = Math.min(abilities.physical.reachM * 0.88, radiusM + ballRadius + 0.16);
  const ahead = (0.15 + (1 - skill) * 0.6 + (sprint ? 1.3 : 0)) * input;
  const angleError = (0.025 + (1 - skill) * 0.32 + (sprint ? 0.1 : 0)) * input;
  const aim = rotate(direction, (2 * randomUnit(state) - 1) * angleError);
  const correction = moving ? 6 : 0;
  let x = player.velocity.x + aim.x * ahead + correction * (player.position.x + aim.x * pocket - ball.position.x);
  let y = player.velocity.y + aim.y * ahead + correction * (player.position.y + aim.y * pocket - ball.position.y);

  // Keep the touch from sending the ball through the player's body during a turn.
  const radialSpeed = (x - player.velocity.x) * outward.x + (y - player.velocity.y) * outward.y;
  const distance = Math.hypot(ball.position.x - player.position.x, ball.position.y - player.position.y);
  const minRadialSpeed = -Math.max(0, distance - radiusM - ballRadius) * correction;
  if (radialSpeed < minRadialSpeed) {
    x += (minRadialSpeed - radialSpeed) * outward.x;
    y += (minRadialSpeed - radialSpeed) * outward.y;
  }
  const maxSpeed = Math.min(state.config.ball.maxSpeedMps, abilities.physical.maxSpeedMps * 1.35);
  const scale = Math.min(1, maxSpeed / Math.max(CONTROL_EPSILON, Math.hypot(x, y)));
  ball.velocity = { x: x * scale, y: y * scale, z: 0 };
  ball.lastTouchPlayerId = holderId;
  state.nextDribbleTouchTick = state.tick + interval;
  events.push({ kind: "ballTouch", playerId: holderId, outcome: "dribbled", position: { ...ball.position } });
}

export function updateDribbleControl(state: EngineState, events: EventData[]): void {
  const { ball } = state;
  if (ball.control.kind !== "dribble") return;
  const holderId = ball.control.playerId;
  const player = state.players.find((entry) => entry.playerId === holderId)!;
  if (inDribbleReach(player, ball.position, state.config.ball.radiusM)) return;

  ball.control = { kind: "free" };
  state.touchExclusion = { playerId: holderId, untilTick: state.tick + 3 };
  events.push({ kind: "possessionLost", playerId: holderId, position: { ...ball.position } });
}
