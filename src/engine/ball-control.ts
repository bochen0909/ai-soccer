import type { PlayerAction, PlayerId, PlayerState, Vec2, Vec3 } from "../contracts/index.js";
import type { ActionSlots, EventData } from "./actions.js";
import { executionFactors, footQuality } from "./abilities.js";
import { capsuleDistanceSquared, playerCapsule, sphereContactTime } from "./physics-geometry.js";
import { randomUnit } from "./random.js";
import { controlHeight, inDribbleReach, secondTouchBarred } from "./reset.js";
import type { EngineState } from "./reset.js";

const MAX_RECEIVING_SPEED_MPS = 19;
const CONTROL_EPSILON = 1e-7;
/** Outward speed a dribbler's body gives their own ball when they run onto it. */
const BODY_PUSH_MPS = 0.8;
/** Restitution of a ball shielded off a body, much softer than a bounce off the goal frame. */
const SHIELD_RESTITUTION = 0.3;
/** Ticks before a player whose body met an opponent's dribble can contest it again. */
const CHALLENGE_COOLDOWN_TICKS = 30;

const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));

/** Where a dribbler keeps the ball, measured from their centre. */
export function dribblePocket(player: PlayerState, ballRadiusM: number): number {
  const { radiusM, abilities } = player.definition;
  return Math.min(abilities.physical.reachM * 0.88, radiusM + ballRadiusM + 0.16);
}

/**
 * Chance that `challenger` takes the ball from `owner`'s dribble. Tackling skill against dribbling skill; a heavy
 * touch (the ball out beyond the owner's pocket) makes it much easier, and a shielded ball (the owner's body between
 * the challenger and the ball) much harder. `base` is the even-skill chance at the pocket, side on.
 */
export function dispossessChance(state: EngineState, challenger: PlayerState, owner: PlayerState, base: number): number {
  const { ball } = state;
  const ballRadius = state.config.ball.radiusM;
  const tackle = challenger.definition.abilities.technical.tackling * executionFactors(state, challenger).exertion;
  const dribble = owner.definition.abilities.technical.dribbling * executionFactors(state, owner).exertion *
    footQuality(owner, ball.position);
  const pocket = dribblePocket(owner, ballRadius);
  const reach = owner.definition.abilities.physical.reachM;
  const away = Math.hypot(ball.position.x - owner.position.x, ball.position.y - owner.position.y);
  const loose = clamp((away - pocket) / Math.max(CONTROL_EPSILON, reach - pocket), 0, 1);
  const fromChallenger = { x: owner.position.x - challenger.position.x, y: owner.position.y - challenger.position.y };
  const apart = Math.hypot(fromChallenger.x, fromChallenger.y);
  const shielded = apart > CONTROL_EPSILON && away > CONTROL_EPSILON
    ? clamp(((ball.position.x - owner.position.x) * fromChallenger.x + (ball.position.y - owner.position.y) * fromChallenger.y) /
      (away * apart), 0, 1) : 0;
  return clamp(base + 0.45 * tackle - 0.4 * dribble + 0.5 * loose - 0.4 * shielded, 0, 1);
}

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
  // A ball that just squirted out from between two players is clear of both for a moment.
  if (ball.control.kind !== "free" || (state.bodyClearUntilTick[player.playerId] ?? 0) > tick ||
      secondTouchBarred(state, player) ||
      (touchExclusion?.playerId === player.playerId && tick < touchExclusion.untilTick)) return undefined;

  const ballRadius = state.config.ball.radiusM;
  const { radiusM, abilities } = player.definition;
  const reach = Math.min(abilities.physical.reachM, radiusM + ballRadius + 0.35);
  const maxHeight = Math.max(ballRadius, controlHeight(player));
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
  // A ball taken on the thigh or chest is harder to kill than one on the ground.
  const ballRadius = state.config.ball.radiusM;
  const height = clamp((ball.position.z - ballRadius) / Math.max(CONTROL_EPSILON, controlHeight(player) - ballRadius), 0, 1);
  const cleanSpeed = (5.5 + 12 * skill) * factors.exertion * factors.movement * factors.pressure * foot * (1 - 0.3 * height);
  const incomingSpeed = Math.hypot(ball.velocity.x - player.velocity.x,
    ball.velocity.y - player.velocity.y, ball.velocity.z);
  // Above the clean speed control gets less likely rather than impossible; at 1.5 times it always fails.
  const chance = incomingSpeed <= cleanSpeed ? 1 : clamp(1 - (incomingSpeed - cleanSpeed) / (0.5 * cleanSpeed), 0, 1);
  const received = chance >= 1 || randomUnit(state) < chance;
  const angleError = (received ? 0.12 : 0.38) * (1 - 0.7 * skill) * (1 + height);
  const direction = rotate(outwardDirection(player, ball.position), (2 * randomUnit(state) - 1) * angleError);
  let speed = received ? Math.min(1, incomingSpeed * (0.03 + 0.06 * (1 - skill)))
    : Math.min(8, 1.5 + (incomingSpeed - cleanSpeed) * 0.65);
  if (received && ball.position.z > ballRadius + 0.05) {
    // A cushioned ball drops toward the feet: it lands in the dribble pocket, give or take the touch error.
    const fall = Math.sqrt(2 * (ball.position.z - ballRadius) / state.config.ball.gravityMps2);
    const away = Math.hypot(ball.position.x - player.position.x, ball.position.y - player.position.y);
    speed = (dribblePocket(player, ballRadius) - away) / Math.max(fall, CONTROL_EPSILON) + speed * 0.3;
  }
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
  // A ball still dropping from a chest or thigh trap is touched once it reaches the feet.
  if (!inDribbleReach(player, ball.position, ballRadius) || ball.position.z > ballRadius + 0.05) return;

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
  const pocket = dribblePocket(player, ballRadius);
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

function excludedFromTouch(state: EngineState, player: PlayerState): boolean {
  const exclusion = state.touchExclusion;
  return exclusion?.playerId === player.playerId && state.tick < exclusion.untilTick;
}

/** A soft bounce off a body that does not take the ball from whoever is dribbling it. */
function shield(state: EngineState, player: PlayerState, normal: Vec3, events: EventData[]): void {
  const { ball } = state;
  const normalSpeed = (ball.velocity.x - player.velocity.x) * normal.x + (ball.velocity.y - player.velocity.y) * normal.y +
    ball.velocity.z * normal.z;
  if (normalSpeed < 0) {
    const impulse = -(1 + SHIELD_RESTITUTION) * normalSpeed;
    ball.velocity = {
      x: ball.velocity.x + normal.x * impulse, y: ball.velocity.y + normal.y * impulse, z: ball.velocity.z + normal.z * impulse,
    };
  }
  ball.lastTouchPlayerId = player.playerId;
  const flightAction = ball.flightAction;
  delete ball.flightAction;
  events.push({ kind: "ballTouch", playerId: player.playerId, outcome: "deflected", position: { ...ball.position }, flightAction });
}

/**
 * A ball low and slow enough to control meets a player's body. The dribbler's own body keeps the ball with a soft
 * touch, a loose ball is trapped (a first touch), and an opponent's dribble is contested on skill, at most once per
 * cooldown; a lost contest shields the ball off the challenger's body. Returns false for a plain bounce: a ball too
 * high or too fast, or a loose ball the player cannot touch just now.
 */
export function resolveBodyContact(
  state: EngineState, player: PlayerState, normal: Vec3, action: PlayerAction | undefined, slots: ActionSlots,
  events: EventData[],
): boolean {
  const { ball } = state;
  if (ball.position.z > controlHeight(player)) return false;
  const relative = Math.hypot(ball.velocity.x - player.velocity.x, ball.velocity.y - player.velocity.y, ball.velocity.z);
  if (relative > MAX_RECEIVING_SPEED_MPS) return false;
  const control = ball.control;

  if (control.kind === "free") {
    if (excludedFromTouch(state, player) || slots.get(player.playerId)?.phase === "windup" ||
        secondTouchBarred(state, player)) return false;
    resolveFirstTouch(state, player, action, events);
    return true;
  }
  if (control.kind !== "dribble") return false;
  if (control.playerId === player.playerId) {
    const flat = Math.hypot(normal.x, normal.y) || 1;
    ball.velocity = {
      x: player.velocity.x + (normal.x / flat) * BODY_PUSH_MPS,
      y: player.velocity.y + (normal.y / flat) * BODY_PUSH_MPS,
      z: 0,
    };
    events.push({ kind: "ballTouch", playerId: player.playerId, outcome: "dribbled", position: { ...ball.position } });
    return true;
  }

  const owner = state.players.find((entry) => entry.playerId === control.playerId)!;
  const ready = (state.challengeReadyTick[player.playerId] ?? 0) <= state.tick;
  if (owner.teamId !== player.teamId && ready && !excludedFromTouch(state, player)) {
    state.challengeReadyTick[player.playerId] = state.tick + CHALLENGE_COOLDOWN_TICKS;
    if (randomUnit(state) < dispossessChance(state, player, owner, 0.2)) {
      events.push({ kind: "possessionLost", playerId: owner.playerId, position: { ...ball.position } });
      ball.control = { kind: "free" };
      resolveFirstTouch(state, player, action, events);
      return true;
    }
  }
  shield(state, player, normal, events);
  return true;
}

/** Ticks the ball passes through both bodies after squirting out from between them. */
const SQUIRT_CLEAR_TICKS = 6;
const SQUIRT_MIN_SPEED_MPS = 3;

/**
 * A ball pinched between two bodies cannot stay there: it squirts out sideways and is loose. Without this the two
 * soft contacts trade the ball back and forth in zero time. Returns false when no other body is against the ball.
 */
export function resolvePinch(state: EngineState, player: PlayerState, events: EventData[]): boolean {
  const { ball } = state;
  const ballRadius = state.config.ball.radiusM;
  const other = state.players.find((candidate) => candidate !== player &&
    (state.bodyClearUntilTick[candidate.playerId] ?? 0) <= state.tick &&
    capsuleDistanceSquared(ball.position, playerCapsule(candidate)) <= (candidate.definition.radiusM + ballRadius + 0.01) ** 2);
  if (!other) return false;
  const dx = other.position.x - player.position.x, dy = other.position.y - player.position.y;
  const gap = Math.hypot(dx, dy) || 1;
  const line = { x: dx / gap, y: dy / gap };
  const across = { x: -line.y, y: line.x };
  const mid = { x: (player.position.x + other.position.x) / 2, y: (player.position.y + other.position.y) / 2 };
  const offset = (ball.position.x - mid.x) * across.x + (ball.position.y - mid.y) * across.y;
  const side = Math.abs(offset) > 1e-3 ? Math.sign(offset) : randomUnit(state) < 0.5 ? -1 : 1;
  const closing = Math.max(0, -((other.velocity.x - player.velocity.x) * line.x + (other.velocity.y - player.velocity.y) * line.y));
  const speed = Math.min(8, SQUIRT_MIN_SPEED_MPS + 0.6 * closing);
  ball.velocity = {
    x: (player.velocity.x + other.velocity.x) / 2 + across.x * side * speed,
    y: (player.velocity.y + other.velocity.y) / 2 + across.y * side * speed,
    z: Math.max(0, ball.velocity.z),
  };
  if (ball.control.kind === "dribble") {
    events.push({ kind: "possessionLost", playerId: ball.control.playerId, position: { ...ball.position } });
    ball.control = { kind: "free" };
    state.nextDribbleTouchTick = 0;
  }
  for (const body of [player, other]) state.bodyClearUntilTick[body.playerId] = state.tick + SQUIRT_CLEAR_TICKS;
  ball.lastTouchPlayerId = player.playerId;
  const flightAction = ball.flightAction;
  delete ball.flightAction;
  events.push({ kind: "ballTouch", playerId: player.playerId, outcome: "deflected", position: { ...ball.position }, flightAction });
  return true;
}
