import type { ActionResult, PlayerAction, PlayerId, PlayerState, Vec2, Vec3 } from "../contracts/index.js";
import { dueLaunch, finishDefenseContact, interruptAction, resolveLaunch } from "./actions.js";
import type { ActionSlots, EventData, PendingAction } from "./actions.js";
import { applyDribbleTouch, firstTouchCandidate, resolveFirstTouch, updateDribbleControl } from "./ball-control.js";
import { defenseCandidate, isDefenseRequest, resolveDefense } from "./defense.js";
import type { DefenseRequest } from "./defense.js";
import { randomUnit } from "./random.js";
import {
  CONTACT_TIME_TOLERANCE,
  along,
  dot,
  goalFrameCapsules,
  playerCapsule,
  sphereContactTime,
  subtract,
  sweepCapsule,
} from "./physics-geometry.js";
import type { Capsule } from "./physics-geometry.js";
import { attackDirection, keeperHoldPosition } from "./reset.js";
import type { EngineState } from "./reset.js";

export const PHYSICS_TICK_SECONDS = 1 / 60;
const SUBSTEPS = 4;
const MAX_EVENTS_PER_SUBSTEP = 128;
const WALL_PIN_TOLERANCE = 1e-6;
const WALL_PUSH_MINIMUM = 0.05;
const CONTACT_GROUP_MARGIN_M = 0.01;
const VELOCITY_TOLERANCE = 1e-8;
const MINIMUM_BOUNCE_SPEED = 0.3;

type Axis = "x" | "y";
type Side = -1 | 1;

interface BallCrossing {
  kind: "goal" | "exit";
  axis: Axis;
  side: Side;
  position: Vec3;
}

export interface PhysicsOutcome {
  elapsedSeconds: number;
  crossing?: BallCrossing;
}

export class PhysicsError extends Error {
  constructor(tick: number) {
    super(`Physics exceeded its collision event limit at tick ${tick}.`);
    this.name = "PhysicsError";
  }
}

type PhysicsEvent = { time: number } & (
  | { kind: "crossing"; axis: Axis; side: Side; plane: number }
  | { kind: "ground" }
  | { kind: "ballContact"; normal: Vec3; collider: Capsule | PlayerState; player?: PlayerState }
  | { kind: "wall"; player: PlayerState; axis: Axis; plane: number }
  | { kind: "playerContact"; first: PlayerState; second: PlayerState }
  | { kind: "firstTouch"; player: PlayerState }
  | { kind: "defense"; player: PlayerState; pending: PendingAction; request: DefenseRequest }
  | { kind: "launch"; player: PlayerState }
);

interface PlayerBounds {
  min: Vec2;
  max: Vec2;
}

interface TickPlay {
  actions: ReadonlyMap<PlayerId, PlayerAction>;
  slots: ActionSlots;
  events: EventData[];
  results: ActionResult[];
}

export function advancePhysics(state: EngineState, play: TickPlay): PhysicsOutcome {
  const { actions, slots, events, results } = play;
  const goalFrames = goalFrameCapsules(state.config.pitch);
  const substepSeconds = PHYSICS_TICK_SECONDS / SUBSTEPS;
  let elapsedSeconds = 0;
  let firstTouchAvailable = true;

  updateDribbleControl(state, events);
  const control = state.ball.control;
  if (control.kind === "dribble" && slots.get(control.playerId)?.phase !== "windup") {
    applyDribbleTouch(state, actions, events);
  }

  for (let substep = 0; substep < SUBSTEPS; substep++) {
    for (const player of state.players) {
      applyMovement(player, actions.get(player.playerId), substepSeconds);
    }
    applyBallForces(state, substepSeconds);

    let remaining = substepSeconds;
    let eventCount = 0;
    // Players in zero-time wall and player contacts. A pile of overlapping bodies pressed against a wall
    // can trade impulses forever without advancing time, so a second overrun is the only real failure.
    const jammed = new Set<PlayerState>();
    let settled = false;
    const ballContacts = new Set<Capsule | PlayerState>();
    while (true) {
      const event = findNextEvent(state, goalFrames, remaining, play, firstTouchAvailable, ballContacts);
      const duration = event === undefined ? remaining : Math.min(event.time, remaining);
      moveBodies(state, duration);
      elapsedSeconds += duration;
      remaining = Math.max(0, remaining - duration);
      if (event === undefined) break;
      if (duration <= CONTACT_TIME_TOLERANCE) {
        if (event.kind === "wall") jammed.add(event.player);
        else if (event.kind === "playerContact") jammed.add(event.first).add(event.second);
      }
      if (++eventCount > MAX_EVENTS_PER_SUBSTEP) {
        if (settled || jammed.size === 0) throw new PhysicsError(state.tick);
        // Treat the pile-up as perfectly inelastic: everyone touching it stops, not only the players that
        // happened to trade impulses lately, or the rest of a long row would restart the chain.
        for (const { velocity } of contactGroup(state, jammed)) {
          velocity.x = 0;
          velocity.y = 0;
        }
        jammed.clear();
        settled = true;
        eventCount = 0;
        continue;
      }
      if (duration > CONTACT_TIME_TOLERANCE) ballContacts.clear();
      if (event.kind === "ballContact") ballContacts.add(event.collider);

      if (event.kind === "crossing") {
        state.ball.position[event.axis] = event.plane;
        updatePlayerState(state, actions, elapsedSeconds);
        updateDribbleControl(state, events);
        return { elapsedSeconds, crossing: classifyCrossing(state, event.axis, event.side) };
      }
      switch (event.kind) {
        case "firstTouch":
          firstTouchAvailable = false;
          resolveFirstTouch(state, event.player, actions.get(event.player.playerId), events);
          break;
        case "defense": {
          const { pending, request } = event;
          const previous = state.ball.control;
          const flight = state.ball.flightAction;
          const position = { ...state.ball.position };
          const outcome = resolveDefense(state, event.player, request);
          finishDefenseContact(state, slots, pending, events, outcome, position, flight);
          if (previous.kind === "dribble" && previous.playerId !== event.player.playerId) {
            events.push({ kind: "possessionLost", playerId: previous.playerId, position });
            interruptAction(state, slots, previous.playerId, results,
              request.kind === "tackle" ? "tackled" : "ballUnavailable");
          }
          firstTouchAvailable = false;
          break;
        }
        case "launch":
          resolveLaunch(state, slots, event.player, events, results);
          break;
        default:
          resolveContact(state, event, events);
      }
      updateDribbleControl(state, events);
    }
    updateDribbleControl(state, events);
  }

  updatePlayerState(state, actions, PHYSICS_TICK_SECONDS);
  return { elapsedSeconds: PHYSICS_TICK_SECONDS };
}

const eventPriority: Record<PhysicsEvent["kind"], number> = {
  wall: 0,
  ground: 0,
  ballContact: 0,
  playerContact: 0,
  defense: 1,
  launch: 2,
  firstTouch: 3,
  crossing: 4,
};

function applyMovement(player: PlayerState, action: PlayerAction | undefined, seconds: number): void {
  const { maxSpeedMps, accelerationMps2, brakingMps2 } = player.definition.abilities.physical;
  const movement = action?.movement ?? { x: 0, y: 0 };
  const speedLimit = maxSpeedMps * (0.75 + player.stamina * 0.25);
  const targetSpeed = speedLimit * (action?.sprint ? 1 : 0.7);
  const target = { x: movement.x * targetSpeed, y: movement.y * targetSpeed };
  const difference = { x: target.x - player.velocity.x, y: target.y - player.velocity.y };
  const distance = Math.hypot(difference.x, difference.y);
  if (distance === 0) return;
  const rate = difference.x * player.velocity.x + difference.y * player.velocity.y < 0
    ? brakingMps2 : accelerationMps2;
  const fraction = Math.min(1, rate * seconds / distance);
  player.velocity.x += difference.x * fraction;
  player.velocity.y += difference.y * fraction;
}

function movementBounds(state: EngineState, player: PlayerState): PlayerBounds {
  const { lengthM, widthM, keeperAreaDepthM, keeperAreaWidthM } = state.config.pitch;
  const { radiusM } = player.definition;
  const min = { x: -lengthM / 2 + radiusM, y: -widthM / 2 + radiusM };
  const max = { x: lengthM / 2 - radiusM, y: widthM / 2 - radiusM };
  if (state.ball.control.kind === "keeperHeld" && state.ball.control.playerId === player.playerId) {
    const defendsNegativeX = attackDirection(state, player.teamId) === 1;
    if (defendsNegativeX) max.x = Math.min(max.x, -lengthM / 2 + keeperAreaDepthM);
    else min.x = Math.max(min.x, lengthM / 2 - keeperAreaDepthM);
    min.y = Math.max(min.y, -keeperAreaWidthM / 2);
    max.y = Math.min(max.y, keeperAreaWidthM / 2);
  }
  // A taker may begin just outside the field. They can reenter without teleportation.
  min.x = Math.min(min.x, player.position.x);
  min.y = Math.min(min.y, player.position.y);
  max.x = Math.max(max.x, player.position.x);
  max.y = Math.max(max.y, player.position.y);
  return { min, max };
}

function applyBallForces(state: EngineState, seconds: number): void {
  if (state.ball.control.kind === "keeperHeld") return;
  const { radiusM, gravityMps2, groundDecelerationMps2, maxSpeedMps } = state.config.ball;
  const { position, velocity } = state.ball;
  if (position.z <= radiusM && velocity.z <= 0) {
    if (velocity.z < 0) return;
    const speed = Math.hypot(velocity.x, velocity.y);
    if (speed > 0) {
      const factor = Math.max(0, speed - groundDecelerationMps2 * seconds) / speed;
      velocity.x *= factor;
      velocity.y *= factor;
    }
  } else {
    velocity.z -= gravityMps2 * seconds;
    limitBallSpeed(velocity, maxSpeedMps);
  }
}

function findNextEvent(
  state: EngineState,
  goalFrames: readonly Capsule[],
  duration: number,
  play: TickPlay,
  firstTouchAvailable: boolean,
  ballContacts: ReadonlySet<Capsule | PlayerState>,
): PhysicsEvent | undefined {
  let first: PhysicsEvent | undefined;
  let tied: PhysicsEvent[] = [];
  const consider = (event: PhysicsEvent): void => {
    if (event.time > duration) return;
    const simultaneous = first !== undefined && Math.abs(event.time - first.time) <= CONTACT_TIME_TOLERANCE;
    if (first === undefined || event.time < first.time - CONTACT_TIME_TOLERANCE ||
        (simultaneous && eventPriority[event.kind] < eventPriority[first.kind])) {
      first = event;
      tied = [];
    } else if (simultaneous && (event.kind === "firstTouch" || event.kind === "defense") &&
        first.kind === event.kind) {
      tied.push(event);
    }
  };

  for (const [index, player] of state.players.entries()) {
    const limit = movementBounds(state, player);
    for (const axis of ["x", "y"] as const) {
      const velocity = player.velocity[axis];
      if (Math.abs(velocity) <= VELOCITY_TOLERANCE) continue;
      const plane = velocity > 0 ? limit.max[axis] : limit.min[axis];
      const time = Math.max(0, (plane - player.position[axis]) / velocity);
      consider({ kind: "wall", time, player, axis, plane });
    }
    for (let other = index + 1; other < state.players.length; other++) {
      const second = state.players[other]!;
      const offset = subtract(onGround(player.position), onGround(second.position));
      const velocity = subtract(onGround(player.velocity), onGround(second.velocity));
      if (dot(offset, velocity) >= -VELOCITY_TOLERANCE) continue;
      const time = sphereContactTime(offset, velocity, player.definition.radiusM + second.definition.radiusM);
      if (time !== undefined) consider({ kind: "playerContact", time, first: player, second });
    }
  }

  for (const player of state.players) {
    if (dueLaunch(state, play.slots.get(player.playerId))) consider({ time: 0, kind: "launch", player });
  }
  if (state.ball.control.kind === "keeperHeld") return first;
  const { position, velocity } = state.ball;
  const { radiusM } = state.config.ball;
  const { lengthM, widthM } = state.config.pitch;
  if (velocity.z < 0) {
    consider({ kind: "ground", time: Math.max(0, (radiusM - position.z) / velocity.z) });
  }

  for (const frame of goalFrames) {
    const hit = sweepCapsule(position, velocity, radiusM, frame, duration);
    if (hit && (hit.time > CONTACT_TIME_TOLERANCE || !ballContacts.has(frame))) {
      consider({ kind: "ballContact", ...hit, collider: frame });
    }
  }
  for (const player of state.players) {
    const relativeVelocity = subtract(velocity, onGround(player.velocity));
    const hit = sweepCapsule(position, relativeVelocity, radiusM, playerCapsule(player), duration);
    // A ball pinched against the ground can overlap several bodies at the same instant.
    // Apply each collider's impulse once until the bodies have advanced in time.
    if (hit && (hit.time > CONTACT_TIME_TOLERANCE || !ballContacts.has(player))) {
      consider({ kind: "ballContact", ...hit, collider: player, player });
    }
    const pending = play.slots.get(player.playerId);
    if (pending?.phase === "active" && isDefenseRequest(pending.request)) {
      const contact = defenseCandidate(state, player, pending.request, duration);
      if (contact) consider({ time: contact.time, kind: "defense", player, pending, request: pending.request });
    } else if (firstTouchAvailable && pending?.phase !== "windup") {
      const contact = firstTouchCandidate(state, player, play.actions.get(player.playerId), duration);
      if (contact) consider({ time: contact.time, kind: "firstTouch", player });
    }
  }

  for (const axis of ["x", "y"] as const) {
    if (velocity[axis] === 0) continue;
    const side = velocity[axis] > 0 ? 1 : -1;
    const plane = side * ((axis === "x" ? lengthM : widthM) / 2 + radiusM);
    const time = Math.max(0, (plane - position[axis]) / velocity[axis]);
    consider({ kind: "crossing", time, axis, side, plane });
  }
  if (first && tied.length > 0) {
    const candidates = [first, ...tied];
    return candidates[Math.floor(randomUnit(state) * candidates.length)];
  }
  return first;
}

function onGround(vector: Vec2): Vec3 {
  return { ...vector, z: 0 };
}

function moveBodies(state: EngineState, seconds: number): void {
  for (const player of state.players) {
    player.position.x += player.velocity.x * seconds;
    player.position.y += player.velocity.y * seconds;
  }
  if (state.ball.control.kind !== "keeperHeld") {
    state.ball.position = along(state.ball.position, state.ball.velocity, seconds);
  }
}

function classifyCrossing(state: EngineState, axis: Axis, side: Side): BallCrossing {
  const { goalWidthM, goalHeightM } = state.config.pitch;
  const { radiusM } = state.config.ball;
  const { y, z } = state.ball.position;
  const fitsOpening = Math.abs(y) + radiusM <= goalWidthM / 2 && z + radiusM <= goalHeightM;
  return { kind: axis === "x" && fitsOpening ? "goal" : "exit", axis, side, position: { ...state.ball.position } };
}

function recordControlLoss(state: EngineState, events: EventData[], playerId: PlayerId): void {
  events.push({ kind: "possessionLost", playerId, position: { ...state.ball.position } });
}

function resolveContact(
  state: EngineState,
  event: Exclude<PhysicsEvent, { kind: "crossing" | "firstTouch" | "defense" | "launch" }>,
  events: EventData[],
): void {
  switch (event.kind) {
    case "wall":
      event.player.position[event.axis] = event.plane;
      event.player.velocity[event.axis] = 0;
      break;
    case "ground": {
      state.ball.position.z = state.config.ball.radiusM;
      const bounce = -state.ball.velocity.z * state.config.ball.groundRestitution;
      state.ball.velocity.z = bounce >= MINIMUM_BOUNCE_SPEED ? bounce : 0;
      break;
    }
    case "playerContact":
      resolvePlayerContact(state, event.first, event.second);
      break;
    case "ballContact": {
      const colliderVelocity = onGround(event.player?.velocity ?? { x: 0, y: 0 });
      const normalSpeed = dot(subtract(state.ball.velocity, colliderVelocity), event.normal);
      const impulseSpeed = -(1 + state.config.ball.contactRestitution) * normalSpeed;
      state.ball.velocity = along(state.ball.velocity, event.normal, impulseSpeed);
      limitBallSpeed(state.ball.velocity, state.config.ball.maxSpeedMps);
      if (event.player !== undefined) {
        state.ball.lastTouchPlayerId = event.player.playerId;
        events.push({
          kind: "ballTouch", playerId: event.player.playerId, outcome: "deflected",
          position: { ...state.ball.position }, flightAction: state.ball.flightAction,
        });
        delete state.ball.flightAction;
        if (state.ball.control.kind === "dribble" && state.ball.control.playerId !== event.player.playerId) {
          recordControlLoss(state, events, state.ball.control.playerId);
          state.ball.control = { kind: "free" };
          state.nextDribbleTouchTick = 0;
        }
      }
      break;
    }
  }
}

/** The seed players plus everyone connected to them through touching or overlapping bodies. */
function contactGroup(state: EngineState, seeds: ReadonlySet<PlayerState>): Set<PlayerState> {
  const group = new Set(seeds);
  const queue = [...seeds];
  for (let current = queue.pop(); current !== undefined; current = queue.pop()) {
    for (const other of state.players) {
      if (group.has(other)) continue;
      const reach = current.definition.radiusM + other.definition.radiusM + CONTACT_GROUP_MARGIN_M;
      if (Math.hypot(current.position.x - other.position.x, current.position.y - other.position.y) <= reach) {
        group.add(other);
        queue.push(other);
      }
    }
  }
  return group;
}

/** Whether a push along `push` drives the player into a movement limit they already stand on. */
function pinnedAgainstWall(state: EngineState, player: PlayerState, push: Vec3): boolean {
  const { min, max } = movementBounds(state, player);
  const { x, y } = player.position;
  // Contacts along a wall carry a tiny normal component across it from rounding. That is not a push into the wall.
  return (push.x > WALL_PUSH_MINIMUM && x >= max.x - WALL_PIN_TOLERANCE) ||
    (push.x < -WALL_PUSH_MINIMUM && x <= min.x + WALL_PIN_TOLERANCE) ||
    (push.y > WALL_PUSH_MINIMUM && y >= max.y - WALL_PIN_TOLERANCE) ||
    (push.y < -WALL_PUSH_MINIMUM && y <= min.y + WALL_PIN_TOLERANCE);
}

function resolvePlayerContact(state: EngineState, first: PlayerState, second: PlayerState): void {
  const separation = subtract(onGround(first.position), onGround(second.position));
  const distance = Math.hypot(separation.x, separation.y);
  const normal = { x: separation.x / distance, y: separation.y / distance, z: 0 };
  const relativeVelocity = subtract(onGround(first.velocity), onGround(second.velocity));
  // A player pinned against a wall cannot move away from it, so the wall takes their share of the impulse.
  // Without this, a pile of players against a wall passes ever smaller impulses back and forth without end.
  // If both are pinned (diagonal contact between players on different walls), the walls cannot absorb it all.
  const firstPinned = pinnedAgainstWall(state, first, normal);
  const secondPinned = pinnedAgainstWall(state, second, { x: -normal.x, y: -normal.y, z: 0 });
  const pins = !(firstPinned && secondPinned);
  const firstInverseMass = pins && firstPinned ? 0 : 1 / first.definition.massKg;
  const secondInverseMass = pins && secondPinned ? 0 : 1 / second.definition.massKg;
  const impulse = -dot(relativeVelocity, normal) / (firstInverseMass + secondInverseMass);
  first.velocity.x += normal.x * impulse * firstInverseMass;
  first.velocity.y += normal.y * impulse * firstInverseMass;
  second.velocity.x -= normal.x * impulse * secondInverseMass;
  second.velocity.y -= normal.y * impulse * secondInverseMass;
}

function limitBallSpeed(velocity: Vec3, maxSpeed: number): void {
  const speed = Math.hypot(velocity.x, velocity.y, velocity.z);
  if (speed <= maxSpeed) return;
  const scale = maxSpeed / speed;
  velocity.x *= scale;
  velocity.y *= scale;
  velocity.z *= scale;
}

function updatePlayerState(
  state: EngineState,
  actions: ReadonlyMap<PlayerId, PlayerAction>,
  elapsedSeconds: number,
): void {
  for (const player of state.players) {
    const action = actions.get(player.playerId);
    const { turnRateRadPerSec, staminaDrainPerSec, staminaRecoveryPerSec } = player.definition.abilities.physical;
    const moving = action !== undefined && Math.hypot(action.movement.x, action.movement.y) > 0;
    if (moving) {
      const target = Math.atan2(action.movement.y, action.movement.x);
      const difference = wrapAngle(target - player.facingRad);
      const turn = turnRateRadPerSec * elapsedSeconds;
      player.facingRad = wrapAngle(player.facingRad + Math.max(-turn, Math.min(turn, difference)));
    }
    const exertion = moving && action.sprint ? -staminaDrainPerSec : staminaRecoveryPerSec;
    player.stamina = Math.max(0, Math.min(1, player.stamina + exertion * elapsedSeconds));
  }

  if (state.ball.control.kind === "free") return;
  const holderId = state.ball.control.playerId;
  const holder = state.players.find((player) => player.playerId === holderId)!;
  const radius = state.config.ball.radiusM;
  if (state.ball.control.kind === "keeperHeld") {
    state.ball.position = keeperHoldPosition(holder, radius, state.config.pitch);
    state.ball.velocity = onGround(holder.velocity);
  }
}

function wrapAngle(angle: number): number {
  return Math.atan2(Math.sin(angle), Math.cos(angle));
}
