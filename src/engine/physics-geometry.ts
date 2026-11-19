import type { PitchConfig, PlayerState, Vec3 } from "../contracts/index.js";

export const PHYSICS_TICK_SECONDS = 1 / 60;

export interface Capsule {
  start: Vec3;
  end: Vec3;
  radiusM: number;
}

export function playerCapsule(player: PlayerState): Capsule {
  const { radiusM, bodyHeightM } = player.definition;
  const capHeight = Math.min(radiusM, bodyHeightM / 2);
  return {
    start: { ...player.position, z: capHeight },
    end: { ...player.position, z: bodyHeightM - capHeight },
    radiusM,
  };
}

export function goalFrameCapsules(pitch: PitchConfig): Capsule[] {
  const frames: Capsule[] = [];
  const y = pitch.goalWidthM / 2 + pitch.postRadiusM;
  const z = pitch.goalHeightM + pitch.postRadiusM;
  for (const x of [-pitch.lengthM / 2, pitch.lengthM / 2]) {
    const left = { x, y: -y, z };
    const right = { x, y, z };
    frames.push(
      { start: { ...left, z: 0 }, end: left, radiusM: pitch.postRadiusM },
      { start: { ...right, z: 0 }, end: right, radiusM: pitch.postRadiusM },
      { start: left, end: right, radiusM: pitch.postRadiusM },
    );
  }
  return frames;
}

export function subtract(a: Vec3, b: Vec3): Vec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function dot(a: Vec3, b: Vec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function along(origin: Vec3, direction: Vec3, distance: number): Vec3 {
  return {
    x: origin.x + direction.x * distance,
    y: origin.y + direction.y * distance,
    z: origin.z + direction.z * distance,
  };
}

function capsuleOffset(position: Vec3, capsule: Capsule): Vec3 {
  const axis = subtract(capsule.end, capsule.start);
  const offset = subtract(position, capsule.start);
  const axisLengthSquared = dot(axis, axis);
  if (axisLengthSquared === 0) return offset;
  const fraction = Math.max(0, Math.min(1, dot(offset, axis) / axisLengthSquared));
  return along(offset, axis, -fraction);
}

export function capsuleDistanceSquared(position: Vec3, capsule: Capsule): number {
  const offset = capsuleOffset(position, capsule);
  return dot(offset, offset);
}

export const CONTACT_TIME_TOLERANCE = 1e-9;
const DISTANCE_TOLERANCE = 1e-8;

/** First contact with a sphere along a linear path, including approaching overlaps. */
export function sphereContactTime(offset: Vec3, velocity: Vec3, radius: number): number | undefined {
  const halfLinear = dot(offset, velocity);
  const distance = dot(offset, offset) - radius * radius;
  if (halfLinear >= 0) return undefined;
  if (distance <= 0) return 0;
  const quadratic = dot(velocity, velocity);
  const discriminant = halfLinear * halfLinear - quadratic * distance;
  if (discriminant <= 0) return undefined;
  // The equivalent root using this denominator avoids subtracting nearly equal values.
  return distance / (-halfLinear + Math.sqrt(discriminant));
}

export interface SweepContact {
  time: number;
  normal: Vec3;
}

export function sweepCapsule(
  position: Vec3,
  velocity: Vec3,
  sphereRadius: number,
  capsule: Capsule,
  duration: number,
): SweepContact | undefined {
  const radius = capsule.radiusM + sphereRadius;
  const axis = subtract(capsule.end, capsule.start);
  const axisLengthSquared = dot(axis, axis);
  const initialOffset = capsuleOffset(position, capsule);
  if (dot(initialOffset, initialOffset) <= radius * radius) {
    return dot(initialOffset, velocity) < 0 ? makeContact(0, initialOffset) : undefined;
  }

  let first: SweepContact | undefined;
  const consider = (time: number | undefined): void => {
    if (time === undefined || time > duration + CONTACT_TIME_TOLERANCE ||
        (first !== undefined && time >= first.time)) return;
    const offset = capsuleOffset(along(position, velocity, time), capsule);
    const distance = Math.hypot(offset.x, offset.y, offset.z);
    if (Math.abs(distance - radius) <= DISTANCE_TOLERANCE && dot(offset, velocity) < 0) {
      first = makeContact(time, offset);
    }
  };

  // A capsule's surface is the cylindrical middle and two spherical caps.
  if (axisLengthSquared > 0) {
    const offset = subtract(position, capsule.start);
    const perpendicularOffset = along(offset, axis, -dot(offset, axis) / axisLengthSquared);
    const perpendicularVelocity = along(velocity, axis, -dot(velocity, axis) / axisLengthSquared);
    consider(sphereContactTime(perpendicularOffset, perpendicularVelocity, radius));
  }
  consider(sphereContactTime(subtract(position, capsule.start), velocity, radius));
  if (axisLengthSquared > 0) {
    consider(sphereContactTime(subtract(position, capsule.end), velocity, radius));
  }
  return first;
}

function makeContact(time: number, offset: Vec3): SweepContact {
  const distance = Math.hypot(offset.x, offset.y, offset.z);
  return { time, normal: { x: offset.x / distance, y: offset.y / distance, z: offset.z / distance } };
}
