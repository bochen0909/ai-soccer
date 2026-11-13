import type { AttackDirection, Vec2 } from "../contracts/index.js";

export const clamp = (value: number, min: number, max: number): number => Math.max(min, Math.min(max, value));

export function toPolicy(point: Vec2, attack: AttackDirection): Vec2 {
  // A half-turn keeps handedness, so preferred feet stay meaningful for both teams.
  return { x: point.x * attack, y: point.y * attack };
}

export const toWorld = toPolicy;

export function lead(position: Vec2, velocity: Vec2, seconds: number): Vec2 {
  return { x: position.x + velocity.x * seconds, y: position.y + velocity.y * seconds };
}

export function unitDirection(from: Vec2, to: Vec2, fallback: Vec2): Vec2 {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  return length > 1e-8 ? { x: dx / length, y: dy / length } : fallback;
}

/** Smallest clearance between a pass segment and the opponents, capped at four metres. */
export function laneClearance(
  from: Vec2, to: Vec2, opponents: readonly { position: Vec2; velocity: Vec2; definition: { radiusM: number } }[],
): number {
  const sx = to.x - from.x;
  const sy = to.y - from.y;
  const squared = sx * sx + sy * sy;
  if (squared < 1e-8) return 0;
  const range = Math.sqrt(squared);
  let clearance = 4;
  for (const opponent of opponents) {
    const fraction = ((opponent.position.x - from.x) * sx + (opponent.position.y - from.y) * sy) / squared;
    if (fraction < 0 || fraction > 1) continue;
    const seconds = Math.min(0.32, fraction * range / 12);
    const ox = opponent.position.x + opponent.velocity.x * seconds;
    const oy = opponent.position.y + opponent.velocity.y * seconds;
    const distance = Math.hypot(from.x + sx * fraction - ox, from.y + sy * fraction - oy);
    clearance = Math.min(clearance, distance - opponent.definition.radiusM);
  }
  return clearance;
}
