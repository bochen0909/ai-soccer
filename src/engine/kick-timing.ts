import type { BallConfig, Vec2, Vec3 } from "../contracts/index.js";
import { ballPositionAfter } from "./ball-flight.js";
import { PHYSICS_TICK_SECONDS } from "./physics-geometry.js";

export type KickType = "pass" | "shot" | "loft";

export const KICK_TECHNIQUES = {
  pass: { speed: 0.55, minElevation: 0, maxElevation: 5, windup: 6, recovery: 8 },
  shot: { speed: 1, minElevation: 0, maxElevation: 25, windup: 8, recovery: 12 },
  loft: { speed: 0.75, minElevation: 15, maxElevation: 50, windup: 12, recovery: 12 },
} as const;

/** Ticks from a kick request to the foot meeting the ball; a harder shot takes a longer backswing. */
export function kickWindupTicks(kickType: KickType, power: number): number {
  return KICK_TECHNIQUES[kickType].windup + (kickType === "shot" ? Math.round(10 * power) : 0);
}

/**
 * Whether a free ball will be within the kicker's reach when a kick started now meets it, so a player can shape up for
 * a first-time shot or pass before the ball arrives. The kicker is taken to carry on at their current velocity.
 */
export function ballInReachAtContact(
  player: { position: Vec2; velocity: Vec2; reachM: number },
  ball: { position: Vec3; velocity: Vec3 },
  windupTicks: number,
  config: BallConfig,
): boolean {
  const seconds = windupTicks * PHYSICS_TICK_SECONDS;
  const at = ballPositionAfter(ball.position, ball.velocity, seconds, config);
  const feet = { x: player.position.x + player.velocity.x * seconds, y: player.position.y + player.velocity.y * seconds };
  return Math.hypot(at.x - feet.x, at.y - feet.y, at.z) <= player.reachM;
}
