import type { PlayerState, Vec2 } from "../contracts/index.js";
import type { Capsule } from "./physics-geometry.js";

export function keeperSaveTiming(skill: number) {
  return { windup: 6 - Math.round(3 * skill), active: 6, recovery: 14 - Math.round(5 * skill) };
}

export function keeperSaveVolumes(player: PlayerState, direction: Vec2, reachScale = 1): Capsule[] {
  const { abilities, radiusM: bodyRadius, bodyHeightM } = player.definition;
  const skill = abilities.technical.goalkeeping;
  const exertion = (0.65 + 0.35 * player.stamina) * (0.7 + 0.3 * player.balance);
  const reach = reachScale * abilities.physical.reachM * (0.82 + 0.18 * skill) * (0.88 + 0.12 * exertion);
  const radiusM = Math.min(bodyRadius * 0.92, reach * 0.36);
  const point = (distance: number, z: number) => ({
    x: player.position.x + direction.x * distance,
    y: player.position.y + direction.y * distance,
    z,
  });
  // A short reach along the ground covers shots below the raised arm.
  const lowRadius = radiusM * 0.68;
  const lowReach = reach * (0.49 + 0.18 * skill);
  const height = bodyHeightM * 0.1;
  return [
    { start: { ...player.position, z: bodyHeightM * 0.2 }, end: point(reach - radiusM, bodyHeightM * 0.7), radiusM },
    { start: point(0, height), end: point(lowReach - lowRadius, height), radiusM: lowRadius },
  ];
}
