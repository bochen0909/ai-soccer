import type { PlayerState, Vec2 } from "../contracts/index.js";
import type { Capsule } from "./physics-geometry.js";

/** Ticks of the active window over which a save stretches from standing reach to a full dive. */
export const KEEPER_DIVE_TICKS = 15;
/** Extra reach of a full dive, at goalkeeping skill 1. */
export const DIVE_REACH_M = 0.8;

export function keeperSaveTiming(skill: number) {
  return { windup: 6 - Math.round(3 * skill), active: 18, recovery: 22 - Math.round(6 * skill) };
}

/** How far into the dive a save is, from 0 (standing reach) to 1 (full stretch), `ticks` after it became active. */
export function keeperDiveExtension(ticks: number): number {
  return Math.max(0, Math.min(1, ticks / KEEPER_DIVE_TICKS));
}

export function keeperSaveVolumes(player: PlayerState, direction: Vec2, reachScale = 1, extension = 0): Capsule[] {
  const { abilities, radiusM: bodyRadius, bodyHeightM } = player.definition;
  const skill = abilities.technical.goalkeeping;
  const exertion = (0.65 + 0.35 * player.stamina) * (0.7 + 0.3 * player.balance);
  const reach = reachScale * abilities.physical.reachM * (0.82 + 0.18 * skill) * (0.88 + 0.12 * exertion);
  const dive = Math.max(0, Math.min(1, extension)) * DIVE_REACH_M * (0.6 + 0.4 * skill) * exertion;
  const radiusM = Math.min(bodyRadius * 0.92, reach * 0.36);
  const point = (distance: number, z: number) => ({
    x: player.position.x + direction.x * distance,
    y: player.position.y + direction.y * distance,
    z,
  });
  // A short reach along the ground covers shots below the raised arm. A dive stretches both along the save direction.
  const lowRadius = radiusM * 0.68;
  const lowReach = reach * (0.49 + 0.18 * skill);
  const height = bodyHeightM * 0.1;
  return [
    { start: { ...player.position, z: bodyHeightM * 0.2 }, end: point(reach + dive - radiusM, bodyHeightM * 0.7), radiusM },
    { start: point(0, height), end: point(lowReach + dive - lowRadius, height), radiusM: lowRadius },
  ];
}
