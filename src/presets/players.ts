import type { PlayerAbilities } from "../contracts/index.js";

export type PlayerAbilityPreset = "youth" | "teen" | "adult";

const agePresets = {
  youth: { speed: 0.76, acceleration: 0.84, power: 0.57, reach: 0.73, skill: 0.82 },
  teen: { speed: 0.91, acceleration: 0.94, power: 0.8, reach: 0.9, skill: 0.93 },
  adult: { speed: 1, acceleration: 1, power: 1, reach: 1, skill: 1 },
};

/** Return fresh, editable defaults for a player definition. */
export function createPlayerAbilities(preset: PlayerAbilityPreset = "adult"): PlayerAbilities {
  const scale = agePresets[preset];
  return {
    physical: {
      maxSpeedMps: 7.5 * scale.speed,
      accelerationMps2: 6 * scale.acceleration,
      brakingMps2: 9 * scale.acceleration,
      turnRateRadPerSec: 7,
      maxKickSpeedMps: 28 * scale.power,
      maxThrowInSpeedMps: 14 * scale.power,
      maxKeeperRollSpeedMps: 12 * scale.power,
      maxKeeperThrowSpeedMps: 20 * scale.power,
      reachM: 0.95 * scale.reach,
      staminaDrainPerSec: 0.065,
      staminaRecoveryPerSec: 0.04,
    },
    technical: {
      passing: 0.7 * scale.skill,
      shooting: 0.65 * scale.skill,
      firstTouch: 0.65 * scale.skill,
      dribbling: 0.65 * scale.skill,
      tackling: 0.6 * scale.skill,
      goalkeeping: 0.2 * scale.skill,
      weakFoot: 0.65 * scale.skill,
    },
  };
}
