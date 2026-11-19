import type { TeamControlOptions } from "../session/contracts.js";

export type AIProfileId = "default" | "tuned" | "trained";

export interface AIProfile {
  id: AIProfileId;
  label: string;
  /** Match formats the profile was made for, as players per side. */
  tunedFor: readonly number[] | "any";
  /** Team AI plugin that runs this profile. Defaults to the baseline AI. */
  plugin?: string;
  /** Settings for the plugin. For the baseline, an empty object keeps its built-in behavior. */
  config: Readonly<Record<string, number>>;
}

/**
 * Named ways to run a computer team.
 *
 * "tuned" is the baseline AI with settings from `npm run tune -- search` on 5-a-side matches against the
 * default profile, retuned after the engine gained first-time kicks, pace-based stamina and collision balance
 * (runs/tune-v2.json). On 200 held-out seeds it beats the default profile by +4.3 goals per match (196 wins) and the
 * previous tuned settings by +2.1 (159 wins, 28 losses). It presses less than before, keeps the ball until an
 * opponent is close, plays short passes through clear lanes and shoots only from close in. It has not been tuned or
 * checked for 11-a-side.
 *
 * "trained" is a neural network trained by imitating the tuned profile (with team-role features), correcting with DAgger and
 * fine-tuning with PPO (see python/ and AI_TRAINING_PLAN.md). It runs as the "neural" plugin, which must be registered with the
 * weights in models/trained-5v5.json first. Over 128 held-out 5-a-side matches it scored about +4.3 goals per match
 * against the default profile and +3.5 against the tuned one (role features, DAgger and league PPO, see the plan).
 */
export const AI_PROFILES: Readonly<Record<AIProfileId, AIProfile>> = {
  default: { id: "default", label: "Default", tunedFor: "any", config: {} },
  tuned: {
    id: "tuned",
    label: "Hard",
    tunedFor: [5],
    config: {
      aggression: 0.637, riskTolerance: 0.732, awareness: 0.769,
      passLane: 2.706, passSpace: 0.490, passProgress: 1.360, passRange: 0.504,
      shotRange: 0.340, dribbleAhead: 0.482, pressureRadius: 0.297,
    },
  },
  trained: { id: "trained", label: "Trained", tunedFor: [5], plugin: "neural", config: {} },
};

export function isAIProfileId(value: string): value is AIProfileId {
  return Object.hasOwn(AI_PROFILES, value);
}

/** Team controller options that run the profile's plugin (the baseline AI unless the profile names another). */
export function profileTeamOptions(id: AIProfileId): TeamControlOptions {
  const { plugin, config } = AI_PROFILES[id];
  return { pluginId: plugin ?? "baseline", config: structuredClone(config) };
}

/** Whether the profile was made for matches with this many players per side. */
export function profileSupports(id: AIProfileId, playersPerSide: number): boolean {
  const { tunedFor } = AI_PROFILES[id];
  return tunedFor === "any" || tunedFor.includes(playersPerSide);
}
