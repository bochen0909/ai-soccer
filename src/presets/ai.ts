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
 * default profile (about +6 goals per match on held-out seeds, 297 wins in 300). It presses hard and shoots from
 * close range, and has not been tuned or checked for 11-a-side.
 *
 * "trained" is a neural network trained by imitating the tuned profile, correcting with DAgger and fine-tuning with
 * PPO (see python/ and AI_TRAINING_PLAN.md). It runs as the "neural" plugin, which must be registered with the
 * weights in models/trained-5v5.json first. Over 64 held-out 5-a-side matches it scored about +1.4 to +1.5 goals per
 * match against the default profile and from even to half a goal behind the tuned one (league PPO, see the plan).
 */
export const AI_PROFILES: Readonly<Record<AIProfileId, AIProfile>> = {
  default: { id: "default", label: "Default", tunedFor: "any", config: {} },
  tuned: {
    id: "tuned",
    label: "Hard",
    tunedFor: [5],
    config: {
      aggression: 0.955, riskTolerance: 0.616, awareness: 0.764,
      passLane: 0.328, passSpace: 0.512, passProgress: 2.815, passRange: 1.099,
      shotRange: 0.571, dribbleAhead: 0.656, pressureRadius: 0.501,
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
