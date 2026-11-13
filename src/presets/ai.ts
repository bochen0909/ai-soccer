import type { TeamControlOptions } from "../session/contracts.js";

export type AIProfileId = "default" | "tuned";

export interface AIProfile {
  id: AIProfileId;
  label: string;
  /** Match formats the parameters were tuned for, as players per side. */
  tunedFor: readonly number[] | "any";
  /** Settings for the baseline plugin. An empty object keeps its built-in behavior. */
  config: Readonly<Record<string, number>>;
}

/**
 * Named settings for the baseline team AI. "tuned" comes from `npm run tune -- search` on 5-a-side
 * matches against the default profile (about +6 goals per match on held-out seeds, 297 wins in 300).
 * It presses hard and shoots from close range, and has not been tuned or checked for 11-a-side.
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
};

export function isAIProfileId(value: string): value is AIProfileId {
  return Object.hasOwn(AI_PROFILES, value);
}

/** Team controller options that run the baseline AI with the profile's settings. */
export function profileTeamOptions(id: AIProfileId): TeamControlOptions {
  return { pluginId: "baseline", config: structuredClone(AI_PROFILES[id].config) };
}

/** Whether the profile was tuned for matches with this many players per side. */
export function profileSupports(id: AIProfileId, playersPerSide: number): boolean {
  const { tunedFor } = AI_PROFILES[id];
  return tunedFor === "any" || tunedFor.includes(playersPerSide);
}
