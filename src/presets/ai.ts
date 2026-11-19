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
 * default profile, retuned after the baseline learned to defend a dribbler (goal-side chaser, cover, no tackles into
 * a shielded ball) and to keep the ball at kickoff (tuning/search-5v5-v3.json). Defending is now much stronger, so
 * games score less: on 200 held-out seeds it beats the default profile by +0.97 goals per match (129 wins, 7 losses)
 * and the previous tuned settings by +0.50. It dribbles further before passing and holds the ball until an opponent
 * is close. It has not been tuned or checked for 11-a-side.
 *
 * "trained" is a neural network trained by imitating the tuned profile (with team-role features), correcting with DAgger and
 * fine-tuning with PPO (see python/ and AI_TRAINING_PLAN.md). It runs as the "neural" plugin, which must be registered with the
 * weights in models/trained-5v5.json first. The shipped model is gen32 (gen30 trained on against the CPUs that defend
 * a dribbler): over 200 held-out 5-a-side matches it loses to the tuned profile by 0.45 goals per match (gen30 by
 * 1.32) and it is level with or slightly ahead of gen30 head to head (see the plan).
 */
export const AI_PROFILES: Readonly<Record<AIProfileId, AIProfile>> = {
  default: { id: "default", label: "Default", tunedFor: "any", config: {} },
  tuned: {
    id: "tuned",
    label: "Hard",
    tunedFor: [5],
    config: {
      aggression: 0.571, riskTolerance: 0.453, awareness: 0.431,
      passLane: 0.492, passSpace: 0.434, passProgress: 1.119, passRange: 1.280,
      shotRange: 0.986, dribbleAhead: 2.103, pressureRadius: 0.265,
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
