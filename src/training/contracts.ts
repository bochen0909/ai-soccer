import type { MatchConfig, ScenarioDefinition, TeamId } from "../contracts/index.js";
import type { TeamAIRegistry } from "../controllers/registry.js";
import type { TeamControlOptions } from "../session/contracts.js";

/** Movement choices: 0 stops, 1..8 walk along 45 degree steps from the attack direction, counterclockwise. */
export const MOVE_COUNT = 9;
/** Each agent submits [move, sprint, act]. */
export const ACTION_FIELDS = 3;

export const ACT_NONE = 0;
export const ACT_SHOOT_LEFT = 1;
export const ACT_SHOOT_CENTER = 2;
export const ACT_SHOOT_RIGHT = 3;
export const ACT_TACKLE = 4;
export const ACT_SAVE = 5;
export const ACT_PASS_BASE = 6;

export interface ActionSpace {
  maxPlayers: number;
  moveCount: number;
  /** Number of values in the `act` field: none, three shots, tackle, save, then pass and loft per slot. */
  actCount: number;
  passBase: number;
  loftBase: number;
}

export function createActionSpace(maxPlayers: number): ActionSpace {
  return {
    maxPlayers, moveCount: MOVE_COUNT, actCount: ACT_PASS_BASE + 2 * maxPlayers,
    passBase: ACT_PASS_BASE, loftBase: ACT_PASS_BASE + maxPlayers,
  };
}

export interface ObservationLayout {
  ego: number;
  ball: number;
  match: number;
  teammateSlot: number;
  opponentSlot: number;
  maxPlayers: number;
  /** Values in the trailing role block (0 for policies trained without it). */
  role: number;
  size: number;
}

export interface RewardWeights {
  goalScored: number;
  goalConceded: number;
  /** Ball advance toward the opposing goal, per pitch length, while play stays live. */
  ballProgress: number;
  /** Per policy step: positive while the team controls the ball, negative while the opponent does. */
  possession: number;
  /** A teammate receives a kick from another teammate. Can be farmed by passing back and forth, so keep it for drills. */
  passCompleted: number;
  /** Added to a goal the learner scores within six seconds of a teammate receiving a pass. Cannot be farmed. */
  assist: number;
  /** Added when the ball passes from the opponent's last touch to ours (tackle, interception, forced error), subtracted for the reverse. */
  turnover: number;
}

export interface TrainingEnvOptions {
  config?: MatchConfig;
  /** Team the trainer controls. Defaults to the first configured team. */
  learnerTeamId?: TeamId;
  /** Controller for the other team. Defaults to the baseline AI. */
  opponent?: TeamControlOptions;
  /** Extra plugins for the opponent. "external" is reserved. */
  registry?: TeamAIRegistry;
  /** Simulation ticks per policy transition. Defaults to six, which is 10 Hz. */
  ticksPerStep?: number;
  /** Pads observations and actions to this roster size so one policy fits 5v5 and 11v11. */
  maxPlayersPerTeam?: number;
  /** Policy steps before truncation. Defaults to no limit besides match completion. */
  maxSteps?: number;
  /** Append derived team-role features to every observation row. Defaults to true; false gives the original layout. */
  roleFeatures?: boolean;
  reward?: Partial<RewardWeights>;
  /** End the episode as soon as a goal is scored or the ball leaves play. Meant for short drills. */
  endOnDeadBall?: boolean;
}

export interface StepInfo {
  tick: number;
  step: number;
  /** [learner goals, opponent goals]. */
  score: readonly [number, number];
  /** Actions replaced by "none" because the mask forbade them. */
  illegalActions: number;
  rewardTerms: Readonly<Record<keyof RewardWeights, number>>;
}

export interface TrainingStepResult {
  /** Row-major [maxPlayers, observationSize]. Rows for absent agents are zero. */
  observation: Float32Array;
  /** Row-major [maxPlayers, actCount]. 1 means the act is legal for that agent now. */
  actionMask: Uint8Array;
  /** 1 for agents that exist in this roster. */
  agentMask: Uint8Array;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: StepInfo;
}

export type { ScenarioDefinition };
