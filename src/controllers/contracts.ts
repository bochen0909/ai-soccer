import type {
  ActionKey, ActionRequest, ActionResult, PlayerDefinition, PlayerId, TeamId, TeamObservation, Vec2,
} from "../contracts/index.js";

type WithoutId<T> = T extends ActionRequest ? Omit<T, "id"> : never;

export type ActionIntent = WithoutId<ActionRequest>;

export interface PassTarget {
  receiverId: PlayerId;
  target: Vec2;
}

export interface PlayerIntent {
  playerId: PlayerId;
  movement: Vec2;
  sprint: boolean;
  request?: ActionIntent;
  pass?: PassTarget;
}

export interface PassIntention extends PassTarget {
  action: ActionKey;
  createdTick: number;
  expiresAtTick: number;
  status: "proposed" | "accepted" | "inFlight";
  autoSwitchEligible: boolean;
}

export interface AIContext {
  teamId: TeamId;
  teamIndex: 0 | 1;
  players: readonly PlayerDefinition[];
  config: unknown;
  contractVersion: 1;
  seed: number;
}

export interface ControlAssignment {
  playerIds: readonly PlayerId[];
  revision: number;
}

export interface TeamDecisionContext {
  actionResults: readonly (ActionResult & { decisionTick: number })[];
  passIntention: PassIntention | null;
}

export interface TeamAI {
  readonly id: string;
  readonly version: string;
  reset(context: AIContext): void;
  decide(
    observation: TeamObservation,
    assignment: ControlAssignment,
    context: TeamDecisionContext,
  ): readonly PlayerIntent[];
  dispose(): void;
}
