import type {
  ActionFrame, PlayerId, RenderSnapshot, TeamId, TickResult, Vec2,
} from "../contracts/index.js";
import type { ActionIntent, PassTarget } from "../controllers/contracts.js";

export interface TeamControlOptions {
  pluginId: string;
  config?: unknown;
  decisionIntervalTicks?: number;
}

export interface SessionOptions {
  human?: { teamId: TeamId; playerId: PlayerId };
  teams?: Readonly<Record<TeamId, TeamControlOptions>>;
  passIntentionTicks?: number;
}

/** World-space input for one tick. Movement and sprint are held until updated. */
export interface HumanInput {
  selectPlayerId?: PlayerId;
  movement?: Vec2;
  sprint?: boolean;
  request?: ActionIntent;
  pass?: PassTarget;
}

export interface ControllerFailure {
  teamId: TeamId;
  pluginId: string;
  tick: number;
  stage: "reset" | "decide" | "dispose";
  message: string;
}

export interface SessionSnapshot extends RenderSnapshot {
  selectedPlayerId: PlayerId | null;
}

export interface SessionTickResult extends TickResult {
  frame: ActionFrame;
  controllerFailures: readonly ControllerFailure[];
}
