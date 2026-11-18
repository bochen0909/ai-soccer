export type PlayerId = string;
export type TeamId = string;
export type AttackDirection = -1 | 1;

export interface Vec2 {
  x: number;
  y: number;
}

export interface Vec3 extends Vec2 {
  z: number;
}

export interface PlayerAbilities {
  physical: {
    maxSpeedMps: number;
    accelerationMps2: number;
    brakingMps2: number;
    turnRateRadPerSec: number;
    maxKickSpeedMps: number;
    maxThrowInSpeedMps: number;
    maxKeeperRollSpeedMps: number;
    maxKeeperThrowSpeedMps: number;
    reachM: number;
    staminaDrainPerSec: number;
    staminaRecoveryPerSec: number;
  };
  technical: {
    passing: number;
    shooting: number;
    firstTouch: number;
    dribbling: number;
    tackling: number;
    goalkeeping: number;
    weakFoot: number;
  };
}

export interface PlayerDefinition {
  id: PlayerId;
  role: "outfield" | "goalkeeper";
  /** Formation position when the team attacks toward positive x. */
  formation: Vec2;
  bodyHeightM: number;
  radiusM: number;
  massKg: number;
  preferredFoot: "left" | "right";
  abilities: PlayerAbilities;
}

export interface TeamDefinition {
  id: TeamId;
  players: readonly PlayerDefinition[];
}

export interface PitchConfig {
  lengthM: number;
  widthM: number;
  goalWidthM: number;
  goalHeightM: number;
  postRadiusM: number;
  goalAreaDepthM: number;
  goalAreaWidthM: number;
  keeperAreaDepthM: number;
  keeperAreaWidthM: number;
  centerCircleRadiusM: number;
}

export interface BallConfig {
  radiusM: number;
  massKg: number;
  gravityMps2: number;
  groundDecelerationMps2: number;
  groundRestitution: number;
  contactRestitution: number;
  maxSpeedMps: number;
}

export interface MatchRules {
  version: 1;
  directionalKickoff?: boolean;
  /**
   * Every kickoff (the opening one, after each goal, and the second half) goes to either team with equal chance,
   * drawn from the match seed. Without it, the first kickoff goes to firstKickoffTeamId, the team that conceded
   * takes the one after a goal, and the other team starts the second half.
   */
  randomKickoff?: boolean;
  restartPlacementTicks: number;
  restartInputTicks: number;
  goalPauseTicks: number;
  halftimeTicks: number;
  keeperHoldTicks: number;
  restartExclusionM: number;
  /**
   * When set, the team awarded a throw-in or corner starts the restart with its outfield players pulled toward the
   * ball: the nearest at most this many meters away, the next at most twice as far, and so on. Off when unset.
   */
  restartSupportM?: number;
  /**
   * When true, a ball that reaches a player's head or upper body is headed on instead of bouncing off. The header goes
   * where the player is moving (or toward goal in the attacking third), with an error that shrinks with skill.
   */
  headerContact?: boolean;
}

export interface MatchConfig {
  pitch: PitchConfig;
  ball: BallConfig;
  teams: readonly [TeamDefinition, TeamDefinition];
  periodTicks: readonly [number, number];
  firstKickoffTeamId: TeamId;
  rules: MatchRules;
  tuning: {
    /** Multiplies how far a goalkeeper's save reaches. Unset means 1. */
    keeperReachScale?: number;
  };
}

export type RestartKind = "kickoff" | "throwIn" | "corner" | "goalKick";

interface RestartPlacement {
  teamId: TeamId;
  takerId: PlayerId;
  spot: Vec2;
  readyTick: number;
  deadlineTick: number;
}

export type MatchPhase =
  | ({ kind: "kickoffSetup" } & RestartPlacement)
  | ({ kind: "restartSetup"; restart: Exclude<RestartKind, "kickoff"> } & RestartPlacement)
  | { kind: "live" }
  | { kind: "goalPause"; teamId: TeamId; untilTick: number }
  | { kind: "halftime"; untilTick: number }
  | { kind: "fullTime" };

export type BallControl =
  | { kind: "free" }
  | { kind: "dribble"; playerId: PlayerId }
  | { kind: "keeperHeld"; playerId: PlayerId };

export interface ScenarioPlayerState {
  playerId: PlayerId;
  position: Vec2;
  facingRad: number;
  velocity?: Vec2;
  stamina?: number;
  balance?: number;
}

export interface ScenarioBallState {
  position: Vec3;
  velocity: Vec3;
  lastTouchPlayerId: PlayerId;
  control: BallControl;
}

export interface ScenarioDefinition {
  version: 1;
  id: string;
  period: 1 | 2;
  remainingPeriodTicks?: number;
  score?: readonly [number, number];
  players: readonly ScenarioPlayerState[];
  ball: ScenarioBallState;
  phase:
    | { kind: "live" }
    | { kind: "kickoffSetup"; teamId: TeamId }
    | {
        kind: "restartSetup";
        teamId: TeamId;
        restart: Exclude<RestartKind, "kickoff">;
        spot: Vec2;
      };
}

export interface PlayerState extends Required<ScenarioPlayerState> {
  teamId: TeamId;
  definition: PlayerDefinition;
}

export interface BallState {
  position: Vec3;
  velocity: Vec3;
  lastTouchPlayerId: PlayerId | null;
  control: BallControl;
  flightAction?: ActionKey;
}

export type ActionRequest =
  | { id: number; kind: "kick"; kickType: "pass" | "shot" | "loft"; direction: Vec2; power: number; loft: number }
  | { id: number; kind: "tackle"; direction: Vec2 }
  | { id: number; kind: "keeperSave"; direction: Vec2 }
  | { id: number; kind: "throwIn"; direction: Vec2; power: number }
  | { id: number; kind: "keeperRelease"; technique: "roll" | "throw"; direction: Vec2; power: number };

export interface PlayerAction {
  playerId: PlayerId;
  movement: Vec2;
  sprint: boolean;
  request?: ActionRequest;
}

export interface ActionFrame {
  tick: number;
  actions: readonly PlayerAction[];
}

export interface ActionKey {
  source: "controller" | "rule";
  playerId: PlayerId;
  requestId: number;
}

export interface ActionResult {
  action: ActionKey;
  tick: number;
  status: "accepted" | "completed" | "interrupted" | "rejected";
  reason?: string;
}

export interface EngineDiagnostic {
  kind: "duplicateRequest";
  action: ActionKey;
}

export interface PlayerActionState {
  key: ActionKey;
  kind: ActionRequest["kind"];
  phase: "windup" | "active" | "recovery";
  contactTick: number;
  activeUntilTick?: number;
  readyTick: number;
}

export type MatchEvent = { tick: number; ordinal: number } & (
  | { kind: "goal"; teamId: TeamId; position: Vec3; lastTouchPlayerId: PlayerId | null }
  | { kind: "ballExit"; restart: Exclude<RestartKind, "kickoff">; teamId: TeamId; position: Vec3 }
  | { kind: "phaseChanged"; phase: MatchPhase }
  | {
      kind: "actionContact";
      action: ActionKey;
      outcome: "kicked" | "thrown" | "released" | "wonBall" | "caught" | "parried" | "missed";
      position: Vec3;
      flightAction?: ActionKey;
    }
  | {
      kind: "ballTouch";
      playerId: PlayerId;
      outcome: "received" | "miscontrolled" | "dribbled" | "deflected" | "headed";
      position: Vec3;
      flightAction?: ActionKey;
    }
  | { kind: "possessionLost"; playerId: PlayerId; position: Vec3 }
);

export interface TickResult {
  tick: number;
  events: readonly MatchEvent[];
  actionResults: readonly ActionResult[];
  diagnostics: readonly EngineDiagnostic[];
  status: "running" | "finished";
}

export interface RenderSnapshot {
  tick: number;
  phase: MatchPhase;
  period: 1 | 2;
  remainingPeriodTicks: number;
  score: readonly [number, number];
  attackDirections: readonly [AttackDirection, AttackDirection];
  ball: BallState;
  players: readonly (PlayerState & { action?: PlayerActionState })[];
}

export interface TeamObservation extends RenderSnapshot {
  teamId: TeamId;
  pitch: PitchConfig;
  ballConfig?: BallConfig;
  tuning?: MatchConfig["tuning"];
  directionalKickoff?: boolean;
  events: readonly MatchEvent[];
}

export interface MatchEngine {
  reset(config: MatchConfig, seed: number, scenario?: ScenarioDefinition): void;
  observe(teamId: TeamId): TeamObservation;
  step(frame: ActionFrame): TickResult;
  snapshot(): RenderSnapshot;
  hashState(): string;
}
