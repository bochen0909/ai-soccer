import type {
  ActionKey, ActionResult, MatchEvent, PlayerId, TeamDefinition, TeamObservation,
} from "../contracts/index.js";
import type {
  ControlAssignment, PassIntention, PlayerIntent, TeamAI, TeamDecisionContext,
} from "../controllers/contracts.js";
import type { TeamAIRegistry } from "../controllers/registry.js";
import { validateIntents } from "../controllers/validation.js";
import type { ControllerFailure, TeamControlOptions } from "./contracts.js";

type ActionFeedback = TeamDecisionContext["actionResults"][number];

export class TeamControl {
  readonly playerIds: ReadonlySet<PlayerId>;
  readonly interval: number;
  readonly pluginId: string;
  readonly teamId: string;
  private plugin?: TeamAI;
  private assigned: Set<PlayerId>;
  private revision = 0;
  private nextDecisionTick = 0;
  private events: MatchEvent[] = [];
  private feedback: ActionFeedback[] = [];
  private decisionTicks = new Map<PlayerId, Map<number, number>>();
  private movement: PlayerIntent[] = [];
  intention: PassIntention | null = null;

  constructor(team: TeamDefinition, options: TeamControlOptions, humanId: PlayerId | null) {
    this.teamId = team.id;
    this.playerIds = new Set(team.players.map(({ id }) => id));
    this.assigned = new Set([...this.playerIds].filter((id) => id !== humanId));
    this.interval = options.decisionIntervalTicks ?? 6;
    this.pluginId = options.pluginId;
  }

  private failure(stage: ControllerFailure["stage"], tick: number, error: unknown): ControllerFailure {
    return {
      teamId: this.teamId, pluginId: this.pluginId, stage, tick,
      message: error instanceof Error ? error.message : String(error),
    };
  }

  reset(
    registry: TeamAIRegistry, team: TeamDefinition, opponent: TeamDefinition, teamIndex: 0 | 1, seed: number,
    options: TeamControlOptions, failures: ControllerFailure[],
  ): void {
    try {
      this.plugin = registry.create(this.pluginId);
      this.plugin.reset(structuredClone({
        teamId: this.teamId, teamIndex, players: team.players, opponents: opponent.players, config: options.config,
        contractVersion: 1 as const, seed,
      }));
    } catch (error) {
      failures.push(this.failure("reset", 0, error));
      this.dispose(0, failures);
    }
  }

  dispose(tick: number, failures: ControllerFailure[]): void {
    const plugin = this.plugin;
    this.plugin = undefined;
    try {
      plugin?.dispose();
    } catch (error) {
      failures.push(this.failure("dispose", tick, error));
    }
  }

  assignment(): ControlAssignment {
    return { playerIds: [...this.assigned], revision: this.revision };
  }

  select(humanId: PlayerId | null): void {
    this.assigned = new Set([...this.playerIds].filter((id) => id !== humanId));
    this.revision += 1;
    this.clearMovement();
  }

  forceDecision(): void {
    this.nextDecisionTick = 0;
  }

  clearMovement(): void {
    this.movement = [];
    this.forceDecision();
  }

  decision(observation: TeamObservation, failures: ControllerFailure[]): readonly PlayerIntent[] {
    if (!this.plugin) return [];
    const { tick } = observation;
    const { phase } = observation;
    const restartReady = (phase.kind === "kickoffSetup" || phase.kind === "restartSetup") &&
      phase.teamId === this.teamId && phase.readyTick === tick;
    if (tick < this.nextDecisionTick && !restartReady) return this.movement;
    this.nextDecisionTick = tick + this.interval;
    const context: TeamDecisionContext = { actionResults: this.feedback, passIntention: this.intention };
    const observed = { ...observation, events: this.events };
    this.events = [];
    this.feedback = [];
    try {
      const submitted = this.plugin.decide(structuredClone(observed), this.assignment(), structuredClone(context));
      const valid = validateIntents(submitted, this.assigned, this.playerIds, observation.pitch);
      this.movement = valid.map(({ playerId, movement, sprint }) => ({ playerId, movement, sprint }));
      return valid;
    } catch (error) {
      this.movement = [];
      failures.push(this.failure("decide", tick, error));
      return [];
    }
  }

  recordAction(action: ActionKey, tick: number): void {
    if (!this.plugin) return;
    let requests = this.decisionTicks.get(action.playerId);
    if (!requests) {
      requests = new Map();
      this.decisionTicks.set(action.playerId, requests);
    }
    requests.set(action.requestId, tick);
  }

  recordFeedback(result: ActionResult): void {
    if (result.action.source !== "controller") return;
    const { playerId, requestId } = result.action;
    const requests = this.decisionTicks.get(playerId);
    const decisionTick = requests?.get(requestId);
    if (decisionTick === undefined) return;
    this.feedback.push({ ...result, decisionTick });
    if (result.status !== "accepted") requests?.delete(requestId);
  }

  recordEvents(events: readonly MatchEvent[]): void {
    if (this.plugin) this.events.push(...events);
  }

  clearIntention(): void {
    if (!this.intention) return;
    this.intention = null;
    this.forceDecision();
  }
}
