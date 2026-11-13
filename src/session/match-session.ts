import type {
  ActionKey, MatchConfig, MatchEngine, MatchEvent, PlayerAction, PlayerId, ScenarioDefinition, TeamId, Vec2,
} from "../contracts/index.js";
import { BaselineTeamAI } from "../controllers/baseline.js";
import type { ControlAssignment, PlayerIntent } from "../controllers/contracts.js";
import { TeamAIRegistry } from "../controllers/registry.js";
import { validateIntents } from "../controllers/validation.js";
import { SoccerEngine } from "../engine/engine.js";
import type {
  ControllerFailure, HumanInput, SessionOptions, SessionSnapshot, SessionTickResult, TeamControlOptions,
} from "./contracts.js";
import { TeamControl } from "./team-control.js";

const neutral = { x: 0, y: 0 };
const defaultTeam: TeamControlOptions = { pluginId: "baseline" };

function sameAction(first: ActionKey, second: ActionKey): boolean {
  return first.source === second.source && first.playerId === second.playerId && first.requestId === second.requestId;
}

function positiveTicks(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

export function createDefaultAIRegistry(): TeamAIRegistry {
  return new TeamAIRegistry().register("baseline", () => new BaselineTeamAI());
}

export class MatchSession {
  private teams: TeamControl[] = [];
  private selected: PlayerId | null = null;
  private pendingSelection: PlayerId | null = null;
  private humanTeam: TeamControl | undefined;
  private movement: Vec2 = neutral;
  private sprint = false;
  private requestIds = new Map<PlayerId, number>();
  private intentionTicks = 360;
  private tick = 0;
  private finished = false;
  private faulted = false;

  constructor(
    private readonly registry = createDefaultAIRegistry(),
    private readonly engine: MatchEngine = new SoccerEngine(),
  ) {}

  reset(
    config: MatchConfig, seed: number, options: SessionOptions = {}, scenario?: ScenarioDefinition,
  ): readonly ControllerFailure[] {
    const { human } = options;
    const intentionTicks = positiveTicks(options.passIntentionTicks ?? 360, "Pass intention duration");
    const teamOptions = config.teams.map((team) => {
      const control = options.teams && Object.hasOwn(options.teams, team.id)
        ? options.teams[team.id] : defaultTeam;
      if (!control || !this.registry.has(control.pluginId)) throw new RangeError(`Unknown AI for team "${team.id}"`);
      positiveTicks(control.decisionIntervalTicks ?? 6, "AI decision interval");
      return control;
    });
    for (const teamId of Object.keys(options.teams ?? {})) {
      if (!config.teams.some((team) => team.id === teamId)) throw new RangeError(`Unknown team "${teamId}"`);
    }
    if (human && !config.teams.some(({ id, players }) =>
      id === human.teamId && players.some(({ id: playerId }) => playerId === human.playerId))) {
      throw new RangeError("Human selection must be an active player on the selected team");
    }

    // Engine reset is atomic. Keep the running controllers until it succeeds.
    this.engine.reset(config, seed, scenario);
    const failures = this.dispose();
    this.tick = 0;
    this.finished = false;
    this.faulted = false;
    this.intentionTicks = intentionTicks;
    this.selected = human?.playerId ?? null;
    this.requestIds = new Map();
    this.teams = config.teams.map((team, index) => {
      const control = new TeamControl(team, teamOptions[index], this.selected);
      // Separate deterministic seeds keep plugins independent of each other and the engine.
      const controllerSeed = (seed ^ (index === 0 ? 0x9e3779b9 : 0x85ebca6b)) >>> 0;
      control.reset(this.registry, team, config.teams[1 - index], index as 0 | 1, controllerSeed, teamOptions[index], failures);
      return control;
    });
    this.humanTeam = this.teams.find(({ teamId }) => teamId === human?.teamId);
    return failures;
  }

  dispose(): ControllerFailure[] {
    const failures: ControllerFailure[] = [];
    for (const team of this.teams) team.dispose(this.tick, failures);
    this.teams = [];
    this.selected = null;
    this.pendingSelection = null;
    this.humanTeam = undefined;
    this.clearHumanInput();
    return failures;
  }

  snapshot(): SessionSnapshot {
    return { ...this.engine.snapshot(), selectedPlayerId: this.selected };
  }

  assignment(teamId: TeamId): ControlAssignment {
    const team = this.teams.find((entry) => entry.teamId === teamId);
    if (!team) throw new RangeError(`Unknown team "${teamId}"`);
    return team.assignment();
  }

  step(input: HumanInput = {}): SessionTickResult {
    if (this.teams.length === 0) throw new Error("Reset the session before stepping");
    if (this.finished) throw new Error("The match has finished");
    if (this.faulted) throw new Error("Reset the session after a simulation failure");

    const observations = this.teams.map(({ teamId }) => this.engine.observe(teamId));
    const humanIntent = this.prepareHumanInput(input, observations[0].pitch);
    const failures: ControllerFailure[] = [];
    const actions = new Map<PlayerId, PlayerAction>();
    for (const { playerId } of observations[0].players) {
      actions.set(playerId, { playerId, movement: neutral, sprint: false });
    }

    for (const team of this.teams) {
      const intention = team.intention;
      if (intention && this.tick >= intention.expiresAtTick) team.clearIntention();
    }
    if (this.humanTeam && humanIntent) {
      const action = this.command(this.humanTeam, humanIntent, true);
      actions.set(action.playerId, action);
    }
    for (const [index, team] of this.teams.entries()) {
      const intents = team.decision(observations[index], failures);
      for (const intent of intents) actions.set(intent.playerId, this.command(team, intent, false));
    }

    const frame = { tick: this.tick, actions: [...actions.values()] };
    try {
      const result = this.engine.step(frame);
      this.tick = result.tick;
      this.finished = result.status === "finished";
      for (const team of this.teams) {
        team.recordEvents(result.events);
        for (const feedback of result.actionResults) {
          team.recordFeedback(feedback);
          if (team.intention && sameAction(team.intention.action, feedback.action)) {
            if (feedback.status === "accepted") team.intention.status = "accepted";
            if (feedback.status === "rejected") team.clearIntention();
          }
        }
        for (const event of result.events) this.handleEvent(team, event);
        const intention = team.intention;
        if (intention && intention.status !== "inFlight" && result.actionResults.some((feedback) =>
          feedback.status === "interrupted" && sameAction(intention.action, feedback.action))) {
          team.clearIntention();
        }
      }
      return structuredClone({ ...result, frame, controllerFailures: failures });
    } catch (error) {
      // Plugin memory cannot be rolled back after a failed tick.
      this.faulted = true;
      throw error;
    }
  }

  private clearHumanInput(): void {
    this.movement = neutral;
    this.sprint = false;
  }

  private prepareHumanInput(input: HumanInput, pitch: MatchConfig["pitch"]): PlayerIntent | undefined {
    const team = this.humanTeam;
    if (!team) {
      if (Object.keys(input).length > 0) throw new Error("This session has no human player");
      return undefined;
    }

    const manual = input.selectPlayerId;
    const next = manual ?? this.pendingSelection ?? this.selected!;
    if (!team.playerIds.has(next)) throw new RangeError("Select a player on the human team");
    const changed = next !== this.selected;
    const [validated] = validateIntents([{
      playerId: next,
      movement: input.movement === undefined ? (changed ? neutral : this.movement) : input.movement,
      sprint: input.sprint === undefined ? (changed ? false : this.sprint) : input.sprint,
      request: input.request,
      pass: input.pass,
    }], new Set([next]), team.playerIds, pitch);

    if (manual !== undefined && team.intention?.autoSwitchEligible) {
      team.intention.autoSwitchEligible = false;
      team.forceDecision();
    }
    this.pendingSelection = null;
    if (changed) {
      this.selected = next;
      team.select(next);
    }
    this.movement = validated.movement;
    this.sprint = validated.sprint;
    return validated;
  }

  private command(team: TeamControl, intent: PlayerIntent, human: boolean): PlayerAction {
    const { playerId, movement, sprint, request, pass } = intent;
    if (!request) return { playerId, movement, sprint };

    const requestId = (this.requestIds.get(playerId) ?? 0) + 1;
    this.requestIds.set(playerId, requestId);
    const key: ActionKey = { source: "controller", playerId, requestId };
    team.recordAction(key, this.tick);
    if (pass) {
      team.intention = {
        ...pass,
        action: key, createdTick: this.tick, expiresAtTick: this.tick + this.intentionTicks,
        status: "proposed", autoSwitchEligible: human,
      };
      if (human) team.forceDecision();
    }
    return { playerId, movement, sprint, request: { ...request, id: requestId } };
  }

  private handleEvent(team: TeamControl, event: MatchEvent): void {
    if (event.kind === "phaseChanged") {
      team.clearMovement();
      if (event.phase.kind !== "live") {
        team.clearIntention();
        this.pendingSelection = null;
        this.clearHumanInput();
      }
      return;
    }
    if (event.kind === "goal" || event.kind === "ballExit") {
      team.clearIntention();
      this.pendingSelection = null;
      return;
    }

    const intention = team.intention;
    if (!intention) return;
    if (event.kind === "actionContact" && event.outcome === "kicked" && sameAction(event.action, intention.action)) {
      intention.status = "inFlight";
      team.forceDecision();
      return;
    }
    if (event.kind === "actionContact" && intention.status === "inFlight" &&
        (event.outcome === "kicked" || event.outcome === "thrown" || event.outcome === "released")) {
      team.clearIntention();
      return;
    }
    if ((event.kind === "ballTouch" || event.kind === "actionContact") &&
        event.flightAction && sameAction(event.flightAction, intention.action)) {
      if (event.kind === "ballTouch" && event.outcome === "received" &&
          event.playerId === intention.receiverId && intention.autoSwitchEligible) {
        this.pendingSelection = intention.receiverId;
      }
      team.clearIntention();
    }
  }
}
