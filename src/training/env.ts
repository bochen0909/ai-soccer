import type { MatchConfig, MatchEvent, PlayerId, ScenarioDefinition, TeamObservation } from "../contracts/index.js";
import { TeamAIRegistry } from "../controllers/registry.js";
import { SoccerEngine } from "../engine/engine.js";
import { createArcadeConfig } from "../presets/arcade.js";
import { createDefaultAIRegistry, MatchSession } from "../session/match-session.js";
import { decodeAction, writeActionMask } from "./actions.js";
import type { AgentContext } from "./actions.js";
import { ACTION_FIELDS, createActionSpace } from "./contracts.js";
import type {
  ActionSpace, ObservationLayout, RewardWeights, StepInfo, TrainingEnvOptions, TrainingStepResult,
} from "./contracts.js";
import { ExternalTeamAI } from "./external-team.js";
import { toPolicy } from "./geometry.js";
import { createObservationLayout, FEEDBACK_STATUSES, writeObservation } from "./observation.js";

export const DEFAULT_REWARD: RewardWeights = {
  goalScored: 1,
  goalConceded: -1,
  ballProgress: 0.1,
  possession: 0.002,
};

type RewardTerms = StepInfo["rewardTerms"];

/**
 * Single-match training environment. One policy transition advances `ticksPerStep` physics ticks.
 * The trainer controls every player on one team; the other team runs any registered controller.
 */
export class SoccerTrainingEnv {
  readonly layout: ObservationLayout;
  readonly actionSpace: ActionSpace;
  readonly learnerTeamId: string;
  /** Multiplies every non-goal reward term. Anneal toward zero as the policy improves. */
  shapingScale = 1;

  readonly config: MatchConfig;
  private readonly learnerIndex: 0 | 1;
  private readonly ticksPerStep: number;
  private readonly maxSteps: number;
  private readonly weights: RewardWeights;
  private readonly registry: TeamAIRegistry;
  private readonly opponent: NonNullable<TrainingEnvOptions["opponent"]>;
  private readonly engine = new SoccerEngine();
  private readonly session: MatchSession;
  private external?: ExternalTeamAI;
  private steps = 0;
  private done = true;
  private feedback = new Map<PlayerId, number>();

  constructor(options: TrainingEnvOptions = {}) {
    this.config = structuredClone(options.config ?? createArcadeConfig());
    const [first, second] = this.config.teams;
    if (options.learnerTeamId !== undefined && options.learnerTeamId !== first.id && options.learnerTeamId !== second.id) {
      throw new RangeError(`Unknown learner team "${options.learnerTeamId}"`);
    }
    this.learnerIndex = options.learnerTeamId === second.id ? 1 : 0;
    this.learnerTeamId = this.config.teams[this.learnerIndex].id;

    const roster = Math.max(first.players.length, second.players.length);
    const maxPlayers = options.maxPlayersPerTeam ?? roster;
    if (!Number.isSafeInteger(maxPlayers) || maxPlayers < roster) {
      throw new RangeError("maxPlayersPerTeam must be an integer of at least the roster size");
    }
    this.ticksPerStep = options.ticksPerStep ?? 6;
    this.maxSteps = options.maxSteps ?? Number.MAX_SAFE_INTEGER;
    if (!Number.isSafeInteger(this.ticksPerStep) || this.ticksPerStep < 1) {
      throw new RangeError("ticksPerStep must be a positive integer");
    }
    if (!Number.isSafeInteger(this.maxSteps) || this.maxSteps < 1) throw new RangeError("maxSteps must be a positive integer");
    this.weights = { ...DEFAULT_REWARD, ...options.reward };
    this.opponent = options.opponent ?? { pluginId: "baseline" };
    if (this.opponent.pluginId === "external") throw new RangeError('The opponent cannot use the reserved "external" plugin');

    this.registry = options.registry ?? createDefaultAIRegistry();
    this.registry.register("external", () => (this.external = new ExternalTeamAI()));
    this.session = new MatchSession(this.registry, this.engine);
    this.layout = createObservationLayout(maxPlayers);
    this.actionSpace = createActionSpace(maxPlayers);
  }

  get agentCount(): number {
    return this.layout.maxPlayers;
  }

  /** Starts a new match. Initial observations carry no reward. */
  reset(seed: number, scenario?: ScenarioDefinition): TrainingStepResult {
    const opponentId = this.config.teams[1 - this.learnerIndex].id;
    const failures = this.session.reset(this.config, seed, {
      teams: {
        [this.learnerTeamId]: { pluginId: "external", decisionIntervalTicks: 1 },
        [opponentId]: this.opponent,
      },
    }, scenario);
    if (failures.length > 0) throw new Error(`Controller failure: ${failures.map(({ message }) => message).join(" | ")}`);
    this.steps = 0;
    this.done = false;
    this.feedback.clear();
    const zero = { goalScored: 0, goalConceded: 0, ballProgress: 0, possession: 0 };
    return this.result(0, false, false, 0, zero);
  }

  /**
   * Advances one policy transition. `actions` holds [move, sprint, act] per agent in roster order.
   * Illegal acts are replaced by "none" and counted in `info.illegalActions`.
   */
  step(actions: ArrayLike<number>): TrainingStepResult {
    if (this.done) throw new Error("Reset the environment before stepping");
    const agents = this.layout.maxPlayers;
    if (actions.length !== agents * ACTION_FIELDS) {
      throw new RangeError(`Expected ${agents * ACTION_FIELDS} action values, received ${actions.length}`);
    }
    const external = this.external;
    if (!external) throw new Error("Reset the environment before stepping");

    const before = this.engine.observe(this.learnerTeamId);
    const contexts = this.contexts(before);
    const mask = new Uint8Array(this.actionSpace.actCount);
    let illegal = 0;
    for (const [slot, context] of contexts.entries()) {
      const base = slot * ACTION_FIELDS;
      const [move, sprint, act] = [actions[base], actions[base + 1], actions[base + 2]];
      this.validateChoice(move, this.actionSpace.moveCount, "move", slot);
      this.validateChoice(sprint, 2, "sprint", slot);
      this.validateChoice(act, this.actionSpace.actCount, "act", slot);
      writeActionMask(context, this.actionSpace, mask, 0);
      const legal = mask[act] === 1;
      if (!legal) illegal += 1;
      const decoded = decodeAction(context, this.actionSpace, move, sprint, legal ? act : 0);
      external.command(context.ego.playerId, decoded.movement, decoded.sprint, decoded.request);
    }

    const attack = before.attackDirections[this.learnerIndex];
    const startedLive = before.phase.kind === "live";
    let interrupted = false;
    const terms = { goalScored: 0, goalConceded: 0, ballProgress: 0, possession: 0 };
    this.feedback.clear();
    let elapsed = 0;
    let finished = false;
    for (; elapsed < this.ticksPerStep && !finished; elapsed += 1) {
      const result = this.session.step();
      if (result.controllerFailures.length > 0) {
        throw new Error(`Controller failure: ${result.controllerFailures.map(({ message }) => message).join(" | ")}`);
      }
      finished = result.status === "finished";
      for (const event of result.events) this.scoreEvent(event, terms) && (interrupted = true);
      for (const { action, status } of result.actionResults) {
        if (action.source === "controller") this.feedback.set(action.playerId, FEEDBACK_STATUSES.indexOf(status) + 1);
      }
    }

    const after = this.engine.observe(this.learnerTeamId);
    if (startedLive && after.phase.kind === "live" && !interrupted) {
      const gain = (toPolicy(after.ball.position, attack).x - toPolicy(before.ball.position, attack).x) / after.pitch.lengthM;
      terms.ballProgress = gain * this.weights.ballProgress * this.shapingScale;
    }
    const { control } = after.ball;
    if (control.kind !== "free") {
      const owner = after.players.find((player) => player.playerId === control.playerId);
      terms.possession = (owner?.teamId === this.learnerTeamId ? 1 : -1) * this.weights.possession * this.shapingScale;
    }

    this.steps += 1;
    const terminated = finished;
    const truncated = !terminated && this.steps >= this.maxSteps;
    this.done = terminated || truncated;
    return this.result(elapsed, terminated, truncated, illegal, terms, after);
  }

  /** The learner team's current observation in world coordinates. */
  observation(): TeamObservation {
    return this.engine.observe(this.learnerTeamId);
  }

  /** Hash of the engine state for reproducibility checks. */
  hashState(): string {
    return this.engine.hashState();
  }

  dispose(): void {
    this.session.dispose();
    this.done = true;
  }

  private validateChoice(value: number, count: number, label: string, slot: number): void {
    if (!Number.isInteger(value) || value < 0 || value >= count) {
      throw new RangeError(`Agent ${slot} ${label} must be an integer from 0 to ${count - 1}`);
    }
  }

  /** Returns true when the event ends continuous play. */
  private scoreEvent(event: MatchEvent, terms: { goalScored: number; goalConceded: number }): boolean {
    if (event.kind === "goal") {
      if (event.teamId === this.learnerTeamId) terms.goalScored += this.weights.goalScored;
      else terms.goalConceded += this.weights.goalConceded;
      return true;
    }
    return event.kind === "phaseChanged" || event.kind === "ballExit";
  }

  private contexts(observation: TeamObservation): AgentContext[] {
    const attack = observation.attackDirections[this.learnerIndex];
    const byId = new Map(observation.players.map((player) => [player.playerId, player]));
    const ordered = (index: 0 | 1) => this.config.teams[index].players.map(({ id }) => byId.get(id)!);
    const team = ordered(this.learnerIndex);
    const opponents = ordered((1 - this.learnerIndex) as 0 | 1);
    return team.map((ego) => ({ observation, attack, team, opponents, ego }));
  }

  private result(
    elapsed: number, terminated: boolean, truncated: boolean, illegal: number, terms: RewardTerms | Record<string, number>,
    observation = this.engine.observe(this.learnerTeamId),
  ): TrainingStepResult {
    const { maxPlayers } = this.layout;
    const { actCount } = this.actionSpace;
    const data = new Float32Array(maxPlayers * this.layout.size);
    const actionMask = new Uint8Array(maxPlayers * actCount);
    const agentMask = new Uint8Array(maxPlayers);
    for (const [slot, context] of this.contexts(observation).entries()) {
      agentMask[slot] = 1;
      writeObservation(context, this.layout, this.feedback.get(context.ego.playerId) ?? 0, this.learnerIndex,
        data, slot * this.layout.size);
      writeActionMask(context, this.actionSpace, actionMask, slot * actCount);
    }
    const rewardTerms = terms as RewardTerms;
    const reward = rewardTerms.goalScored + rewardTerms.goalConceded + rewardTerms.ballProgress + rewardTerms.possession;
    const score = [observation.score[this.learnerIndex], observation.score[1 - this.learnerIndex]] as const;
    return {
      observation: data, actionMask, agentMask, reward, terminated, truncated,
      info: { tick: observation.tick, step: this.steps, score, illegalActions: illegal, rewardTerms },
    };
  }
}
