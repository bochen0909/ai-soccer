import type { MatchConfig, MatchEvent, PlayerId, ScenarioDefinition, TeamObservation } from "../contracts/index.js";
import { TeamAIRegistry } from "../controllers/registry.js";
import { SoccerEngine } from "../engine/engine.js";
import { createArcadeConfig } from "../presets/arcade.js";
import { createDefaultAIRegistry, MatchSession } from "../session/match-session.js";
import { decodeAction, writeActionMask } from "./actions.js";
import { buildAgentContexts } from "./agents.js";
import type { AgentContext } from "./actions.js";
import { ACTION_FIELDS, createActionSpace } from "./contracts.js";
import type {
  ActionSpace, ObservationLayout, RewardWeights, StepInfo, TrainingEnvOptions, TrainingStepResult,
} from "./contracts.js";
import { ExternalTeamAI } from "./external-team.js";
import { toPolicy } from "./geometry.js";
import { createObservationLayout, FEEDBACK_STATUSES, writeObservation } from "./observation.js";

const ASSIST_WINDOW_TICKS = 360;
const PROGRESS_END_MARGIN_M = 6;
const SHOT_SPEED_MPS = 21.5;
const SHOT_RANGE_M = 20;
const OWN_THIRD_M = 20;

export const DEFAULT_REWARD: RewardWeights = {
  goalScored: 1,
  goalConceded: -1,
  ballProgress: 0.1,
  possession: 0.002,
  passCompleted: 0,
  assist: 0,
  turnover: 0,
  passBreak: 0,
  save: 0,
  passFailed: 0,
  outOfBounds: 0,
  shotOnTarget: 0,
  ownThirdGiveaway: 0,
  regain: 0,
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
  private lastPassTick = -Infinity;
  /** Team of the last player to touch or kick the ball, or null at the start and after a goal. */
  private lastTouchLearner: boolean | null = null;
  /** Opponents ahead of the ball when a learner last kicked it, by id, with the ball's attack-frame x then. */
  private kickedPast = new Set<PlayerId>();
  /** A learner pass or loft is in the air and no teammate has controlled it yet. */
  private passInFlight = false;

  readonly config: MatchConfig;
  private readonly learnerIndex: 0 | 1;
  private readonly ticksPerStep: number;
  private readonly maxSteps: number;
  private readonly endOnDeadBall: boolean;
  private readonly endOnBallLoss: boolean;
  private readonly passBonusOnce: boolean;
  private readonly goalNeedsPass: boolean;
  private readonly endOnSave: boolean;
  private readonly endOnRegain: boolean;
  private regained = false;
  private readonly possessionByLastTouch: boolean;
  /** Team of the last player to touch the ball (true: ours), from the scenario's carrier at the start, or null after a goal. */
  private holder: boolean | null = null;
  /** A learner pass has been completed in this episode. */
  private passRewarded = false;
  private readonly learnerPlayers: ReadonlySet<PlayerId>;
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
    this.learnerPlayers = new Set(this.config.teams[this.learnerIndex].players.map(({ id }) => id));

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
    this.endOnDeadBall = options.endOnDeadBall ?? false;
    this.endOnBallLoss = options.endOnBallLoss ?? false;
    this.passBonusOnce = options.passBonusOnce ?? false;
    this.goalNeedsPass = options.goalNeedsPass ?? false;
    this.endOnSave = options.endOnSave ?? false;
    this.endOnRegain = options.endOnRegain ?? false;
    this.possessionByLastTouch = options.possessionByLastTouch ?? false;
    this.weights = { ...DEFAULT_REWARD, ...options.reward };
    this.opponent = options.opponent ?? { pluginId: "baseline" };
    if (this.opponent.pluginId === "external") throw new RangeError('The opponent cannot use the reserved "external" plugin');

    this.registry = options.registry ?? createDefaultAIRegistry();
    this.registry.register("external", () => (this.external = new ExternalTeamAI()));
    this.session = new MatchSession(this.registry, this.engine);
    this.layout = createObservationLayout(maxPlayers, options.roleFeatures !== false);
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
    this.passRewarded = false;
    this.regained = false;
    this.holder = scenario ? this.learnerPlayers.has(scenario.ball.lastTouchPlayerId) : null;
    this.steps = 0;
    this.done = false;
    this.lastPassTick = -Infinity;
    this.lastTouchLearner = null;
    this.kickedPast = new Set();
    this.passInFlight = false;
    this.feedback.clear();
    const zero = { goalScored: 0, goalConceded: 0, ballProgress: 0, possession: 0, passCompleted: 0, assist: 0, turnover: 0, passBreak: 0, save: 0, passFailed: 0, outOfBounds: 0, shotOnTarget: 0, ownThirdGiveaway: 0, regain: 0 };
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
    const terms = { goalScored: 0, goalConceded: 0, ballProgress: 0, possession: 0, passCompleted: 0, assist: 0, turnover: 0, passBreak: 0, save: 0, passFailed: 0, outOfBounds: 0, shotOnTarget: 0, ownThirdGiveaway: 0, regain: 0 };
    this.feedback.clear();
    let elapsed = 0;
    let finished = false;
    for (; elapsed < this.ticksPerStep && !finished; elapsed += 1) {
      const result = this.session.step();
      if (result.controllerFailures.length > 0) {
        throw new Error(`Controller failure: ${result.controllerFailures.map(({ message }) => message).join(" | ")}`);
      }
      finished = result.status === "finished";
      for (const event of result.events) {
        if (this.scoreEvent(event, terms)) interrupted = true;
        if (this.endOnDeadBall && (event.kind === "goal" || event.kind === "ballExit")) finished = true;
        if (this.endOnBallLoss && this.rivalTouched(event)) finished = true;
        if (this.endOnRegain && this.regained) finished = true;
        if (this.endOnSave && event.kind === "actionContact" && this.learnerPlayers.has(event.action.playerId) &&
            (event.outcome === "caught" || event.outcome === "parried")) finished = true;
      }
      for (const { action, status } of result.actionResults) {
        if (action.source === "controller") this.feedback.set(action.playerId, FEEDBACK_STATUSES.indexOf(status) + 1);
      }
    }

    const after = this.engine.observe(this.learnerTeamId);
    if (startedLive && after.phase.kind === "live" && !interrupted) {
      // No credit for pushing the ball into the last few meters: past there, the shot is what pays.
      const cap = after.pitch.lengthM / 2 - PROGRESS_END_MARGIN_M;
      const gain = (Math.min(cap, toPolicy(after.ball.position, attack).x) -
        Math.min(cap, toPolicy(before.ball.position, attack).x)) / after.pitch.lengthM;
      terms.ballProgress = gain * this.weights.ballProgress * this.shapingScale;
    }
    const { control } = after.ball;
    if (control.kind !== "free") {
      const owner = after.players.find((player) => player.playerId === control.playerId);
      terms.possession = (owner?.teamId === this.learnerTeamId ? 1 : -1) * this.weights.possession * this.shapingScale;
    } else if (this.possessionByLastTouch && this.holder !== null) {
      // A ball in flight still belongs to the team that last touched it, so a pass costs no possession.
      terms.possession = (this.holder ? 1 : -1) * this.weights.possession * this.shapingScale;
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

  /** A rival touched or won the ball: a tackle, an interception, a keeper catch or parry. A missed tackle does not count. */
  private rivalTouched(event: MatchEvent): boolean {
    if (event.kind === "ballTouch") return !this.learnerPlayers.has(event.playerId);
    if (event.kind === "actionContact") return !this.learnerPlayers.has(event.action.playerId) && event.outcome !== "missed";
    return false;
  }

  private validateChoice(value: number, count: number, label: string, slot: number): void {
    if (!Number.isInteger(value) || value < 0 || value >= count) {
      throw new RangeError(`Agent ${slot} ${label} must be an integer from 0 to ${count - 1}`);
    }
  }

  /** Returns true when the event ends continuous play. */
  private scoreEvent(event: MatchEvent, terms: { goalScored: number; goalConceded: number; passCompleted: number; assist: number; turnover: number; passBreak: number; save: number; passFailed: number; outOfBounds: number; shotOnTarget: number; ownThirdGiveaway: number; regain: number }): boolean {
    if (event.kind === "goal") {
      this.lastTouchLearner = null;
      this.holder = null;
      this.passInFlight = false;
      if (event.teamId === this.learnerTeamId) {
        if (!this.goalNeedsPass || this.passRewarded) terms.goalScored += this.weights.goalScored;
        if (event.tick - this.lastPassTick <= ASSIST_WINDOW_TICKS) terms.assist += this.weights.assist * this.shapingScale;
      }
      else terms.goalConceded += this.weights.goalConceded;
      return true;
    }
    if (event.kind === "ballExit") {
      if (event.teamId !== this.learnerTeamId) terms.outOfBounds += this.weights.outOfBounds * this.shapingScale;
      this.passInFlight = false;
    }
    if (event.kind === "actionContact" && this.learnerPlayers.has(event.action.playerId) &&
        (event.outcome === "caught" || event.outcome === "parried")) {
      terms.save += this.weights.save * this.shapingScale;
    }
    if (this.passInFlight && ((event.kind === "ballTouch" && !this.learnerPlayers.has(event.playerId)) ||
        (event.kind === "actionContact" && !this.learnerPlayers.has(event.action.playerId) &&
          (event.outcome === "wonBall" || event.outcome === "caught" || event.outcome === "parried")))) {
      terms.passFailed += this.weights.passFailed * this.shapingScale;
      this.passInFlight = false;
    }
    const toucher = event.kind === "ballTouch" ? event.playerId
      : event.kind === "actionContact" && event.outcome !== "missed" ? event.action.playerId : undefined;
    if (toucher !== undefined && this.learnerPlayers.has(toucher) && this.holder === false && !this.regained) {
      // A learner touches the ball after the opponent had it.
      this.regained = true;
      terms.regain += this.weights.regain * this.shapingScale;
    }
    if (toucher !== undefined) {
      const learner = this.learnerPlayers.has(toucher);
      if (this.lastTouchLearner !== null && this.lastTouchLearner !== learner) {
        terms.turnover += (learner ? 1 : -1) * this.weights.turnover * this.shapingScale;
        if (!learner && this.weights.ownThirdGiveaway !== 0 && "position" in event) {
          const view = this.engine.observe(this.learnerTeamId);
          const depth = event.position.x * view.attackDirections[this.learnerIndex] + view.pitch.lengthM / 2;
          if (depth < OWN_THIRD_M) terms.ownThirdGiveaway += this.weights.ownThirdGiveaway * this.shapingScale;
        }
      }
      this.lastTouchLearner = learner;
      this.holder = learner;
    }
    if (event.kind === "actionContact" && event.outcome === "kicked") {
      this.kickedPast = new Set();
      this.passInFlight = false;
      if (this.learnerPlayers.has(event.action.playerId)) this.scoreKick(event.position, terms);
      if (this.learnerPlayers.has(event.action.playerId) && this.weights.passBreak !== 0) {
        const view = this.engine.observe(this.learnerTeamId);
        const attack = view.attackDirections[this.learnerIndex];
        for (const player of view.players) {
          if (!this.learnerPlayers.has(player.playerId) && player.position.x * attack > event.position.x * attack + 0.5) {
            this.kickedPast.add(player.playerId);
          }
        }
      }
    }
    if (event.kind === "ballTouch" && event.outcome === "received" && event.flightAction !== undefined &&
        event.flightAction.playerId !== event.playerId && this.learnerPlayers.has(event.flightAction.playerId) &&
        this.learnerPlayers.has(event.playerId)) {
      if (!this.passBonusOnce || !this.passRewarded) terms.passCompleted += this.weights.passCompleted * this.shapingScale;
      this.passRewarded = true;
      this.passInFlight = false;
      this.lastPassTick = event.tick;
      if (this.kickedPast.size > 0) {
        const view = this.engine.observe(this.learnerTeamId);
        const attack = view.attackDirections[this.learnerIndex];
        const receiver = view.players.find((player) => player.playerId === event.playerId)!;
        const bypassed = view.players.filter((player) => this.kickedPast.has(player.playerId) &&
          player.position.x * attack < receiver.position.x * attack - 0.5).length;
        terms.passBreak += Math.min(2, bypassed) * this.weights.passBreak * this.shapingScale;
        this.kickedPast = new Set();
      }
    }
    return event.kind === "phaseChanged" || event.kind === "ballExit";
  }

  /** Called right after a learner kick: a shot on target is paid, anything slower is a pass in flight. */
  private scoreKick(origin: { x: number; y: number }, terms: { shotOnTarget: number }): void {
    const view = this.engine.observe(this.learnerTeamId);
    const attack = view.attackDirections[this.learnerIndex];
    const { position, velocity } = view.ball;
    const speed = Math.hypot(velocity.x, velocity.y, velocity.z);
    const goalX = attack * view.pitch.lengthM / 2;
    const distance = Math.abs(goalX - origin.x);
    if (speed >= SHOT_SPEED_MPS && velocity.x * attack > 0.3 * speed) {
      const time = (goalX - position.x) / velocity.x;
      const y = position.y + velocity.y * time;
      // A ball that would land first just bounces on toward goal, so only the upper limit matters.
      const z = position.z + velocity.z * time - 0.5 * (view.ballConfig?.gravityMps2 ?? 9.81) * time * time;
      if (distance <= SHOT_RANGE_M && Math.abs(y) <= view.pitch.goalWidthM / 2 && z <= view.pitch.goalHeightM) {
        const close = Math.max(0.25, Math.min(1.5, 1.5 - (1.25 * distance) / SHOT_RANGE_M));
        terms.shotOnTarget += close * this.weights.shotOnTarget * this.shapingScale;
      }
      return;
    }
    this.passInFlight = true;
  }

  private contexts(observation: TeamObservation): AgentContext[] {
    return buildAgentContexts(observation, this.config, this.learnerIndex);
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
    const reward = rewardTerms.goalScored + rewardTerms.goalConceded + rewardTerms.ballProgress +
      rewardTerms.possession + rewardTerms.passCompleted + rewardTerms.assist + rewardTerms.turnover +
      rewardTerms.passBreak + rewardTerms.save + rewardTerms.passFailed + rewardTerms.outOfBounds +
      rewardTerms.shotOnTarget + rewardTerms.ownThirdGiveaway + rewardTerms.regain;
    const score = [observation.score[this.learnerIndex], observation.score[1 - this.learnerIndex]] as const;
    return {
      observation: data, actionMask, agentMask, reward, terminated, truncated,
      info: { tick: observation.tick, step: this.steps, score, illegalActions: illegal, rewardTerms },
    };
  }
}
