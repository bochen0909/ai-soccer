import type { AttackDirection, PlayerId, TeamObservation, Vec2 } from "../contracts/index.js";
import { keeperSaveTiming, keeperSaveVolumes } from "../engine/keeper-save.js";
import { capsuleDistanceSquared } from "../engine/physics-geometry.js";
import type {
  AIContext, ControlAssignment, PassIntention, PlayerIntent, TeamAI, TeamDecisionContext,
} from "./contracts.js";

type PlayerState = TeamObservation["players"][number];
type Mode = "attack" | "defend" | "loose";

interface Traits {
  aggression: number;
  riskTolerance: number;
  awareness: number;
}

/** Multipliers on the baseline's hand-tuned constants. All default to 1, which keeps the original behavior. */
export interface Weights {
  /** Pass score: clear lane. */
  passLane: number;
  /** Pass score: space around the receiver. */
  passSpace: number;
  /** Pass score: forward progress. */
  passProgress: number;
  /** Pass score: penalty per metre of range. */
  passRange: number;
  /** Maximum shooting distance. */
  shotRange: number;
  /** Distance the carrier dribbles ahead. */
  dribbleAhead: number;
  /** Opponent distance that makes the carrier look for a pass. */
  pressureRadius: number;
}

interface PassOption {
  receiverId: PlayerId;
  target: Vec2;
  range: number;
  score: number;
}

interface Tactics {
  world: TeamObservation;
  attack: AttackDirection;
  teammates: readonly PlayerState[];
  opponents: readonly PlayerState[];
  traits: Traits;
  weights: Weights;
  fullSide: boolean;
}

const defaults: Traits = { aggression: 0.55, riskTolerance: 0.5, awareness: 0.65 };
export const DEFAULT_WEIGHTS: Weights = {
  passLane: 1, passSpace: 1, passProgress: 1, passRange: 1, shotRange: 1, dribbleAhead: 1, pressureRadius: 1,
};
const MAX_WEIGHT = 4;
const clamp = (number: number, min: number, max: number) => Math.max(min, Math.min(max, number));
const difference = (from: Vec2, to: Vec2): Vec2 => ({ x: to.x - from.x, y: to.y - from.y });
const length = (vector: Vec2): number => Math.hypot(vector.x, vector.y);
const distance = (a: Vec2, b: Vec2): number => length(difference(a, b));

function direction(from: Vec2, to: Vec2, attack: AttackDirection): Vec2 {
  const offset = difference(from, to);
  const magnitude = length(offset);
  return magnitude > 1e-8 ? { x: offset.x / magnitude, y: offset.y / magnitude } : { x: attack, y: 0 };
}

function lead(position: Vec2, velocity: Vec2, seconds: number): Vec2 {
  return { x: position.x + velocity.x * seconds, y: position.y + velocity.y * seconds };
}

function onPitch(tactics: Tactics, target: Vec2, margin = 0.7): Vec2 {
  const { lengthM, widthM } = tactics.world.pitch;
  const x = Math.max(0, lengthM / 2 - margin);
  const y = Math.max(0, widthM / 2 - margin);
  return { x: clamp(target.x, -x, x), y: clamp(target.y, -y, y) };
}

function idle(player: PlayerState): PlayerIntent {
  return { playerId: player.playerId, movement: { x: 0, y: 0 }, sprint: false };
}

function move(player: PlayerState, target: Vec2, sprint = false): PlayerIntent {
  const offset = difference(lead(player.position, player.velocity, 0.12), target);
  const scale = Math.min(1 / 1.8, 1 / Math.max(1e-8, length(offset)));
  return { playerId: player.playerId, movement: { x: offset.x * scale, y: offset.y * scale }, sprint };
}

function trait(config: unknown, name: keyof Traits): number {
  const value = config && typeof config === "object" ? (config as Partial<Traits>)[name] : undefined;
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, 0, 1) : defaults[name];
}

function weight(config: unknown, name: keyof Weights): number {
  const value = config && typeof config === "object" ? (config as Partial<Weights>)[name] : undefined;
  return typeof value === "number" && Number.isFinite(value) ? clamp(value, 0, MAX_WEIGHT) : DEFAULT_WEIGHTS[name];
}

function laneClearance(from: Vec2, to: Vec2, opponents: readonly PlayerState[]): number {
  const segment = difference(from, to);
  const squared = segment.x ** 2 + segment.y ** 2;
  if (squared < 1e-8) return 0;
  let clearance = 4;
  for (const opponent of opponents) {
    const offset = difference(from, opponent.position);
    const fraction = (offset.x * segment.x + offset.y * segment.y) / squared;
    if (fraction < 0 || fraction > 1) continue;
    const future = lead(opponent.position, opponent.velocity, Math.min(0.32, fraction * Math.sqrt(squared) / 12));
    const point = lead(from, segment, fraction);
    clearance = Math.min(clearance, distance(point, future) - opponent.definition.radiusM);
  }
  return clearance;
}

function nearestDistance(position: Vec2, players: readonly PlayerState[], maximum: number): number {
  return players.reduce((closest, player) => Math.min(closest, distance(position, player.position)), maximum);
}

function passingOption(tactics: Tactics, passer: PlayerState, origin: Vec2, maxRange: number): PassOption | undefined {
  let best: PassOption | undefined;
  for (const receiver of tactics.teammates) {
    if (receiver.playerId === passer.playerId) continue;
    const target = onPitch(tactics, lead(receiver.position, receiver.velocity, 0.32));
    const range = distance(origin, target);
    if (range < 2 || range > maxRange) continue;
    // During setup, avoid kicking back through the stationary restart taker.
    const ballFromTaker = difference(passer.position, origin);
    const aim = direction(origin, target, tactics.attack);
    const phase = tactics.world.phase.kind;
    const canPassBack = tactics.fullSide && phase === "kickoffSetup" && tactics.world.directionalKickoff;
    if (phase !== "live" && !canPassBack &&
        ballFromTaker.x * aim.x + ballFromTaker.y * aim.y < -0.05) continue;
    const space = nearestDistance(target, tactics.opponents, 5);
    const lane = laneClearance(origin, target, tactics.opponents);
    const progress = tactics.attack * (target.x - origin.x);
    const keeperPenalty = receiver.definition.role === "goalkeeper" ? 2 : 0;
    const w = tactics.weights;
    const score = 1.15 * w.passLane * lane + 0.38 * w.passSpace * space +
      (0.08 + 0.1 * tactics.traits.riskTolerance) * w.passProgress * progress -
      0.06 * w.passRange * range - keeperPenalty;
    if (!best || score > best.score) best = { receiverId: receiver.playerId, target, range, score };
  }
  return best;
}

function passingPower(range: number, speedLimit: number): number {
  // Leave enough pace to reach the receiving player's foot-control radius.
  return clamp(Math.sqrt(3.2 * range + 4) / speedLimit, 0.18, 0.88);
}

function restart(tactics: Tactics, player: PlayerState): PlayerIntent {
  const { phase, tick, ball } = tactics.world;
  const intent = idle(player);
  if ((phase.kind !== "kickoffSetup" && phase.kind !== "restartSetup") || player.playerId !== phase.takerId ||
      player.action || tick < phase.readyTick || tick >= phase.deadlineTick) return intent;
  const kind = phase.kind === "restartSetup" ? phase.restart : "kickoff";
  const physical = player.definition.abilities.physical;
  const option = passingOption(tactics, player, ball.position, kind === "throwIn" ? 13 : 17);
  const target = option?.target ?? (kind === "kickoff" ? { x: 6 * tactics.attack, y: 2 } : { x: 0, y: 0 });
  const aim = direction(ball.position, target, tactics.attack);
  if (kind === "throwIn") {
    return { ...intent, request: {
      kind: "throwIn", direction: aim, power: option ? clamp(option.range / 13, 0.28, 0.7) : 0.45,
    } };
  }
  return {
    ...intent,
    request: { kind: "kick", kickType: "pass", direction: aim, loft: 0,
      power: passingPower(distance(ball.position, target), physical.maxKickSpeedMps * 0.55) },
    pass: option ? { receiverId: option.receiverId, target } : undefined,
  };
}

function carrying(tactics: Tactics, player: PlayerState): PlayerIntent {
  const { ball, pitch } = tactics.world;
  const origin = ball.position;
  const goalX = tactics.attack * pitch.lengthM / 2;
  const pressure = nearestDistance(origin, tactics.opponents, Infinity);
  const side = origin.y >= 0 ? 1 : -1;
  const target = onPitch(tactics, { x: origin.x + tactics.attack * 6 * tactics.weights.dribbleAhead, y: origin.y * 0.63 });
  const dribble = move(player, target);
  if (player.action?.phase === "windup") return idle(player);
  if (player.action || distance(player.position, origin) > player.definition.abilities.physical.reachM * 0.95) return dribble;

  let shotTarget: Vec2 = { x: goalX, y: -side * pitch.goalWidthM * 0.29 };
  let clearance = laneClearance(origin, shotTarget, tactics.opponents);
  for (const y of [side * pitch.goalWidthM * 0.29, 0]) {
    const candidate = { x: goalX, y };
    const open = laneClearance(origin, candidate, tactics.opponents);
    if (open > clearance) {
      shotTarget = candidate;
      clearance = open;
    }
  }
  const shotRange = (6 + player.definition.abilities.technical.shooting * 11) * (0.8 + tactics.traits.riskTolerance * 0.4) *
    tactics.weights.shotRange;
  const shotDistance = distance(origin, shotTarget);
  if (shotDistance <= shotRange && (clearance > 0.35 || shotDistance < 3.5)) {
    return { ...idle(player), request: {
      kind: "kick", kickType: "shot", direction: direction(origin, shotTarget, tactics.attack),
      power: clamp(0.53 + shotDistance / 38, 0.53, 0.9), loft: 0.12,
    } };
  }

  const passSpeed = player.definition.abilities.physical.maxKickSpeedMps * 0.55;
  const pass = passingOption(tactics, player, origin, Math.min(tactics.fullSide ? 26 : 19, passSpeed ** 2 * 0.23));
  const forwardLane = laneClearance(origin, target, tactics.opponents);
  if (pass && (pressure < 2.4 * tactics.weights.pressureRadius || forwardLane < 0.9) && pass.score > 0.6 + tactics.traits.riskTolerance) {
    return {
      ...idle(player),
      request: { kind: "kick", kickType: "pass", direction: direction(origin, pass.target, tactics.attack),
        power: passingPower(pass.range, passSpeed), loft: 0 },
      pass: { receiverId: pass.receiverId, target: pass.target },
    };
  }
  if (pressure < 3 && forwardLane < 1.3) {
    const up = onPitch(tactics, { x: target.x, y: origin.y + 4 });
    const down = onPitch(tactics, { x: target.x, y: origin.y - 4 });
    return move(player, laneClearance(origin, up, tactics.opponents) >= laneClearance(origin, down, tactics.opponents)
      ? up : down);
  }
  return dribble;
}

function estimateArrival(player: PlayerState, target: Vec2): number {
  const offset = difference(player.position, target);
  const range = length(offset);
  const physical = player.definition.abilities.physical;
  const cruise = physical.maxSpeedMps * (0.75 + player.stamina * 0.25);
  const speedToward = (player.velocity.x * offset.x + player.velocity.y * offset.y) / Math.max(range, 1e-8);
  const acceleration = physical.accelerationMps2;
  const timeToCruise = Math.max(0, (cruise - speedToward) / acceleration);
  const accelerationDistance = speedToward * timeToCruise + 0.5 * acceleration * timeToCruise ** 2;
  return range < accelerationDistance
    ? (-speedToward + Math.sqrt(speedToward ** 2 + 2 * acceleration * range)) / acceleration
    : timeToCruise + Math.max(0, range - accelerationDistance) / cruise;
}

function interception(tactics: Tactics, player: PlayerState): { target: Vec2; eta: number } {
  const { position, velocity } = tactics.world.ball;
  const maxLookahead = 0.5 + tactics.traits.awareness * 0.6;
  let target: Vec2 = onPitch(tactics, position);
  let eta = estimateArrival(player, target);
  for (let time = 0.15; time <= maxLookahead; time += 0.15) {
    target = onPitch(tactics, lead(position, velocity, time * (position.z < 0.25 ? 1 - 0.16 * time : 1)));
    eta = estimateArrival(player, target);
    if (eta <= time) break;
  }
  return { target, eta };
}

function formation(tactics: Tactics, player: PlayerState, mode: Mode): Vec2 {
  const { pitch, ball } = tactics.world;
  const { x, y } = player.definition.formation;
  if (tactics.fullSide) {
    const ballX = ball.position.x * tactics.attack;
    const defensiveLine = x <= -pitch.lengthM * 0.28;
    // Keep the back four between the ball and goal while advancing the midfield in possession.
    const base = defensiveLine && mode !== "attack" ? Math.min(x, ballX - 4) : x;
    const advance = mode === "attack" ? (defensiveLine ? 7 : 18) : 0;
    const shift = mode === "attack" ? 0.42 : defensiveLine ? 0.17 : 0.36;
    const lineShift = defensiveLine && mode !== "attack" ? Math.max(0, ballX * shift) : ballX * shift;
    return onPitch(tactics, {
      x: (base + lineShift + advance) * tactics.attack,
      y: y * tactics.attack + ball.position.y * (defensiveLine ? 0.17 : 0.24),
    });
  }
  const shift = mode === "attack" ? 0.49 : mode === "defend" ? 0.33 : 0.36;
  const run = mode === "attack" ? Math.max(0, 1 + 2 * x / pitch.lengthM) * pitch.lengthM * 0.21 : 0;
  return onPitch(tactics, {
    x: x * tactics.attack + ball.position.x * shift + tactics.attack * run,
    y: y * tactics.attack + ball.position.y * 0.29,
  });
}

function saveAim(tactics: Tactics, player: PlayerState, ownGoalX: number): Vec2 | undefined {
  const { ball, ballConfig, pitch, tuning } = tactics.world;
  const { windup, active } = keeperSaveTiming(player.definition.abilities.technical.goalkeeping);
  const ballRadius = ballConfig?.radiusM ?? 0.11;
  const gravity = ballConfig?.gravityMps2 ?? 9.81;
  let best: { aim: Vec2; miss: number } | undefined;

  for (let tick = windup; tick < windup + active; tick++) {
    const time = tick / 60;
    const predicted = lead(ball.position, ball.velocity, time);
    const height = Math.max(ballRadius, ball.position.z + ball.velocity.z * time - 0.5 * gravity * time ** 2);
    const areaDepth = (predicted.x - ownGoalX) * tactics.attack;
    if (areaDepth < 0 || areaDepth > pitch.keeperAreaDepthM ||
        Math.abs(predicted.y) > pitch.keeperAreaWidthM / 2) continue;
    const position = lead(player.position, player.velocity, time);
    const aim = direction(position, predicted, tactics.attack);
    for (const volume of keeperSaveVolumes({ ...player, position }, aim, tuning?.version === 2)) {
      const miss = Math.sqrt(capsuleDistanceSquared({ ...predicted, z: height }, volume)) - volume.radiusM;
      if (miss < ballRadius && (!best || miss < best.miss)) best = { aim, miss };
    }
  }
  return best?.aim;
}

function goalkeeper(tactics: Tactics, player: PlayerState): PlayerIntent {
  const { pitch, ball } = tactics.world;
  const { attack } = tactics;
  const ownGoalX = -attack * pitch.lengthM / 2;
  const physical = player.definition.abilities.physical;
  const depth = (ball.position.x - ownGoalX) * attack;
  const keeperDepth = (player.position.x - ownGoalX) * attack;
  const halfArea = pitch.keeperAreaWidthM / 2;
  const inArea = keeperDepth >= 0 && keeperDepth <= pitch.keeperAreaDepthM && Math.abs(player.position.y) <= halfArea;
  const lineDepth = Math.min(pitch.keeperAreaDepthM * 0.72, 0.75 + clamp(depth / pitch.lengthM, 0, 1) * 2.3);
  const line = ownGoalX + attack * lineDepth;
  const towardGoal = ball.velocity.x * attack < -0.5;
  const threatTime = towardGoal ? (line - ball.position.x) / ball.velocity.x : -1;
  const aimY = threatTime > 0 && threatTime < 1
    ? ball.position.y + ball.velocity.y * threatTime : ball.position.y * 0.28;
  const coverage = Math.min(pitch.goalWidthM * 0.48, halfArea * 0.85);
  const intent = move(player, onPitch(tactics, { x: line, y: clamp(aimY, -coverage, coverage) }));
  if (player.action) return intent;

  if (ball.control.kind === "keeperHeld" && ball.control.playerId === player.playerId) {
    const pass = passingOption(tactics, player, player.position, Math.min(18, physical.maxKeeperThrowSpeedMps));
    const target = pass?.target ?? { x: ownGoalX + attack * (pitch.keeperAreaDepthM + 3), y: 0 };
    const range = distance(player.position, target);
    const roll = range < Math.min(11, physical.maxKeeperRollSpeedMps ** 2 * 0.22);
    return { ...idle(player), request: {
      kind: "keeperRelease", technique: roll ? "roll" : "throw", direction: direction(player.position, target, attack),
      power: roll ? passingPower(range, physical.maxKeeperRollSpeedMps)
        : clamp(range / physical.maxKeeperThrowSpeedMps, 0.32, 0.82),
    } };
  }
  if (ball.control.kind === "dribble" && ball.control.playerId === player.playerId) return carrying(tactics, player);
  if (!inArea || ball.control.kind === "keeperHeld") return intent;

  if (towardGoal) {
    const aim = saveAim(tactics, player, ownGoalX);
    if (aim) return { ...intent, request: { kind: "keeperSave", direction: aim } };
  }
  if (ball.control.kind === "free" && depth > 0.5 && depth < pitch.keeperAreaDepthM * 0.65 &&
      Math.abs(ball.position.y) < halfArea * 0.7 && distance(player.position, ball.position) < 3 && length(ball.velocity) < 2) {
    return move(player, ball.position);
  }
  return intent;
}

function passingRun(tactics: Tactics, player: PlayerState, intention: PassIntention): PlayerIntent {
  const target = intention.status === "inFlight" ? interception(tactics, player).target : intention.target;
  if (tactics.fullSide && intention.status === "inFlight") {
    // Hold the receiving lane when the pass is coming to us. Closing it at a sprint
    // adds our running speed to the impact and makes a routine first touch harder.
    const { position, velocity } = tactics.world.ball;
    const speed = length(velocity);
    const ballRadius = tactics.world.ballConfig?.radiusM ?? 0.11;
    if (speed > 4.5 && position.z <= ballRadius + 1e-6 && Math.abs(velocity.z) < 1e-6) {
      const offset = difference(position, player.position);
      const along = (offset.x * velocity.x + offset.y * velocity.y) / speed;
      const across = (offset.x * velocity.y - offset.y * velocity.x) / speed;
      const friction = tactics.world.ballConfig?.groundDecelerationMps2 ?? 1.6;
      if (along > 0 && speed ** 2 - along * 2 * friction > 4.5 ** 2) {
        const lane = {
          x: player.position.x - (across * velocity.y + 0.3 * velocity.x) / speed,
          y: player.position.y + (across * velocity.x - 0.3 * velocity.y) / speed,
        };
        return move(player, onPitch(tactics, lane));
      }
    }
  }
  return move(player, target, distance(player.position, target) > 4);
}

/** Deterministic formation tactics with one pressing or chasing outfield player. */
export class BaselineTeamAI implements TeamAI {
  readonly id = "baseline";
  readonly version = "4";
  private teamIndex: 0 | 1 = 0;
  private traits: Traits = defaults;
  private weights: Weights = DEFAULT_WEIGHTS;
  private chaser: PlayerId | undefined;
  private mode: Mode | undefined;

  reset(context: AIContext): void {
    this.teamIndex = context.teamIndex;
    this.traits = {
      aggression: trait(context.config, "aggression"),
      riskTolerance: trait(context.config, "riskTolerance"),
      awareness: trait(context.config, "awareness"),
    };
    this.weights = Object.fromEntries(
      (Object.keys(DEFAULT_WEIGHTS) as (keyof Weights)[]).map((name) => [name, weight(context.config, name)]),
    ) as unknown as Weights;
    this.dispose();
  }

  dispose(): void {
    this.chaser = undefined;
    this.mode = undefined;
  }

  decide(observation: TeamObservation, assignment: ControlAssignment, context: TeamDecisionContext): readonly PlayerIntent[] {
    const { ball, players, teamId } = observation;
    const teammates = players.filter((player) => player.teamId === teamId);
    const assigned = new Set(assignment.playerIds);
    const controlled = teammates.filter((player) => assigned.has(player.playerId));
    const tactics: Tactics = {
      world: observation, attack: observation.attackDirections[this.teamIndex], teammates,
      opponents: players.filter((player) => player.teamId !== teamId), traits: this.traits, weights: this.weights,
      fullSide: teammates.length === 11,
    };
    if (observation.phase.kind !== "live") {
      this.dispose();
      return controlled.map((player) => restart(tactics, player));
    }

    const ownerId = ball.control.kind === "free" ? undefined : ball.control.playerId;
    const owner = players.find((player) => player.playerId === ownerId);
    const intention = context.passIntention;
    const mode: Mode = intention ? "attack" : !owner ? "loose" : owner.teamId === teamId ? "attack" : "defend";
    const outfield = controlled.filter((player) => player.definition.role === "outfield");
    const candidates = mode !== "attack" && ball.control.kind !== "keeperHeld"
      ? outfield.map((player) => {
        const intercept = interception(tactics, player);
        const centerBack = tactics.fullSide && player.definition.formation.x <= -observation.pitch.lengthM * 0.28 &&
          Math.abs(player.definition.formation.y) < observation.pitch.widthM * 0.18;
        const leavingLine = centerBack && intercept.target.x * tactics.attack > -observation.pitch.lengthM * 0.24;
        return { player, ...intercept, eta: intercept.eta + (leavingLine ? 0.85 : 0) };
      }) : [];
    const previous = mode === this.mode ? candidates.find(({ player }) => player.playerId === this.chaser) : undefined;
    const fastest = candidates.reduce<typeof candidates[number] | undefined>((best, candidate) =>
      !best || candidate.eta < best.eta ? candidate : best, undefined);
    const primary = previous && fastest && previous.eta <= fastest.eta + 0.2 + this.traits.aggression * 0.2
      ? previous : fastest;
    this.chaser = primary?.player.playerId;
    this.mode = mode;

    const marks = tactics.opponents.filter((player) => player.definition.role === "outfield" && player.playerId !== ownerId);
    const markerById = new Map<PlayerId, PlayerState>();
    if (mode === "defend") {
      for (const defender of outfield) {
        if (defender.playerId === this.chaser || marks.length === 0) continue;
        const anchor = formation(tactics, defender, mode);
        let closest = 0;
        for (let i = 1; i < marks.length; i += 1) {
          if (distance(anchor, marks[i].position) < distance(anchor, marks[closest].position)) closest = i;
        }
        markerById.set(defender.playerId, marks.splice(closest, 1)[0]);
      }
    }
    return controlled.map((player): PlayerIntent => {
      if (ownerId === player.playerId && ball.control.kind === "dribble") return carrying(tactics, player);
      if (intention?.receiverId === player.playerId) return passingRun(tactics, player, intention);
      if (player.definition.role === "goalkeeper") return goalkeeper(tactics, player);
      if (primary?.player.playerId === player.playerId) {
        const range = distance(player.position, ball.position);
        const chase = move(player, primary.target, range > 3.5 - this.traits.aggression);
        const reach = player.definition.abilities.physical.reachM;
        if (mode === "defend" && !player.action && range < reach * 1.06 && ball.position.z < player.definition.radiusM + 0.1) {
          return { ...chase, request: { kind: "tackle", direction: direction(player.position, primary.target, tactics.attack) } };
        }
        return chase;
      }
      const anchor = formation(tactics, player, mode);
      const mark = markerById.get(player.playerId);
      if (!mark) return move(player, anchor);
      const markWeight = tactics.fullSide ? 0.28 : 0.52;
      return move(player, onPitch(tactics, {
        x: (1 - markWeight) * anchor.x + markWeight * (mark.position.x - 1.5 * tactics.attack),
        y: (1 - markWeight) * anchor.y + markWeight * mark.position.y,
      }));
    });
  }
}
