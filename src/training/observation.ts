import type { ActionResult, AttackDirection, TeamObservation } from "../contracts/index.js";
import type { ObservationLayout } from "./contracts.js";
import { clamp, laneClearance, toPolicy } from "./geometry.js";
import type { AgentContext } from "./actions.js";

type PlayerState = TeamObservation["players"][number];

const EGO_SIZE = 39;
const BALL_SIZE = 12;
const MATCH_SIZE = 12;
const TEAMMATE_SLOT = 11;
const OPPONENT_SLOT = 10;

/** Offsets inside the ego block of an observation row. Divide offsets by pitch length to get metres. */
export const EGO_INDEX = {
  positionX: 0, positionY: 1, velocityX: 2, velocityY: 3, keeper: 8,
  hasBall: 29, ballReachable: 30, ballDistance: 33, ballOffsetX: 34, ballOffsetY: 35,
  goalOffsetX: 36, goalOffsetY: 37, goalDistance: 38,
} as const;

export function createObservationLayout(maxPlayers: number): ObservationLayout {
  return {
    ego: EGO_SIZE, ball: BALL_SIZE, match: MATCH_SIZE,
    teammateSlot: TEAMMATE_SLOT, opponentSlot: OPPONENT_SLOT, maxPlayers,
    size: EGO_SIZE + BALL_SIZE + MATCH_SIZE + (TEAMMATE_SLOT + OPPONENT_SLOT) * maxPlayers,
  };
}

/** Status codes for the previous policy step's action feedback. */
export const FEEDBACK_NONE = 0;
export const FEEDBACK_STATUSES: readonly ActionResult["status"][] = ["accepted", "completed", "interrupted", "rejected"];

const PHASE_ONE_HOT = ["none", "windup", "active", "recovery"] as const;
const SPEED_SCALE = 10;
const BALL_SPEED_SCALE = 30;

class Row {
  private cursor: number;
  constructor(private readonly data: Float32Array, private readonly start: number, private readonly end: number) {
    this.cursor = start;
  }
  put(...values: number[]): void {
    for (const value of values) this.data[this.cursor++] = value;
  }
  oneHot(index: number, count: number): void {
    for (let i = 0; i < count; i += 1) this.data[this.cursor++] = i === index ? 1 : 0;
  }
  skip(count: number): void {
    this.cursor += count;
  }
  close(): void {
    if (this.cursor !== this.end) throw new Error(`Observation row wrote ${this.cursor - this.start} of ${this.end - this.start} values`);
  }
}

/**
 * Writes one agent's observation in the team's attack frame: positions are rotated so the team always
 * attacks toward positive x, then divided by half the pitch size. Slots keep roster order so pass
 * choice k always refers to teammate block k.
 */
export function writeObservation(
  context: AgentContext, layout: ObservationLayout, feedback: number, learnerIndex: 0 | 1,
  data: Float32Array, offset: number,
): void {
  const { observation, attack, ego, team, opponents } = context;
  const { pitch, ball, phase } = observation;
  const halfX = pitch.lengthM / 2;
  const halfY = pitch.widthM / 2;
  const row = new Row(data, offset, offset + layout.size);

  const position = (point: { x: number; y: number }) => {
    const local = toPolicy(point, attack);
    return { x: local.x / halfX, y: local.y / halfY };
  };
  const velocity = (point: { x: number; y: number }, scale = SPEED_SCALE) => {
    const local = toPolicy(point, attack);
    return { x: local.x / scale, y: local.y / scale };
  };

  // Ego block.
  const physical = ego.definition.abilities.physical;
  const technical = ego.definition.abilities.technical;
  const egoPosition = position(ego.position);
  const egoVelocity = velocity(ego.velocity);
  const facing = attack === 1 ? ego.facingRad : ego.facingRad + Math.PI;
  const toBall = Math.hypot(ball.position.x - ego.position.x, ball.position.y - ego.position.y);
  const ballOffset = toPolicy({ x: ball.position.x - ego.position.x, y: ball.position.y - ego.position.y }, attack);
  const formation = position({ x: ego.definition.formation.x * attack, y: ego.definition.formation.y * attack });
  const goal = toPolicy({ x: attack * halfX - ego.position.x, y: -ego.position.y }, attack);
  row.put(egoPosition.x, egoPosition.y, egoVelocity.x, egoVelocity.y, Math.cos(facing), Math.sin(facing));
  row.put(ego.stamina, ego.balance, ego.definition.role === "goalkeeper" ? 1 : 0);
  row.put(
    physical.maxSpeedMps / SPEED_SCALE, physical.accelerationMps2 / SPEED_SCALE, physical.maxKickSpeedMps / 40,
    physical.reachM / 2, technical.passing, technical.shooting, technical.firstTouch, technical.dribbling,
    technical.tackling, technical.goalkeeping, technical.weakFoot,
  );
  row.oneHot(ego.action ? PHASE_ONE_HOT.indexOf(ego.action.phase) : 0, PHASE_ONE_HOT.length);
  row.oneHot(feedback, FEEDBACK_STATUSES.length + 1);
  row.put(
    ball.control.kind === "dribble" && ball.control.playerId === ego.playerId ? 1 : 0,
    toBall <= physical.reachM ? 1 : 0,
    formation.x, formation.y, clamp(toBall / pitch.lengthM, 0, 1),
    ballOffset.x / pitch.lengthM, ballOffset.y / pitch.lengthM,
    goal.x / pitch.lengthM, goal.y / pitch.lengthM, Math.hypot(goal.x, goal.y) / pitch.lengthM,
  );

  // Ball block.
  const ballPosition = position(ball.position);
  const ballVelocity = velocity(ball.velocity, BALL_SPEED_SCALE);
  const ownerId = ball.control.kind === "free" ? undefined : ball.control.playerId;
  const owner = ownerId === undefined ? undefined : observation.players.find((player) => player.playerId === ownerId);
  const ownerIsOurs = owner?.teamId === ego.teamId;
  const lastTouch = ball.lastTouchPlayerId === null ? undefined
    : observation.players.find((player) => player.playerId === ball.lastTouchPlayerId);
  row.put(ballPosition.x, ballPosition.y, ball.position.z / 3, ballVelocity.x, ballVelocity.y, ball.velocity.z / BALL_SPEED_SCALE);
  row.put(
    ball.control.kind === "free" ? 1 : 0,
    ball.control.kind === "dribble" && ownerIsOurs ? 1 : 0,
    ball.control.kind === "dribble" && !ownerIsOurs ? 1 : 0,
    ball.control.kind === "keeperHeld" && ownerIsOurs ? 1 : 0,
    ball.control.kind === "keeperHeld" && !ownerIsOurs ? 1 : 0,
    lastTouch ? (lastTouch.teamId === ego.teamId ? 1 : -1) : 0,
  );

  // Match block.
  const own = observation.score[learnerIndex];
  const other = observation.score[1 - learnerIndex];
  const periodTicks = Math.max(1, observation.remainingPeriodTicks);
  const setup = phase.kind === "kickoffSetup" || phase.kind === "restartSetup" ? phase : undefined;
  const restart = setup === undefined ? -1 : setup.kind === "kickoffSetup" ? 0 : ["throwIn", "corner", "goalKick"].indexOf(setup.restart) + 1;
  row.put(
    clamp(periodTicks / 10_800, 0, 1), observation.period - 1, clamp((own - other) / 5, -1, 1),
    phase.kind === "live" ? 1 : 0,
    setup && setup.teamId === ego.teamId ? 1 : 0,
    setup && setup.teamId !== ego.teamId ? 1 : 0,
    phase.kind === "goalPause" || phase.kind === "halftime" || phase.kind === "fullTime" ? 1 : 0,
    setup && setup.takerId === ego.playerId ? 1 : 0,
  );
  row.oneHot(restart, 4);

  // Teammate slots.
  for (let slot = 0; slot < layout.maxPlayers; slot += 1) {
    const mate = team[slot] as PlayerState | undefined;
    if (!mate) { row.skip(layout.teammateSlot); continue; }
    const self = mate.playerId === ego.playerId;
    const absolute = position(mate.position);
    const relative = toPolicy({ x: mate.position.x - ego.position.x, y: mate.position.y - ego.position.y }, attack);
    const range = Math.hypot(relative.x, relative.y);
    const mateVelocity = velocity(mate.velocity);
    row.put(
      self ? 0 : 1, relative.x / pitch.lengthM, relative.y / pitch.lengthM, absolute.x, absolute.y,
      mateVelocity.x, mateVelocity.y, mate.definition.role === "goalkeeper" ? 1 : 0, mate.stamina,
      clamp(range / pitch.lengthM, 0, 1),
      self ? 0 : laneClearance(ball.position, mate.position, opponents) / 4,
    );
  }
  // Opponent slots, ordered as the opponent roster.
  for (let slot = 0; slot < layout.maxPlayers; slot += 1) {
    const foe = opponents[slot];
    if (!foe) { row.skip(layout.opponentSlot); continue; }
    const absolute = position(foe.position);
    const relative = toPolicy({ x: foe.position.x - ego.position.x, y: foe.position.y - ego.position.y }, attack);
    const foeVelocity = velocity(foe.velocity);
    row.put(
      1, relative.x / pitch.lengthM, relative.y / pitch.lengthM, absolute.x, absolute.y,
      foeVelocity.x, foeVelocity.y, foe.definition.role === "goalkeeper" ? 1 : 0,
      clamp(Math.hypot(relative.x, relative.y) / pitch.lengthM, 0, 1),
      clamp(Math.hypot(foe.position.x - ball.position.x, foe.position.y - ball.position.y) / pitch.lengthM, 0, 1),
    );
  }
  row.close();
}

export type { AttackDirection };
