import type {
  AttackDirection, BallState, MatchConfig, MatchEvent, MatchPhase, PitchConfig, PlayerDefinition,
  PlayerId, PlayerState, RestartKind, ScenarioDefinition, TeamId, Vec2, Vec3,
} from "../contracts/index.js";
import { capsuleDistanceSquared, goalFrameCapsules, playerCapsule } from "./physics-geometry.js";
import { kickoffTeamId } from "./random.js";

export interface EngineState {
  config: MatchConfig;
  tick: number;
  phase: MatchPhase;
  period: 1 | 2;
  remainingPeriodTicks: number;
  score: [number, number];
  attackDirections: [AttackDirection, AttackDirection];
  ball: BallState;
  players: PlayerState[];
  events: MatchEvent[];
  rngState: number;
  keeperReleaseTick: number | null;
  touchExclusion: { playerId: PlayerId; untilTick: number } | null;
  nextDribbleTouchTick: number;
  requestIds: Map<PlayerId, number>;
  nextRuleRequestId: number;
}

type ResetErrorKind = "configuration" | "scenario";

export class ResetError extends Error {
  override name = "ResetError";

  constructor(
    readonly kind: ResetErrorKind,
    readonly field: string,
    detail: string,
  ) {
    super(`${field}: ${detail}`);
  }
}

const tolerance = 1e-8;
const maxSimulationValue = 1e6;

function check(condition: unknown, kind: ResetErrorKind, field: string, detail: string): asserts condition {
  if (!condition) {
    throw new ResetError(kind, field, detail);
  }
}

function finite(value: unknown, kind: ResetErrorKind, field: string): asserts value is number {
  check(typeof value === "number" && Number.isFinite(value), kind, field, "expected a finite number");
}

function inRange(value: number, min: number, max: number, kind: ResetErrorKind, field: string): void {
  finite(value, kind, field);
  check(value >= min && value <= max, kind, field, `expected a value from ${min} to ${max}`);
}

function integer(value: number, min: number, max: number, kind: ResetErrorKind, field: string): void {
  inRange(value, min, max, kind, field);
  check(Number.isSafeInteger(value), kind, field, "expected an integer");
}

function positive(value: number, field: string): void {
  finite(value, "configuration", field);
  check(value > 0 && value <= maxSimulationValue, "configuration", field,
    `expected a positive value no larger than ${maxSimulationValue}`);
}

function vec2(value: Vec2, kind: ResetErrorKind, field: string): void {
  check(value && typeof value === "object", kind, field, "expected a vector");
  finite(value.x, kind, `${field}.x`);
  finite(value.y, kind, `${field}.y`);
}

function vec3(value: Vec3, kind: ResetErrorKind, field: string): void {
  vec2(value, kind, field);
  finite(value.z, kind, `${field}.z`);
}

function compareIds(first: string, second: string): number {
  return first < second ? -1 : first > second ? 1 : 0;
}

function distance(first: Vec2, second: Vec2): number {
  return Math.hypot(first.x - second.x, first.y - second.y);
}

function insidePitch(position: Vec2, pitch: PitchConfig, radius: number): boolean {
  return Math.abs(position.x) <= pitch.lengthM / 2 - radius + tolerance &&
    Math.abs(position.y) <= pitch.widthM / 2 - radius + tolerance;
}

function validatePlayer(definition: PlayerDefinition, pitch: PitchConfig, field: string): void {
  const { abilities } = definition;
  check(definition.role === "outfield" || definition.role === "goalkeeper",
    "configuration", `${field}.role`, "expected outfield or goalkeeper");
  check(definition.preferredFoot === "left" || definition.preferredFoot === "right",
    "configuration", `${field}.preferredFoot`, "expected left or right");
  positive(definition.bodyHeightM, `${field}.bodyHeightM`);
  positive(definition.radiusM, `${field}.radiusM`);
  positive(definition.massKg, `${field}.massKg`);
  vec2(definition.formation, "configuration", `${field}.formation`);
  check(insidePitch(definition.formation, pitch, definition.radiusM),
    "configuration", `${field}.formation`, "player body must fit inside the pitch");
  check(abilities && abilities.physical && abilities.technical,
    "configuration", `${field}.abilities`, "expected physical and technical abilities");

  const physical = abilities.physical;
  const positiveFields = [
    "maxSpeedMps", "accelerationMps2", "brakingMps2", "turnRateRadPerSec",
    "maxKickSpeedMps", "maxThrowInSpeedMps", "maxKeeperRollSpeedMps",
    "maxKeeperThrowSpeedMps", "reachM",
  ] as const;
  for (const key of positiveFields) {
    positive(physical[key], `${field}.abilities.physical.${key}`);
  }
  for (const key of ["staminaDrainPerSec", "staminaRecoveryPerSec"] as const) {
    inRange(physical[key], 0, 1, "configuration", `${field}.abilities.physical.${key}`);
  }
  const technicalFields = ["passing", "shooting", "firstTouch", "dribbling", "tackling", "goalkeeping", "weakFoot"] as const;
  for (const key of technicalFields) {
    inRange(abilities.technical[key], 0, 1, "configuration", `${field}.abilities.technical.${key}`);
  }
}

function validateConfig(config: MatchConfig, seed: number): void {
  check(config && typeof config === "object", "configuration", "config", "expected a match configuration");
  integer(seed, 0, 0xffffffff, "configuration", "seed");

  const { pitch, ball, teams, rules, tuning, periodTicks } = config;
  check(pitch && ball && rules && tuning, "configuration", "config", "missing pitch, ball, rules, or tuning");
  check(rules.version === 1, "configuration", "rules.version", "unsupported version");

  const pitchDimensions = [
    "lengthM", "widthM", "goalWidthM", "goalHeightM", "postRadiusM", "goalAreaDepthM",
    "goalAreaWidthM", "keeperAreaDepthM", "keeperAreaWidthM", "centerCircleRadiusM",
  ] as const;
  for (const key of pitchDimensions) {
    positive(pitch[key], `pitch.${key}`);
  }
  check(pitch.goalWidthM + 4 * pitch.postRadiusM <= pitch.widthM,
    "configuration", "pitch.goalWidthM", "goal and posts must fit within the pitch width");
  check(pitch.centerCircleRadiusM <= Math.min(pitch.lengthM, pitch.widthM) / 2,
    "configuration", "pitch.centerCircleRadiusM", "center circle must fit on the pitch");
  for (const [label, depth, width] of [
    ["goalArea", pitch.goalAreaDepthM, pitch.goalAreaWidthM],
    ["keeperArea", pitch.keeperAreaDepthM, pitch.keeperAreaWidthM],
  ] as const) {
    check(depth <= pitch.lengthM / 2, "configuration", `pitch.${label}DepthM`, "area must fit within one half");
    check(width <= pitch.widthM, "configuration", `pitch.${label}WidthM`, "area must fit within pitch width");
  }

  for (const key of ["radiusM", "massKg", "gravityMps2", "maxSpeedMps"] as const) {
    positive(ball[key], `ball.${key}`);
  }
  check(2 * ball.radiusM < Math.min(pitch.goalWidthM, pitch.goalHeightM),
    "configuration", "ball.radiusM", "ball must fit through the goal opening");
  inRange(ball.groundDecelerationMps2, 0, maxSimulationValue, "configuration", "ball.groundDecelerationMps2");
  inRange(ball.groundRestitution, 0, 1, "configuration", "ball.groundRestitution");
  inRange(ball.contactRestitution, 0, 1, "configuration", "ball.contactRestitution");

  check(Array.isArray(periodTicks) && periodTicks.length === 2,
    "configuration", "periodTicks", "expected two period lengths");
  for (const [index, ticks] of periodTicks.entries()) {
    integer(ticks, 1, Number.MAX_SAFE_INTEGER, "configuration", `periodTicks[${index}]`);
  }
  for (const key of ["restartPlacementTicks", "goalPauseTicks", "halftimeTicks"] as const) {
    integer(rules[key], 0, Number.MAX_SAFE_INTEGER, "configuration", `rules.${key}`);
  }
  for (const key of ["restartInputTicks", "keeperHoldTicks"] as const) {
    integer(rules[key], 1, Number.MAX_SAFE_INTEGER, "configuration", `rules.${key}`);
  }
  if (rules.restartSupportM !== undefined) {
    inRange(rules.restartSupportM, 0, Math.hypot(pitch.lengthM, pitch.widthM), "configuration", "rules.restartSupportM");
  }
  if (rules.headerContact !== undefined) {
    check(typeof rules.headerContact === "boolean", "configuration", "rules.headerContact", "expected a boolean");
  }
  inRange(rules.restartExclusionM, 0, Math.hypot(pitch.lengthM, pitch.widthM),
    "configuration", "rules.restartExclusionM");

  check(Array.isArray(teams) && teams.length === 2, "configuration", "teams", "expected two teams");
  const teamIds = new Set<TeamId>();
  const playerIds = new Set<PlayerId>();
  for (const [index, team] of teams.entries()) {
    const field = `teams[${index}]`;
    check(team && typeof team.id === "string" && team.id.length > 0 && !teamIds.has(team.id),
      "configuration", `${field}.id`, "expected a unique nonempty team ID");
    teamIds.add(team.id);
    check(Array.isArray(team.players) && team.players.length > 0,
      "configuration", `${field}.players`, "expected a nonempty roster");

    let goalkeeperCount = 0;
    for (const [playerIndex, player] of team.players.entries()) {
      const playerField = `${field}.players[${playerIndex}]`;
      check(player && typeof player.id === "string" && player.id.length > 0 && !playerIds.has(player.id),
        "configuration", `${playerField}.id`, "expected a unique nonempty player ID");
      playerIds.add(player.id);
      validatePlayer(player, pitch, playerField);
      goalkeeperCount += Number(player.role === "goalkeeper");
    }
    check(goalkeeperCount === 1, "configuration", `${field}.players`, "expected exactly one goalkeeper");
  }
  check(teamIds.has(config.firstKickoffTeamId), "configuration", "firstKickoffTeamId", "unknown team");
}

export function attackDirection(state: EngineState, teamId: TeamId): AttackDirection {
  return state.attackDirections[state.config.teams[0].id === teamId ? 0 : 1];
}

export function inKeeperArea(state: EngineState, player: PlayerState): boolean {
  const pitch = state.config.pitch;
  const ownGoalX = -attackDirection(state, player.teamId) * pitch.lengthM / 2;
  return Math.abs(player.position.x - ownGoalX) <= pitch.keeperAreaDepthM + tolerance &&
    Math.abs(player.position.y) <= pitch.keeperAreaWidthM / 2 + tolerance;
}

export function keeperHoldPosition(player: PlayerState, ballRadiusM: number, pitch: PitchConfig): Vec3 {
  const forward = player.definition.radiusM + ballRadiusM;
  const limitX = Math.max(0, pitch.lengthM / 2 - ballRadiusM - pitch.postRadiusM);
  const limitY = pitch.widthM / 2 - ballRadiusM;
  const x = player.position.x + Math.cos(player.facingRad) * forward;
  const y = player.position.y + Math.sin(player.facingRad) * forward;
  return {
    x: Math.max(-limitX, Math.min(limitX, x)),
    y: Math.max(-limitY, Math.min(limitY, y)),
    z: Math.max(ballRadiusM, player.definition.bodyHeightM * 0.72),
  };
}

export function inDribbleReach(player: PlayerState, position: Vec3, ballRadiusM: number): boolean {
  return distance(position, player.position) <= player.definition.abilities.physical.reachM + tolerance &&
    position.z <= ballRadiusM + player.definition.radiusM + tolerance;
}

function defaultPlayers(config: MatchConfig, period: 1 | 2): PlayerState[] {
  const periodDirection = period === 1 ? 1 : -1;
  return config.teams.flatMap((team, index) => {
    const direction = index === 0 ? periodDirection : -periodDirection;
    return team.players.map((definition) => ({
      playerId: definition.id,
      teamId: team.id,
      definition,
      position: { x: direction * definition.formation.x, y: direction * definition.formation.y },
      velocity: { x: 0, y: 0 },
      facingRad: direction === 1 ? 0 : Math.PI,
      stamina: 1,
      balance: 1,
    }));
  }).sort((a, b) => compareIds(a.playerId, b.playerId));
}

function validateScenarioPlayers(state: EngineState, scenario: ScenarioDefinition): void {
  const { players, config: { pitch } } = state;
  check(Array.isArray(scenario.players) && scenario.players.length === players.length,
    "scenario", "players", "expected exactly one state per configured player");
  const byId = new Map(players.map((player) => [player.playerId, player]));
  for (const [index, input] of scenario.players.entries()) {
    const field = `players[${index}]`;
    const player = input && byId.get(input.playerId);
    check(player, "scenario", `${field}.playerId`, "unknown or duplicate player");
    byId.delete(input.playerId);

    vec2(input.position, "scenario", `${field}.position`);
    check(insidePitch(input.position, pitch, player.definition.radiusM),
      "scenario", `${field}.position`, "player body must fit inside the pitch");
    finite(input.facingRad, "scenario", `${field}.facingRad`);
    const velocity = input.velocity === undefined ? { x: 0, y: 0 } : input.velocity;
    vec2(velocity, "scenario", `${field}.velocity`);
    check(Math.hypot(velocity.x, velocity.y) <= player.definition.abilities.physical.maxSpeedMps + tolerance,
      "scenario", `${field}.velocity`, "exceeds player speed limit");
    const stamina = input.stamina === undefined ? 1 : input.stamina;
    const balance = input.balance === undefined ? 1 : input.balance;
    inRange(stamina, 0, 1, "scenario", `${field}.stamina`);
    inRange(balance, 0, 1, "scenario", `${field}.balance`);
    player.position = { ...input.position };
    player.velocity = { ...velocity };
    player.facingRad = input.facingRad;
    player.stamina = stamina;
    player.balance = balance;
  }
  for (const [index, player] of players.entries()) {
    for (const other of players.slice(index + 1)) {
      check(distance(player.position, other.position) + tolerance >= player.definition.radiusM + other.definition.radiusM,
        "scenario", "players", `player bodies overlap: ${player.playerId} and ${other.playerId}`);
    }
  }
}

function validateBall(state: EngineState, scenario: ScenarioDefinition): BallState {
  const { ball } = scenario;
  const { pitch, ball: config } = state.config;
  check(ball && typeof ball === "object", "scenario", "ball", "expected a ball state");
  vec3(ball.position, "scenario", "ball.position");
  vec3(ball.velocity, "scenario", "ball.velocity");
  check(Math.hypot(ball.velocity.x, ball.velocity.y, ball.velocity.z) <= config.maxSpeedMps + tolerance,
    "scenario", "ball.velocity", "exceeds ball speed limit");
  check(state.players.some((player) => player.playerId === ball.lastTouchPlayerId),
    "scenario", "ball.lastTouchPlayerId", "unknown player");

  const control = ball.control;
  check(control && (control.kind === "free" || control.kind === "dribble" || control.kind === "keeperHeld"),
    "scenario", "ball.control", "unknown control kind");
  const owner = control.kind === "free" ? undefined : state.players.find((player) => player.playerId === control.playerId);
  if (control.kind !== "free") {
    check(owner, "scenario", "ball.control.playerId", "unknown player");
    check(ball.lastTouchPlayerId === owner.playerId,
      "scenario", "ball.lastTouchPlayerId", "ball owner must be the most recent player to touch it");
  }
  const copy = { ...ball, position: { ...ball.position }, velocity: { ...ball.velocity }, control: { ...control } };
  if (scenario.phase.kind !== "live") return copy;

  check(insidePitch(ball.position, pitch, config.radiusM) && ball.position.z >= config.radiusM - tolerance,
    "scenario", "ball.position", "ball must fit inside pitch boundaries and above the ground");
  for (const frame of goalFrameCapsules(pitch)) {
    check(capsuleDistanceSquared(ball.position, frame) >= Math.max(0, frame.radiusM + config.radiusM - tolerance) ** 2,
      "scenario", "ball.position", "ball overlaps the goal frame");
  }
  if (control.kind === "free") {
    for (const player of state.players) {
      const capsule = playerCapsule(player);
      check(capsuleDistanceSquared(ball.position, capsule) >= Math.max(0, capsule.radiusM + config.radiusM - tolerance) ** 2,
        "scenario", "ball.position", `free ball overlaps player ${player.playerId}`);
    }
  } else if (owner) {
    if (control.kind === "dribble") {
      check(inDribbleReach(owner, ball.position, config.radiusM),
      "scenario", "ball.control", "ball is outside the player's dribbling reach");
    } else {
      check(owner.definition.role === "goalkeeper" && inKeeperArea(state, owner),
        "scenario", "ball.control", "keeper hold requires the goalkeeper in its handling area");
      const hold = keeperHoldPosition(owner, config.radiusM, pitch);
      check(Math.hypot(ball.position.x - hold.x, ball.position.y - hold.y, ball.position.z - hold.z) <= tolerance,
        "scenario", "ball.position", "ball must be at the goalkeeper's hold point");
      check(Math.hypot(ball.velocity.x - owner.velocity.x, ball.velocity.y - owner.velocity.y, ball.velocity.z) <= tolerance,
        "scenario", "ball.velocity", "held ball must move with the goalkeeper");
    }
  }
  return copy;
}

function validateRestartSpot(state: EngineState, teamId: TeamId, restart: RestartKind, spot: Vec2): void {
  const { pitch, ball } = state.config;
  const halfLength = pitch.lengthM / 2;
  const halfWidth = pitch.widthM / 2;
  const goalDistance = halfLength + attackDirection(state, teamId) * spot.x;
  const onTouchline = Math.abs(Math.abs(spot.y) - halfWidth) <= tolerance;
  const onGoalLine = Math.abs(Math.abs(spot.x) - halfLength) <= tolerance;
  const legal = restart === "kickoff" ? distance(spot, { x: 0, y: 0 }) <= tolerance
    : restart === "throwIn" ? onTouchline && Math.abs(spot.x) <= halfLength + tolerance
    : restart === "corner" ? onTouchline && onGoalLine
    : insidePitch(spot, pitch, ball.radiusM) &&
      goalDistance <= pitch.goalAreaDepthM + tolerance &&
      Math.abs(spot.y) <= pitch.goalAreaWidthM / 2 + tolerance;
  check(legal, "scenario", "phase.spot", "invalid placement for the restart");
}

function chooseTaker(state: EngineState, teamId: TeamId, restart: RestartKind, spot: Vec2): PlayerState {
  const teammates = state.players.filter((player) => player.teamId === teamId);
  const keeper = teammates.find((player) => player.definition.role === "goalkeeper")!;
  if (restart === "goalKick") return keeper;

  let nearest: PlayerState | undefined;
  for (const player of teammates) {
    if (player.definition.role === "outfield" &&
      (!nearest || distance(player.position, spot) < distance(nearest.position, spot))) {
      nearest = player;
    }
  }
  return nearest ?? keeper;
}

function takerPosition(state: EngineState, player: PlayerState, restart: RestartKind, spot: Vec2): Vec2 {
  const radius = player.definition.radiusM;
  const offset = radius + state.config.ball.radiusM;
  if (restart === "throwIn") {
    return { x: spot.x, y: spot.y + Math.sign(spot.y) * offset };
  }
  if (restart === "corner") {
    const diagonal = offset / Math.sqrt(2);
    return { x: spot.x + Math.sign(spot.x) * diagonal, y: spot.y + Math.sign(spot.y) * diagonal };
  }

  const direction = attackDirection(state, player.teamId);
  const behind = { x: spot.x - direction * offset, y: spot.y };
  if (insidePitch(behind, state.config.pitch, radius)) return behind;

  const alternatives = [
    { x: spot.x, y: spot.y + offset },
    { x: spot.x, y: spot.y - offset },
    { x: spot.x + direction * offset, y: spot.y },
  ];
  const position = alternatives.find((candidate) => insidePitch(candidate, state.config.pitch, radius));
  check(position, "configuration", "pitch", "cannot place the restart taker");
  return position;
}

/**
 * Where a player starts a restart. With `rules.restartSupportM`, the awarded team's outfield players close in on the
 * ball along the line to where they stood: the nearest ends up within one step, the next within two, and so on.
 */
function supportPosition(
  state: EngineState, player: PlayerState, restart: RestartKind, awardedTeam: TeamId, spot: Vec2, takerId: PlayerId,
): Vec2 {
  const step = state.config.rules.restartSupportM;
  if (!step || (restart !== "throwIn" && restart !== "corner") || player.teamId !== awardedTeam ||
      player.definition.role === "goalkeeper") return player.position;
  const mates = state.players
    .filter((other) => other.teamId === awardedTeam && other.definition.role !== "goalkeeper" &&
      other.playerId !== takerId)
    .map((other) => ({ id: other.playerId, d: distance(other.position, spot) }))
    .sort((a, b) => a.d - b.d || compareIds(a.id, b.id));
  const rank = mates.findIndex((mate) => mate.id === player.playerId);
  const away = distance(player.position, spot);
  const limit = step * (rank + 1);
  if (rank < 0 || away <= limit || away === 0) return player.position;
  const scale = limit / away;
  return { x: spot.x + (player.position.x - spot.x) * scale, y: spot.y + (player.position.y - spot.y) * scale };
}

function placeOtherPlayer(
  state: EngineState,
  player: PlayerState,
  restart: RestartKind,
  awardedTeam: TeamId,
  spot: Vec2,
  occupied: readonly PlayerState[],
): Vec2 {
  const { pitch, rules, ball } = state.config;
  const { radiusM } = player.definition;
  const attack = attackDirection(state, player.teamId);
  let minX = -pitch.lengthM / 2 + radiusM;
  let maxX = pitch.lengthM / 2 - radiusM;
  if (restart === "kickoff") {
    if (attack === 1) maxX = -radiusM;
    else minX = radiusM;
  }
  const maxY = pitch.widthM / 2 - radiusM;
  check(minX <= maxX && maxY >= 0, "configuration", "pitch", "cannot fit player body in the allowed area");
  const exclusion = player.teamId === awardedTeam ? 0
    : restart === "kickoff" ? Math.max(rules.restartExclusionM, pitch.centerCircleRadiusM)
    : rules.restartExclusionM;
  const minDistance = Math.max(radiusM + ball.radiusM, exclusion);
  const project = (position: Vec2): Vec2 => ({
    x: Math.max(minX, Math.min(maxX, position.x)),
    y: Math.max(-maxY, Math.min(maxY, position.y)),
  });
  const legal = (position: Vec2): boolean =>
    distance(position, spot) + tolerance >= minDistance &&
    occupied.every((other) => distance(position, other.position) + tolerance >= radiusM + other.definition.radiusM);

  const desired = restart === "kickoff"
    ? { x: player.definition.formation.x * attack, y: player.definition.formation.y * attack }
    : supportPosition(state, player, restart, awardedTeam, spot, occupied[0].playerId);
  const start = project(desired);
  if (legal(start)) return start;

  // Search nearby radial offsets first, then a bounded pitch-wide grid. Stable order breaks ties.
  let best: Vec2 | undefined;
  let bestDistance = Infinity;
  function consider(candidate: Vec2): void {
    const position = project(candidate);
    const displacement = distance(position, start);
    if (displacement < bestDistance && legal(position)) {
      best = position;
      bestDistance = displacement;
    }
  }

  const step = Math.max(radiusM, ball.radiusM);
  for (let ring = 1; ring <= 8; ring++) {
    for (let index = 0; index < 16; index++) {
      const angle = index * Math.PI / 8;
      consider({ x: start.x + Math.cos(angle) * ring * step, y: start.y + Math.sin(angle) * ring * step });
    }
    if (best !== undefined && bestDistance <= ring * step) return best;
  }
  const gridSize = 32;
  for (let x = 0; x <= gridSize; x++) {
    for (let y = 0; y <= gridSize; y++) {
      consider({
        x: minX + (maxX - minX) * x / gridSize,
        y: -maxY + 2 * maxY * y / gridSize,
      });
    }
  }
  check(best, "configuration", "pitch", `cannot place player ${player.playerId} for ${restart}`);
  return best;
}

export function placeRestart(state: EngineState, teamId: TeamId, restart: RestartKind, spot: Vec2): void {
  const taker = chooseTaker(state, teamId, restart, spot);
  const placed: PlayerState[] = [{ ...taker, position: takerPosition(state, taker, restart, spot) }];
  for (const player of state.players) {
    if (player === taker) continue;
    const position = placeOtherPlayer(state, player, restart, teamId, spot, placed);
    placed.push({ ...player, position });
  }

  state.players = placed.sort((a, b) => compareIds(a.playerId, b.playerId));
  for (const player of state.players) {
    player.velocity = { x: 0, y: 0 };
    player.facingRad = player.playerId === taker.playerId
      ? Math.atan2(spot.y - player.position.y, spot.x - player.position.x)
      : attackDirection(state, player.teamId) === 1 ? 0 : Math.PI;
  }

  state.ball = {
    position: { ...spot, z: state.config.ball.radiusM },
    velocity: { x: 0, y: 0, z: 0 },
    lastTouchPlayerId: null,
    control: { kind: "free" },
  };
  state.keeperReleaseTick = null;
  state.touchExclusion = null;
  state.nextDribbleTouchTick = 0;

  const readyTick = state.tick + state.config.rules.restartPlacementTicks;
  const placement = {
    teamId,
    takerId: taker.playerId,
    spot: { ...spot },
    readyTick,
    deadlineTick: readyTick + state.config.rules.restartInputTicks,
  };
  state.phase = restart === "kickoff"
    ? { kind: "kickoffSetup", ...placement }
    : { kind: "restartSetup", restart, ...placement };
}

function validateScenarioHeader(config: MatchConfig, scenario: ScenarioDefinition): void {
  check(scenario && typeof scenario === "object", "scenario", "scenario", "expected a scenario definition");
  check(scenario.version === 1, "scenario", "version", "unsupported version");
  check(typeof scenario.id === "string" && scenario.id.length > 0, "scenario", "id", "expected a nonempty ID");
  check(scenario.period === 1 || scenario.period === 2, "scenario", "period", "expected period 1 or 2");
  if (scenario.remainingPeriodTicks !== undefined) {
    integer(scenario.remainingPeriodTicks, 1, config.periodTicks[scenario.period === 1 ? 0 : 1],
      "scenario", "remainingPeriodTicks");
  }
  if (scenario.score !== undefined) {
    check(Array.isArray(scenario.score) && scenario.score.length === 2,
      "scenario", "score", "expected two scores");
    for (const [index, score] of scenario.score.entries()) {
      integer(score, 0, Number.MAX_SAFE_INTEGER, "scenario", `score[${index}]`);
    }
  }
  const { phase } = scenario;
  check(phase && (phase.kind === "live" || phase.kind === "kickoffSetup" || phase.kind === "restartSetup"),
    "scenario", "phase", "unknown scenario phase");
  if (phase.kind !== "live") {
    check(config.teams.some((team) => team.id === phase.teamId), "scenario", "phase.teamId", "unknown team");
  }
  if (phase.kind === "restartSetup") {
    check(phase.restart === "throwIn" || phase.restart === "corner" || phase.restart === "goalKick",
      "scenario", "phase.restart", "unknown restart kind");
    vec2(phase.spot, "scenario", "phase.spot");
  }
}

export function buildResetState(config: MatchConfig, seed: number, scenario?: ScenarioDefinition): EngineState {
  validateConfig(config, seed);
  if (scenario !== undefined) validateScenarioHeader(config, scenario);

  const configCopy = structuredClone(config);
  const period = scenario?.period ?? 1;
  const state: EngineState = {
    config: configCopy,
    tick: 0,
    phase: { kind: "live" },
    period,
    remainingPeriodTicks: scenario?.remainingPeriodTicks ?? configCopy.periodTicks[period === 1 ? 0 : 1],
    score: scenario?.score ? [...scenario.score] : [0, 0],
    attackDirections: period === 1 ? [1, -1] : [-1, 1],
    ball: {
      position: { x: 0, y: 0, z: configCopy.ball.radiusM },
      velocity: { x: 0, y: 0, z: 0 },
      lastTouchPlayerId: null,
      control: { kind: "free" },
    },
    players: defaultPlayers(configCopy, period),
    events: [],
    rngState: seed,
    keeperReleaseTick: null,
    touchExclusion: null,
    nextDribbleTouchTick: 0,
    requestIds: new Map(),
    nextRuleRequestId: 1,
  };

  // Check both kickoff layouts before accepting the match, even when it starts in open play.
  for (const team of configCopy.teams) {
    const kickoffCheck = { ...state, players: defaultPlayers(configCopy, period) };
    placeRestart(kickoffCheck, team.id, "kickoff", { x: 0, y: 0 });
  }
  if (scenario === undefined) {
    placeRestart(state, kickoffTeamId(state, configCopy.firstKickoffTeamId), "kickoff", { x: 0, y: 0 });
    return state;
  }

  validateScenarioPlayers(state, scenario);
  state.ball = validateBall(state, scenario);
  const { phase } = scenario;
  if (phase.kind === "live") {
    if (state.ball.control.kind === "keeperHeld") {
      state.keeperReleaseTick = configCopy.rules.keeperHoldTicks;
    }
  } else if (phase.kind === "kickoffSetup") {
    placeRestart(state, phase.teamId, "kickoff", { x: 0, y: 0 });
  } else {
    validateRestartSpot(state, phase.teamId, phase.restart, phase.spot);
    placeRestart(state, phase.teamId, phase.restart, phase.spot);
  }
  return state;
}
