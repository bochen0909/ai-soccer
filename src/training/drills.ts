import type { MatchConfig, ScenarioDefinition, ScenarioPlayerState, Vec2 } from "../contracts/index.js";

export type DrillKind = "shoot" | "attack" | "pass" | "three" | "counter" | "defend" | "defend2";

const MIN_SPACING_M = 1.2;
const MAX_PLACEMENT_ATTEMPTS = 50;

export const DRILL_KINDS: readonly DrillKind[] = ["shoot", "attack", "pass", "three", "counter", "defend", "defend2"];

/** Small deterministic generator so the same seed always builds the same drill. */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Builds a live-play scenario for short training drills. Coordinates are written for a learner that attacks
 * toward +x and mirrored by a half turn for the other team, so both sides practise the same situation.
 *   shoot   a striker with the ball 8 to 16 m from goal, with the opposing keeper in goal
 *   attack  a two-on-one: the ball carrier and a runner against one defender and the keeper
 *   pass    like attack, but the defender is right in front of the carrier and the runner is open near goal,
 *           so the way through is a pass
 *   three   a three-on-two: the carrier and two runners against two defenders, one of them pressing the carrier
 *   counter the carrier has the ball near the halfway line with two runners ahead; one defender is back and
 *           one presses, so dribbling alone gets caught and a pass releases a runner
 *   defend  a two-on-one against the learner: an opposing carrier and a runner come at the learner's goal, with one
 *           learner defender (and the keeper) back. Defending is the goal, so the learner starts without the ball.
 *   defend2 a three-on-two against the learner, with two learner defenders
 * Everyone else holds a spot away from the action. The striker is the last player on the roster.
 */
export function createDrill(kind: DrillKind, config: MatchConfig, learnerIndex: 0 | 1, seed: number): ScenarioDefinition {
  const next = random(seed);
  const between = (low: number, high: number) => low + (high - low) * next();
  const attack = learnerIndex === 0 ? 1 : -1;
  const goalX = config.pitch.lengthM / 2;
  const edgeY = config.pitch.widthM / 2 - 1.5;
  const learner = config.teams[learnerIndex].players;
  const rival = config.teams[1 - learnerIndex].players;
  const outfield = (players: typeof learner) => players.filter((player) => player.role === "outfield");
  const keeper = (players: typeof learner) => players.find((player) => player.role === "goalkeeper")!;

  let placed = new Map<string, Vec2>();
  const put = (id: string, x: number, y: number) => {
    placed.set(id, { x: Math.max(-goalX + 1, Math.min(goalX - 1, x)), y: Math.max(-edgeY, Math.min(edgeY, y)) });
  };
  /** Everyone needs room around them, and the ball needs room too unless its carrier is next to it. */
  const crowded = (ball: Vec2, carrierId: string): boolean => {
    const spots = [...placed.entries()];
    return spots.some(([id, a], i) =>
      (id !== carrierId && Math.hypot(a.x - ball.x, a.y - ball.y) < MIN_SPACING_M) ||
      spots.slice(i + 1).some(([, b]) => Math.hypot(a.x - b.x, a.y - b.y) < MIN_SPACING_M));
  };

  const carrier = outfield(learner).at(-1)!;
  const mates = outfield(learner).filter((player) => player !== carrier);
  const foes = outfield(rival);
  const attacker = foes.at(-1)!;
  const attackMates = foes.slice(0, -1);
  const defending = kind === "defend" || kind === "defend2";

  let ballAt: Vec2 = { x: 0, y: 0 };
  for (let attempt = 0; attempt < MAX_PLACEMENT_ATTEMPTS; attempt += 1) {
    placed = new Map();
    put(keeper(learner).id, -goalX + 1.5, 0);
    put(keeper(rival).id, goalX - 1.5, between(-1.5, 1.5));
    if (kind === "shoot") {
      ballAt = { x: goalX - between(8, 16), y: between(-6, 6) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      // Teammates stay behind the play. Rivals start deep in their half of the pitch, 14 m or more away.
      mates.forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      foes.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    } else if (kind === "pass") {
      const side = next() < 0.5 ? -1 : 1;
      ballAt = { x: goalX - between(18, 24), y: side * between(0, 5) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      const runner = mates[0];
      if (runner) put(runner.id, goalX - between(10, 14), -side * between(5, 9));
      mates.slice(1).forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      const [defender, ...rest] = foes;
      // Close enough to block the run and the shot, but not touching.
      if (defender) put(defender.id, ballAt.x + between(1.8, 2.8), ballAt.y + between(-0.6, 0.6));
      rest.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    } else if (kind === "defend" || kind === "defend2") {
      // The rival attacks toward -x, so these coordinates are mirrored relative to the other drills.
      const side = next() < 0.5 ? -1 : 1;
      const three = kind === "defend2";
      ballAt = { x: -goalX + between(16, 24), y: side * between(0, 6) };
      put(attacker.id, ballAt.x + 0.45, ballAt.y);
      const [runnerOne, runnerTwo, ...spare] = attackMates;
      if (runnerOne) put(runnerOne.id, ballAt.x - between(5, 9), -side * between(5, 9));
      if (three && runnerTwo) put(runnerTwo.id, ballAt.x - between(3, 7), side * between(7, 11));
      else if (runnerTwo) spare.unshift(runnerTwo);
      spare.forEach((foe, index) => put(foe.id, between(2, 10) + 2 * index, (index % 2 ? 1 : -1) * between(4, 12)));
      const defenders = outfield(learner);
      defenders.forEach((player, index) => {
        if (index === 0) put(player.id, ballAt.x - between(3, 5), ballAt.y + between(-1, 1));
        else if (index === 1 && three) put(player.id, ballAt.x - between(6, 9), -side * between(2, 5));
        else put(player.id, between(2, 10) + 2 * index, (index % 2 ? 1 : -1) * between(4, 12));
      });
    } else if (kind === "three" || kind === "counter") {
      const side = next() < 0.5 ? -1 : 1;
      const counter = kind === "counter";
      ballAt = counter ? { x: between(-6, 4), y: between(-8, 8) } : { x: goalX - between(22, 28), y: side * between(0, 5) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      const [one, two] = mates;
      if (one) put(one.id, ballAt.x + (counter ? between(10, 16) : between(5, 9)), -side * between(6, 10));
      if (two) put(two.id, ballAt.x + (counter ? between(8, 14) : between(3, 7)), side * between(7, 11));
      mates.slice(2).forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      const [back, presser, ...rest] = foes;
      if (back) put(back.id, goalX - between(counter ? 14 : 12, counter ? 20 : 16), between(-4, 4));
      if (presser) put(presser.id, ballAt.x + between(2.5, 4), ballAt.y + between(-1.5, 1.5));
      rest.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    } else {
      const side = next() < 0.5 ? -1 : 1;
      ballAt = { x: goalX - between(20, 26), y: side * between(0, 6) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      const runner = mates[0];
      if (runner) put(runner.id, ballAt.x + between(4, 8), -side * between(5, 10));
      mates.slice(1).forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      const [defender, ...rest] = foes;
      if (defender) put(defender.id, goalX - between(9, 13), between(-3, 3));
      rest.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    }
    if (!crowded(ballAt, defending ? attacker.id : carrier.id)) break;
    if (attempt === MAX_PLACEMENT_ATTEMPTS - 1) throw new Error(`Drill "${kind}" could not find room for every player`);
  }

  const learnerIds = new Set(learner.map(({ id }) => id));
  const players: ScenarioPlayerState[] = [...learner, ...rival].map(({ id }) => {
    const spot = placed.get(id);
    if (!spot) throw new Error(`Drill "${kind}" left player ${id} without a position`);
    return {
      playerId: id,
      position: { x: spot.x * attack, y: spot.y * attack },
      // In defensive drills the rival attacks the other way and faces it.
      facingRad: (attack === 1) === (!defending || learnerIds.has(id)) ? 0 : Math.PI,
    };
  });
  return {
    version: 1,
    id: `drill-${kind}-${seed}`,
    period: 1,
    phase: { kind: "live" },
    players,
    ball: {
      position: { x: ballAt.x * attack, y: ballAt.y * attack, z: config.ball.radiusM },
      velocity: { x: 0, y: 0, z: 0 },
      lastTouchPlayerId: defending ? attacker.id : carrier.id,
      control: { kind: "free" },
    },
  };
}

const swapTeam = (id: string): string => id.startsWith("home") ? `away${id.slice(4)}` : id.startsWith("away") ? `home${id.slice(4)}` : id;

/**
 * Turns a scenario half a turn and swaps the two teams' roles, so a situation recorded for the home team reads the
 * same for the away team. Applying it twice gives back the original.
 */
export function mirrorScenario(scenario: ScenarioDefinition): ScenarioDefinition {
  const flip = (v: Vec2): Vec2 => ({ x: -v.x, y: -v.y });
  const { ball } = scenario;
  return {
    ...scenario,
    id: `${scenario.id}-mirror`,
    score: scenario.score && [scenario.score[1], scenario.score[0]],
    players: scenario.players.map((player) => ({
      ...player,
      playerId: swapTeam(player.playerId),
      position: flip(player.position),
      facingRad: player.facingRad + Math.PI,
      velocity: player.velocity && flip(player.velocity),
    })),
    ball: {
      position: { ...flip(ball.position), z: ball.position.z },
      velocity: { ...flip(ball.velocity), z: ball.velocity.z },
      lastTouchPlayerId: swapTeam(ball.lastTouchPlayerId),
      control: ball.control.kind === "free" ? ball.control : { ...ball.control, playerId: swapTeam(ball.control.playerId) },
    },
  };
}
