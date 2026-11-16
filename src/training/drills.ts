import type { MatchConfig, ScenarioDefinition, ScenarioPlayerState, Vec2 } from "../contracts/index.js";

export type DrillKind = "shoot" | "attack" | "pass" | "three" | "counter" | "defend" | "defend2" | "giveandgo" | "rondo" | "keeper" | "cross" | "lob" | "lane";

const MIN_SPACING_M = 1.2;
const MAX_PLACEMENT_ATTEMPTS = 50;

export const DRILL_KINDS: readonly DrillKind[] = ["shoot", "attack", "pass", "three", "counter", "defend", "defend2", "giveandgo", "rondo", "keeper", "cross", "lob", "lane"];

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
 *   giveandgo a defender presses the carrier from the front, and a teammate stands to the side and slightly ahead;
 *           the way through is a pass to the teammate and a return pass behind the defender
 *   rondo   keep-away in a small square in midfield: four learner players around the square and two rivals inside it,
 *           the carrier is one of the four. There is no goal to score; train it with a reward for holding the ball.
 *   keeper  a lone rival striker with the ball 9 to 18 m from the learner's goal and no defenders, with a second rival
 *           waiting outside the area to press the keeper's release. Trains the learner's keeper.
 * Everyone else holds a spot away from the action. The learner's outfield players are drawn at random for the roles (carrier,
 * runner and so on), and so are the rivals' outfield players (the rival striker, defenders and so on). Both keepers keep their place.
 */
export function createDrill(kind: DrillKind, config: MatchConfig, learnerIndex: 0 | 1, seed: number): ScenarioDefinition {
  return createDrillWithRoles(kind, config, learnerIndex, seed).scenario;
}

/** Like `createDrill`, and also says who the ball carrier and the runner (the carrier's teammate in the play) are. */
export function createDrillWithRoles(
  kind: DrillKind, config: MatchConfig, learnerIndex: 0 | 1, seed: number,
): { scenario: ScenarioDefinition; carrierId: string; runnerId: string | undefined } {
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

  const field = outfield(learner);
  // Every drill draws its learner outfield players at random (the keeper keeps its place): the ball carrier, then the
  // teammates in the order the drill uses them (the first is the runner or the near option, and so on).
  const carrier = field[Math.floor(next() * field.length)];
  const mates = field.filter((player) => player !== carrier);
  for (let i = mates.length - 1; i > 0; i -= 1) {
    const j = Math.floor(next() * (i + 1));
    [mates[i], mates[j]] = [mates[j], mates[i]];
  }
  /** All learner outfield players in the drawn order, for the drills where the learner defends. */
  const drawn = [carrier, ...mates];
  // The rivals' outfield players are drawn too, from their own generator so the placements above stay the same for a seed.
  const foes = outfield(rival);
  const rivalNext = random((seed ^ 0x5bd1e995) >>> 0);
  for (let i = foes.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rivalNext() * (i + 1));
    [foes[i], foes[j]] = [foes[j], foes[i]];
  }
  const attacker = foes.at(-1)!;
  const attackMates = foes.slice(0, -1);
  const defending = kind === "defend" || kind === "defend2" || kind === "keeper";

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
    } else if (kind === "giveandgo") {
      const side = next() < 0.5 ? -1 : 1;
      ballAt = { x: goalX - between(20, 26), y: side * between(0, 5) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      const [wall] = mates;
      if (wall) put(wall.id, ballAt.x + between(3, 5), ballAt.y + side * between(4, 7));
      mates.slice(1).forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      const [presser, back, ...rest] = foes;
      if (presser) put(presser.id, ballAt.x + between(2.5, 3.5), ballAt.y + between(-0.8, 0.8));
      if (back) put(back.id, goalX - between(10, 14), -side * between(0, 4));
      rest.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    } else if (kind === "rondo") {
      const centre = { x: between(-6, 6), y: between(-4, 4) };
      const radius = between(5.5, 7);
      const turn = between(0, 2 * Math.PI);
      const ring = [carrier, ...mates.slice(0, 3)];
      ring.forEach((player, index) => {
        const angle = turn + (2 * Math.PI * index) / ring.length + between(-0.2, 0.2);
        put(player.id, centre.x + radius * Math.cos(angle), centre.y + radius * Math.sin(angle));
      });
      ballAt = { ...placed.get(carrier.id)! };
      ballAt = { x: ballAt.x + 0.45 * Math.sign(centre.x - ballAt.x || 1), y: ballAt.y };
      foes.slice(0, 2).forEach((foe, index) => put(foe.id, centre.x + (index ? 1.5 : -1.5), centre.y + between(-1.5, 1.5)));
      foes.slice(2).forEach((foe, index) => put(foe.id, goalX - 6 - 2 * index, (index % 2 ? 1 : -1) * between(8, 14)));
      mates.slice(3).forEach((mate, index) => put(mate.id, -goalX + 8 + 2 * index, (index % 2 ? 1 : -1) * between(8, 14)));
    } else if (kind === "cross") {
      // A winger on the touchline with two runners arriving in the box and a defender marking one of them.
      const side = next() < 0.5 ? -1 : 1;
      ballAt = { x: goalX - between(8, 14), y: side * between(13, 17) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      const [near, far] = mates;
      if (near) put(near.id, goalX - between(4, 8), -side * between(0.5, 3.5));
      if (far) put(far.id, goalX - between(7, 11), side * between(2, 5));
      mates.slice(2).forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      const [marker, presser, ...rest] = foes;
      if (marker) put(marker.id, goalX - between(5, 8), -side * between(2, 5));
      if (presser) put(presser.id, ballAt.x - between(4, 7), ballAt.y - side * between(1, 3));
      rest.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    } else if (kind === "lane") {
      // Two outlets: one straight ahead behind a defender who blocks the lane, one to the side with a clear lane.
      const side = next() < 0.5 ? -1 : 1;
      ballAt = { x: goalX - between(20, 26), y: side * between(0, 4) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      const [blocked, free] = mates;
      if (blocked) put(blocked.id, ballAt.x + between(8, 11), ballAt.y + between(-1, 1));
      if (free) put(free.id, ballAt.x + between(3, 6), ballAt.y - side * between(7, 10));
      mates.slice(2).forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      const [wall, presser, ...rest] = foes;
      if (wall) put(wall.id, ballAt.x + between(4, 5.5), ballAt.y + between(-0.5, 0.5));
      if (presser) put(presser.id, ballAt.x - between(2.5, 3.5), ballAt.y + between(-1.5, 1.5));
      rest.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    } else if (kind === "lob") {
      // A defender stands in the passing lane and a runner waits behind him: only a high ball gets through.
      const side = next() < 0.5 ? -1 : 1;
      ballAt = { x: goalX - between(22, 28), y: side * between(0, 5) };
      put(carrier.id, ballAt.x - 0.45, ballAt.y);
      const [runner] = mates;
      if (runner) put(runner.id, ballAt.x + between(9, 12), ballAt.y + between(-1.5, 1.5));
      mates.slice(1).forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
      const [wall, back, ...rest] = foes;
      if (wall) put(wall.id, ballAt.x + between(3.5, 5), ballAt.y + between(-0.8, 0.8));
      if (back) put(back.id, goalX - between(8, 12), between(-4, 4));
      rest.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
    } else if (kind === "keeper") {
      ballAt = { x: -goalX + between(9, 18), y: between(-7, 7) };
      put(attacker.id, ballAt.x + 0.45, ballAt.y);
      const [presser, ...others] = attackMates;
      if (presser) put(presser.id, -goalX + between(11, 15), between(-6, 6));
      others.forEach((foe, index) => put(foe.id, goalX - 6 - 2 * index, (index % 2 ? 1 : -1) * between(8, 14)));
      drawn.forEach((player, index) => put(player.id, between(2, 10) + 2 * index, (index % 2 ? 1 : -1) * between(4, 12)));
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
      const defenders = drawn;
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
  const scenario: ScenarioDefinition = {
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
  return { scenario, carrierId: carrier.id, runnerId: mates[0]?.id };
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
