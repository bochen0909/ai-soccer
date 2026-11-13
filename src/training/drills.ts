import type { MatchConfig, ScenarioDefinition, ScenarioPlayerState, Vec2 } from "../contracts/index.js";

export type DrillKind = "shoot" | "attack";

export const DRILL_KINDS: readonly DrillKind[] = ["shoot", "attack"];

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

  const placed = new Map<string, Vec2>();
  const put = (id: string, x: number, y: number) => {
    placed.set(id, { x: Math.max(-goalX + 1, Math.min(goalX - 1, x)), y: Math.max(-edgeY, Math.min(edgeY, y)) });
  };

  const carrier = outfield(learner).at(-1)!;
  const mates = outfield(learner).filter((player) => player !== carrier);
  const foes = outfield(rival);
  put(keeper(learner).id, -goalX + 1.5, 0);
  put(keeper(rival).id, goalX - 1.5, between(-1.5, 1.5));

  let ballAt: Vec2;
  if (kind === "shoot") {
    ballAt = { x: goalX - between(8, 16), y: between(-6, 6) };
    put(carrier.id, ballAt.x - 0.45, ballAt.y);
    // Teammates stay behind the play. Rivals start deep in their half of the pitch, 14 m or more away.
    mates.forEach((mate, index) => put(mate.id, -8 - 3 * index, (index % 2 ? 1 : -1) * between(6, 12)));
    foes.forEach((foe, index) => put(foe.id, ballAt.x - between(14, 22) - index, (index % 2 ? 1 : -1) * between(2, 12)));
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

  const players: ScenarioPlayerState[] = [...learner, ...rival].map(({ id }) => {
    const spot = placed.get(id);
    if (!spot) throw new Error(`Drill "${kind}" left player ${id} without a position`);
    return {
      playerId: id,
      position: { x: spot.x * attack, y: spot.y * attack },
      facingRad: attack === 1 ? 0 : Math.PI,
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
      lastTouchPlayerId: carrier.id,
      control: { kind: "free" },
    },
  };
}
