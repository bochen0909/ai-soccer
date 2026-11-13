import type { PlayerDefinition, TeamObservation } from "../contracts/index.js";
import { decodeAction, writeActionMask } from "../training/actions.js";
import { buildRosterContexts } from "../training/agents.js";
import { createActionSpace } from "../training/contracts.js";
import type { ActionSpace } from "../training/contracts.js";
import { createObservationLayout, FEEDBACK_STATUSES, writeObservation } from "../training/observation.js";
import type { AIContext, ControlAssignment, PlayerIntent, TeamAI, TeamDecisionContext } from "./contracts.js";
import { TeamAIRegistry } from "./registry.js";

/** A trained policy as written by python/export_policy.py. */
export interface PolicyWeights {
  version: 1;
  obsSize: number;
  hidden: number;
  actCount: number;
  moveCount: number;
  tensors: Readonly<Record<string, { shape: readonly number[]; data: string }>>;
}

export interface Network {
  obsSize: number;
  hidden: number;
  actCount: number;
  moveCount: number;
  w1: Float32Array; b1: Float32Array; w2: Float32Array; b2: Float32Array;
  moveW: Float32Array; moveB: Float32Array; sprintW: Float32Array; sprintB: Float32Array;
  actW: Float32Array; actB: Float32Array;
}

function floats(weights: PolicyWeights, name: string, length: number): Float32Array {
  const tensor = weights.tensors[name];
  if (!tensor) throw new RangeError(`Policy weights are missing "${name}"`);
  const binary = atob(tensor.data);
  if (binary.length !== length * 4) throw new RangeError(`Policy tensor "${name}" has the wrong size`);
  const view = new DataView(new ArrayBuffer(binary.length));
  for (let i = 0; i < binary.length; i += 1) view.setUint8(i, binary.charCodeAt(i));
  const values = new Float32Array(length);
  for (let i = 0; i < length; i += 1) values[i] = view.getFloat32(i * 4, true);
  return values;
}

/** Decodes the base64 tensors once, so deciding does not repeat it. */
export function decodeWeights(weights: PolicyWeights): Network {
  if (weights.version !== 1) throw new RangeError("Unsupported policy version");
  const { obsSize, hidden, actCount, moveCount } = weights;
  return {
    obsSize, hidden, actCount, moveCount,
    w1: floats(weights, "w1", hidden * obsSize), b1: floats(weights, "b1", hidden),
    w2: floats(weights, "w2", hidden * hidden), b2: floats(weights, "b2", hidden),
    moveW: floats(weights, "moveW", moveCount * hidden), moveB: floats(weights, "moveB", moveCount),
    sprintW: floats(weights, "sprintW", 2 * hidden), sprintB: floats(weights, "sprintB", 2),
    actW: floats(weights, "actW", actCount * hidden), actB: floats(weights, "actB", actCount),
  };
}

function affine(
  weights: Float32Array, bias: Float32Array, input: ArrayLike<number>, out: Float64Array,
  rows: number, columns: number, tanh: boolean,
): void {
  for (let row = 0; row < rows; row += 1) {
    let sum = bias[row];
    const base = row * columns;
    for (let column = 0; column < columns; column += 1) sum += weights[base + column] * input[column];
    out[row] = tanh ? Math.tanh(sum) : sum;
  }
}

export interface NeuralOutput {
  move: Float64Array;
  sprint: Float64Array;
  act: Float64Array;
}

/** Logits for one agent's observation row. Illegal acts are not masked here. */
export function forward(network: Network, observation: ArrayLike<number>): NeuralOutput {
  const { obsSize, hidden, actCount, moveCount } = network;
  const h1 = new Float64Array(hidden);
  const h2 = new Float64Array(hidden);
  affine(network.w1, network.b1, observation, h1, hidden, obsSize, true);
  affine(network.w2, network.b2, h1, h2, hidden, hidden, true);
  const move = new Float64Array(moveCount);
  const sprint = new Float64Array(2);
  const act = new Float64Array(actCount);
  affine(network.moveW, network.moveB, h2, move, moveCount, hidden, false);
  affine(network.sprintW, network.sprintB, h2, sprint, 2, hidden, false);
  affine(network.actW, network.actB, h2, act, actCount, hidden, false);
  return { move, sprint, act };
}

function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Picks an index from logits. `legal` hides illegal choices. Greedy takes the highest logit. */
function choose(
  logits: ArrayLike<number>, legal: ((index: number) => boolean) | undefined, greedy: boolean, random: () => number,
): number {
  let best = -1;
  let maximum = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    if (legal && !legal(i)) continue;
    if (logits[i] > maximum) { maximum = logits[i]; best = i; }
  }
  if (greedy) return best;
  let total = 0;
  const weights = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    if (legal && !legal(i)) continue;
    weights[i] = Math.exp(logits[i] - maximum);
    total += weights[i];
  }
  let draw = random() * total;
  for (let i = 0; i < logits.length; i += 1) {
    draw -= weights[i];
    if (weights[i] > 0 && draw <= 0) return i;
  }
  return best;
}

export interface NeuralOptions {
  /** Take the most likely action instead of sampling. Sampling is how the policy was trained and measured. */
  greedy?: boolean;
}

/** Team controller that plays with a trained policy. One network controls every player it is assigned. */
export class NeuralTeamAI implements TeamAI {
  readonly id: string;
  readonly version = "1";
  private teamIndex: 0 | 1 = 0;
  private players: readonly PlayerDefinition[] = [];
  private opponents: readonly PlayerDefinition[] = [];
  private greedy = false;
  private random: () => number = Math.random;
  private readonly space: ActionSpace;
  private readonly layout: ReturnType<typeof createObservationLayout>;

  constructor(private readonly network: Network, id = "neural") {
    this.id = id;
    const rosterSize = (network.obsSize - 63) / 21;
    if (!Number.isInteger(rosterSize) || rosterSize < 1) {
      throw new RangeError("Policy input size does not match any roster size");
    }
    this.layout = createObservationLayout(rosterSize);
    this.space = createActionSpace(rosterSize);
  }

  reset(context: AIContext): void {
    if (context.players.length !== this.layout.maxPlayers || context.opponents?.length !== this.layout.maxPlayers) {
      throw new RangeError(`This policy was trained for ${this.layout.maxPlayers} players per side`);
    }
    const options = (context.config ?? {}) as NeuralOptions;
    this.greedy = options.greedy === true;
    this.teamIndex = context.teamIndex;
    this.players = context.players;
    this.opponents = context.opponents;
    // Seeded so a recorded match replays exactly.
    this.random = mulberry32(context.seed);
  }

  dispose(): void {}

  decide(
    observation: TeamObservation, assignment: ControlAssignment, context: TeamDecisionContext,
  ): readonly PlayerIntent[] {
    const assigned = new Set(assignment.playerIds);
    const feedback = new Map(context.actionResults.map((result) => [result.action.playerId, result.status]));
    const row = new Float32Array(this.layout.size);
    const mask = new Uint8Array(this.space.actCount);
    const intents: PlayerIntent[] = [];
    for (const agent of buildRosterContexts(observation, this.teamIndex, this.players, this.opponents)) {
      if (!assigned.has(agent.ego.playerId)) continue;
      const status = feedback.get(agent.ego.playerId);
      writeObservation(agent, this.layout, status === undefined ? 0 : FEEDBACK_STATUSES.indexOf(status) + 1,
        this.teamIndex, row, 0);
      writeActionMask(agent, this.space, mask, 0);
      const output = forward(this.network, row);
      const move = choose(output.move, undefined, this.greedy, this.random);
      const sprint = choose(output.sprint, undefined, this.greedy, this.random);
      const act = choose(output.act, (index) => mask[index] === 1, this.greedy, this.random);
      const decoded = decodeAction(agent, this.space, move, sprint, act);
      intents.push({
        playerId: agent.ego.playerId, movement: decoded.movement, sprint: decoded.sprint, request: decoded.request,
      });
    }
    return intents;
  }
}

/** Registers an already decoded network as team AI "neural" (or another ID). */
export function registerNeuralNetwork(registry: TeamAIRegistry, network: Network, id = "neural"): TeamAIRegistry {
  return registry.register(id, () => new NeuralTeamAI(network, id));
}

/** Registers the trained policy as team AI "neural" (or another ID). */
export function registerNeuralPolicy(registry: TeamAIRegistry, weights: PolicyWeights, id = "neural"): TeamAIRegistry {
  return registerNeuralNetwork(registry, decodeWeights(weights), id);
}
