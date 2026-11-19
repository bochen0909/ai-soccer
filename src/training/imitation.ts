import type { PlayerIntent } from "../controllers/contracts.js";
import {
  ACT_NONE, ACT_SAVE, ACT_SHOOT_CENTER, ACT_SHOOT_LEFT, ACT_SHOOT_RIGHT, ACT_TACKLE,
} from "./contracts.js";
import type { ActionSpace } from "./contracts.js";
import type { AgentContext } from "./actions.js";
import { toPolicy } from "./geometry.js";
import { ballOf, FIRM_PASS_ARRIVAL_MPS, SOFT_PASS_ARRIVAL_MPS } from "../controllers/baseline.js";
import { rollSpeedAfter } from "../engine/ball-flight.js";

export interface Choice {
  move: number;
  sprint: number;
  act: number;
  /** The controller asked for an act the policy's action set cannot express or the mask forbids. */
  lost: boolean;
}

const STOP_BELOW = 0.12;

/** The closest of the nine movement choices to a world-space movement vector. */
export function moveChoice(movement: { x: number; y: number }, attack: 1 | -1): number {
  const local = toPolicy(movement, attack);
  if (Math.hypot(local.x, local.y) < STOP_BELOW) return 0;
  const angle = (Math.atan2(local.y, local.x) + 2 * Math.PI) % (2 * Math.PI);
  return 1 + (Math.round(angle / (Math.PI / 4)) % 8);
}

/**
 * The teammate slot a kick is best aimed at. A receiver the controller named wins when its act is legal. Otherwise
 * (a kickoff toward open space, a clearance) take the legal teammate whose direction is closest to the aim.
 */
function targetSlot(
  intent: PlayerIntent, context: AgentContext, aim: { x: number; y: number }, legal: (slot: number) => boolean,
): number {
  const named = intent.pass ? context.team.findIndex(({ playerId }) => playerId === intent.pass!.receiverId) : -1;
  if (named >= 0 && legal(named)) return named;
  let best = -1;
  let bestCosine = -Infinity;
  const { ego, observation } = context;
  context.team.forEach((mate, slot) => {
    if (mate.playerId === ego.playerId || !legal(slot)) return;
    const dx = mate.position.x - observation.ball.position.x;
    const dy = mate.position.y - observation.ball.position.y;
    const range = Math.hypot(dx, dy);
    if (range < 1e-6) return;
    const cosine = (dx * aim.x + dy * aim.y) / (range * Math.hypot(aim.x, aim.y));
    if (cosine > bestCosine) { bestCosine = cosine; best = slot; }
  });
  return best;
}

/**
 * Turns a controller's decision into the nearest [move, sprint, act] choice, so a trained policy can imitate it.
 * Choices that the mask forbids, or that no act expresses, become "none" and set `lost`.
 */
export function intentToChoice(
  intent: PlayerIntent, context: AgentContext, space: ActionSpace, mask: ArrayLike<number>,
): Choice {
  const move = moveChoice(intent.movement, context.attack);
  const sprint = intent.sprint ? 1 : 0;
  const { request } = intent;
  let act: number = ACT_NONE;
  if (request) {
    switch (request.kind) {
      case "tackle": act = ACT_TACKLE; break;
      case "keeperSave": act = ACT_SAVE; break;
      case "kick":
        if (request.kickType === "shot") {
          // Which corner of the goal the aim line crosses, in the attack frame, where left is +y.
          const local = toPolicy(request.direction, context.attack);
          const origin = toPolicy(context.observation.ball.position, context.attack);
          const goalX = context.observation.pitch.lengthM / 2;
          const crossing = local.x > 1e-6 ? origin.y + ((goalX - origin.x) / local.x) * local.y : 0;
          const side = Math.abs(crossing) < context.observation.pitch.goalWidthM * 0.15 ? 0 : crossing > 0 ? 1 : -1;
          act = side === 0 ? ACT_SHOOT_CENTER : side > 0 ? ACT_SHOOT_LEFT : ACT_SHOOT_RIGHT;
          break;
        }
      // Passes and lofted passes fall through to a target slot.
      case "throwIn":
      case "keeperRelease": {
        const loft = request.kind === "kick" && request.kickType === "loft";
        let base = loft ? space.loftBase : space.passBase;
        const slot = targetSlot(intent, context, request.direction, (candidate) => mask[base + candidate] === 1);
        if (slot >= 0 && request.kind === "kick" && request.kickType === "pass" && mask[space.softBase + slot] === 1) {
          // A ground pass that would arrive nearer the soft pace than the firm one is the soft act.
          const { observation, ego } = context;
          const mate = context.team[slot]!;
          const range = Math.hypot(mate.position.x - observation.ball.position.x, mate.position.y - observation.ball.position.y);
          const speed = request.power * ego.definition.abilities.physical.maxKickSpeedMps * 0.55;
          if (rollSpeedAfter(speed, range, ballOf(observation)) < (FIRM_PASS_ARRIVAL_MPS + SOFT_PASS_ARRIVAL_MPS) / 2) {
            base = space.softBase;
          }
        }
        if (slot >= 0) act = base + slot;
        break;
      }
    }
  }
  const lost = request !== undefined && (act === ACT_NONE || mask[act] !== 1);
  return { move, sprint, act: lost ? ACT_NONE : act, lost };
}
