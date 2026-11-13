import type { PitchConfig, PlayerId, Vec2 } from "../contracts/index.js";
import type { ActionIntent, PassTarget, PlayerIntent } from "./contracts.js";

function record(value: unknown, label: string): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new TypeError(`${label} must be an object`);
  }
  return value as Record<string, unknown>;
}

function finite(value: unknown, label: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new TypeError(`${label} must be finite`);
  }
  return value;
}

function vector(value: unknown, label: string): Vec2 {
  const { x, y } = record(value, label);
  return { x: finite(x, label), y: finite(y, label) };
}

function validateRequest(value: unknown): ActionIntent {
  const request = record(value, "Action request");
  const kind = request.kind;
  const direction = vector(request.direction, "Action direction");
  if (direction.x === 0 && direction.y === 0) throw new RangeError("Action direction must be nonzero");

  switch (kind) {
    case "kick": {
      const { kickType } = request;
      if (kickType !== "pass" && kickType !== "shot" && kickType !== "loft") {
        throw new TypeError("Invalid kick technique");
      }
      return {
        kind, kickType, direction,
        power: finite(request.power, "Kick power"), loft: finite(request.loft, "Kick loft"),
      };
    }
    case "throwIn":
      return { kind, direction, power: finite(request.power, "Throw-in power") };
    case "keeperRelease": {
      const { technique } = request;
      if (technique !== "roll" && technique !== "throw") throw new TypeError("Invalid release technique");
      return { kind, technique, direction, power: finite(request.power, "Release power") };
    }
    case "tackle":
    case "keeperSave":
      return { kind, direction };
    default:
      throw new TypeError("Invalid action kind");
  }
}

export function isPassRequest(request: ActionIntent | undefined): boolean {
  return request?.kind === "kick" && (request.kickType === "pass" || request.kickType === "loft");
}

function validatePass(
  value: unknown,
  passer: PlayerId,
  teamIds: ReadonlySet<PlayerId>,
  pitch: PitchConfig,
): PassTarget {
  const { receiverId, target } = record(value, "Pass target");
  if (typeof receiverId !== "string" || receiverId === passer || !teamIds.has(receiverId)) {
    throw new RangeError("Pass receiver must be a different active teammate");
  }
  const position = vector(target, "Pass target position");
  if (Math.abs(position.x) > pitch.lengthM / 2 || Math.abs(position.y) > pitch.widthM / 2) {
    throw new RangeError("Pass target must be inside the field");
  }
  return { receiverId, target: position };
}

/** Validates a complete decision and copies its supported fields before it can be applied. */
export function validateIntents(
  value: unknown,
  assigned: ReadonlySet<PlayerId>,
  teamIds: ReadonlySet<PlayerId>,
  pitch: PitchConfig,
): PlayerIntent[] {
  if (!Array.isArray(value)) throw new TypeError("Team AI decision must be an array of intents");

  const seen = new Set<PlayerId>();
  const result: PlayerIntent[] = [];
  for (const item of value as unknown[]) {
    const { playerId, movement, sprint, request, pass } = record(item, "Player intent");
    if (typeof playerId !== "string" || !assigned.has(playerId)) {
      throw new RangeError("Player intent must target a player assigned to this controller");
    }
    if (seen.has(playerId)) throw new RangeError(`Duplicate intent for player "${playerId}"`);
    seen.add(playerId);
    if (typeof sprint !== "boolean") throw new TypeError("Sprint intent must be a boolean");

    const intent: PlayerIntent = { playerId, movement: vector(movement, "Movement"), sprint };
    if (request !== undefined) intent.request = validateRequest(request);
    if (pass !== undefined) {
      if (!isPassRequest(intent.request)) throw new TypeError("Pass target requires a pass or loft request");
      intent.pass = validatePass(pass, playerId, teamIds, pitch);
    }
    result.push(intent);
  }
  return result;
}
