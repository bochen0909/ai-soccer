import type { PlayerState, Vec2 } from "../contracts/index.js";
import type { EngineState } from "./reset.js";

export function executionFactors(state: EngineState, player: PlayerState) {
  const speed = Math.hypot(player.velocity.x, player.velocity.y);
  const fatigue = 0.65 + 0.35 * player.stamina;
  const balance = 0.7 + 0.3 * player.balance;
  const movement = 1 - 0.2 * Math.min(1, speed / player.definition.abilities.physical.maxSpeedMps);
  let pressure = 1;
  for (const other of state.players) {
    if (other.teamId === player.teamId) continue;
    const distance = Math.hypot(other.position.x - player.position.x, other.position.y - player.position.y);
    pressure = Math.min(pressure, 0.75 + 0.25 * Math.min(1, distance / 2.5));
  }
  return { exertion: fatigue * balance, fatigue, balance, movement, pressure };
}

export function footQuality(player: PlayerState, ballPosition: Vec2): number {
  const lateral = Math.cos(player.facingRad) * (ballPosition.y - player.position.y) -
    Math.sin(player.facingRad) * (ballPosition.x - player.position.x);
  const weakSide = player.definition.preferredFoot === "right" ? lateral > 0 : lateral < 0;
  if (!weakSide) return 1;
  const weight = Math.min(1, Math.abs(lateral) / player.definition.radiusM);
  return 1 - 0.35 * weight * (1 - player.definition.abilities.technical.weakFoot);
}
