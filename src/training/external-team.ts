import type { PlayerId, Vec2 } from "../contracts/index.js";
import type { ActionIntent, AIContext, PlayerIntent, TeamAI } from "../controllers/contracts.js";

const neutral: Vec2 = { x: 0, y: 0 };

/**
 * Team plugin whose decisions come from the training environment.
 * Movement persists across calls. Requests are delivered once, so a decision the session forces
 * in the middle of a policy step cannot repeat a kick.
 */
export class ExternalTeamAI implements TeamAI {
  readonly id = "external";
  readonly version = "1";
  private held = new Map<PlayerId, { movement: Vec2; sprint: boolean }>();
  private requests = new Map<PlayerId, ActionIntent>();
  private players: readonly PlayerId[] = [];

  reset(context: AIContext): void {
    this.players = context.players.map(({ id }) => id);
    this.dispose();
  }

  dispose(): void {
    this.held.clear();
    this.requests.clear();
  }

  command(playerId: PlayerId, movement: Vec2, sprint: boolean, request?: ActionIntent): void {
    this.held.set(playerId, { movement, sprint });
    if (request) this.requests.set(playerId, request);
    else this.requests.delete(playerId);
  }

  decide(): readonly PlayerIntent[] {
    const intents = this.players.map((playerId): PlayerIntent => {
      const held = this.held.get(playerId);
      const request = this.requests.get(playerId);
      return { playerId, movement: held?.movement ?? neutral, sprint: held?.sprint ?? false, request };
    });
    this.requests.clear();
    return intents;
  }
}
