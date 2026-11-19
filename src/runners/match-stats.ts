import type { ActionKey, MatchConfig, MatchEvent, PlayerId, TeamId } from "../contracts/index.js";
import type { SessionSnapshot, SessionTickResult } from "../session/contracts.js";

/** Who controls a team, as shown to the player, plus the trained model's name when a network plays. */
export interface TeamLabel {
  name: string;
  control: string;
  model?: string;
}

export interface TeamStats {
  goals: number;
  shots: number;
  shotsOnTarget: number;
  passes: number;
  passesCompleted: number;
  /** Share of completed passes, 0 to 1, or null with no passes. */
  passAccuracy: number | null;
  /** Share of live play in which the team touched the ball last, 0 to 1 (both teams add up to 1). */
  possession: number;
  tackles: number;
  tacklesWon: number;
  saves: number;
  corners: number;
  throwIns: number;
  goalKicks: number;
  /** Metres run by the whole team during live play. */
  distanceM: number;
}

export interface GoalRecord {
  /** Match clock when the goal went in, "mm:ss". */
  time: string;
  period: 1 | 2;
  teamId: TeamId;
  scorerId: PlayerId | null;
  ownGoal: boolean;
}

export interface MatchReport {
  version: 1;
  recordedAt: string;
  seed: number;
  format: string;
  playersPerSide: number;
  durationSeconds: number;
  score: [number, number];
  teams: (TeamLabel & { id: TeamId; stats: TeamStats })[];
  goals: GoalRecord[];
}

type RequestKind = "shot" | "pass" | "loft" | "tackle" | "keeperSave" | "other";

const keyOf = (action: ActionKey): string => `${action.source}:${action.playerId}:${action.requestId}`;

function emptyStats(): Omit<TeamStats, "passAccuracy" | "possession"> & { liveTicks: number } {
  return {
    goals: 0, shots: 0, shotsOnTarget: 0, passes: 0, passesCompleted: 0, tackles: 0, tacklesWon: 0, saves: 0,
    corners: 0, throwIns: 0, goalKicks: 0, distanceM: 0, liveTicks: 0,
  };
}

/** Collects a match's team statistics from each session step, for the full-time summary and its JSON copy. */
export class MatchStatsRecorder {
  private readonly teamOf = new Map<PlayerId, TeamId>();
  private readonly kinds = new Map<string, RequestKind>();
  private readonly totals = new Map<TeamId, ReturnType<typeof emptyStats>>();
  private readonly goals: GoalRecord[] = [];
  private lastTick = 0;

  constructor(
    private readonly config: MatchConfig,
    private readonly seed: number,
    private readonly labels: readonly [TeamLabel, TeamLabel],
    private readonly format: string,
  ) {
    for (const team of config.teams) {
      this.totals.set(team.id, emptyStats());
      for (const player of team.players) this.teamOf.set(player.id, team.id);
    }
  }

  /** Call after every session step with its result and the snapshot taken right after it. */
  record(result: SessionTickResult, snapshot: SessionSnapshot): void {
    this.lastTick = snapshot.tick;
    for (const action of result.frame.actions) {
      const request = action.request;
      if (!request) continue;
      const kind: RequestKind = request.kind === "kick" ? request.kickType
        : request.kind === "tackle" || request.kind === "keeperSave" ? request.kind : "other";
      this.kinds.set(keyOf({ source: "controller", playerId: action.playerId, requestId: request.id }), kind);
    }
    for (const event of result.events) this.event(event, snapshot);

    if (snapshot.phase.kind !== "live") return;
    const holder = snapshot.ball.lastTouchPlayerId;
    const holderTeam = holder === null ? undefined : this.teamOf.get(holder);
    if (holderTeam !== undefined) this.totals.get(holderTeam)!.liveTicks += 1;
    for (const player of snapshot.players) {
      this.totals.get(player.teamId)!.distanceM += Math.hypot(player.velocity.x, player.velocity.y) / 60;
    }
  }

  private kind(action: ActionKey | undefined): RequestKind | undefined {
    if (action === undefined) return undefined;
    // Rule-generated actions are restart passes and automatic keeper rolls.
    return action.source === "rule" ? "pass" : this.kinds.get(keyOf(action));
  }

  private event(event: MatchEvent, snapshot: SessionSnapshot): void {
    switch (event.kind) {
      case "goal": {
        const scorerTeam = event.lastTouchPlayerId === null ? undefined : this.teamOf.get(event.lastTouchPlayerId);
        const ownGoal = scorerTeam !== undefined && scorerTeam !== event.teamId;
        const stats = this.totals.get(event.teamId)!;
        stats.goals += 1;
        if (!ownGoal) stats.shotsOnTarget += 1;
        this.goals.push({ time: this.clock(snapshot), period: snapshot.period, teamId: event.teamId,
          scorerId: event.lastTouchPlayerId, ownGoal });
        break;
      }
      case "ballExit": {
        const stats = this.totals.get(event.teamId)!;
        if (event.restart === "corner") stats.corners += 1;
        else if (event.restart === "throwIn") stats.throwIns += 1;
        else stats.goalKicks += 1;
        break;
      }
      case "actionContact": {
        const team = this.teamOf.get(event.action.playerId);
        if (team === undefined) break;
        const stats = this.totals.get(team)!;
        const kind = this.kind(event.action);
        if (event.outcome === "kicked") {
          if (kind === "shot") stats.shots += 1;
          else if (kind === "pass" || kind === "loft") stats.passes += 1;
        } else if (event.outcome === "wonBall" || event.outcome === "poked") {
          stats.tackles += 1;
          stats.tacklesWon += 1;
        } else if (event.outcome === "missed" && kind === "tackle") {
          stats.tackles += 1;
        } else if (event.outcome === "caught" || event.outcome === "parried") {
          stats.saves += 1;
          const shooter = event.flightAction && this.teamOf.get(event.flightAction.playerId);
          if (shooter !== undefined && shooter !== team && this.kind(event.flightAction) === "shot") {
            this.totals.get(shooter)!.shotsOnTarget += 1;
          }
        }
        break;
      }
      case "ballTouch": {
        if ((event.outcome !== "received" && event.outcome !== "headed") || !event.flightAction) break;
        const passer = event.flightAction.playerId;
        const kind = this.kind(event.flightAction);
        const team = this.teamOf.get(passer);
        if (passer !== event.playerId && team !== undefined && team === this.teamOf.get(event.playerId) &&
            (kind === "pass" || kind === "loft")) {
          this.totals.get(team)!.passesCompleted += 1;
        }
        break;
      }
    }
  }

  private clock(snapshot: SessionSnapshot): string {
    const elapsedTicks = (snapshot.period === 2 ? this.config.periodTicks[0] : 0) +
      this.config.periodTicks[snapshot.period - 1] - snapshot.remainingPeriodTicks;
    const seconds = Math.floor(elapsedTicks / 60);
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }

  report(score: readonly [number, number]): MatchReport {
    const touched = [...this.totals.values()].reduce((sum, totals) => sum + totals.liveTicks, 0);
    return {
      version: 1,
      recordedAt: new Date().toISOString(),
      seed: this.seed,
      format: this.format,
      playersPerSide: this.config.teams[0].players.length,
      durationSeconds: Math.round(this.lastTick / 60),
      score: [score[0], score[1]],
      teams: this.config.teams.map((team, index) => {
        const { liveTicks, ...totals } = this.totals.get(team.id)!;
        return {
          id: team.id, ...this.labels[index],
          stats: {
            ...totals,
            distanceM: Math.round(totals.distanceM),
            passAccuracy: totals.passes > 0 ? +(totals.passesCompleted / totals.passes).toFixed(3) : null,
            possession: touched > 0 ? +(liveTicks / touched).toFixed(3) : 0,
          },
        };
      }),
      goals: [...this.goals],
    };
  }
}
