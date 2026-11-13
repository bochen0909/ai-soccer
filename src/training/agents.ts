import type { MatchConfig, PlayerDefinition, TeamObservation } from "../contracts/index.js";
import type { AgentContext } from "./actions.js";

/** One context per player on the learner team, with both rosters in the given order. */
export function buildRosterContexts(
  observation: TeamObservation, learnerIndex: 0 | 1,
  teamRoster: readonly PlayerDefinition[], opponentRoster: readonly PlayerDefinition[],
): AgentContext[] {
  const attack = observation.attackDirections[learnerIndex];
  const byId = new Map(observation.players.map((player) => [player.playerId, player]));
  const ordered = (roster: readonly PlayerDefinition[]) => roster.map(({ id }) => byId.get(id)!);
  const team = ordered(teamRoster);
  const opponents = ordered(opponentRoster);
  return team.map((ego) => ({ observation, attack, team, opponents, ego }));
}

/** One context per player on the learner team, with both rosters in their configured order. */
export function buildAgentContexts(
  observation: TeamObservation, config: MatchConfig, learnerIndex: 0 | 1,
): AgentContext[] {
  return buildRosterContexts(observation, learnerIndex, config.teams[learnerIndex].players,
    config.teams[1 - learnerIndex].players);
}
