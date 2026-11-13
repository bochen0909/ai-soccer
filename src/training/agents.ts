import type { MatchConfig, TeamObservation } from "../contracts/index.js";
import type { AgentContext } from "./actions.js";

/** One context per player on the learner team, with both rosters in their configured order. */
export function buildAgentContexts(
  observation: TeamObservation, config: MatchConfig, learnerIndex: 0 | 1,
): AgentContext[] {
  const attack = observation.attackDirections[learnerIndex];
  const byId = new Map(observation.players.map((player) => [player.playerId, player]));
  const ordered = (index: 0 | 1) => config.teams[index].players.map(({ id }) => byId.get(id)!);
  const team = ordered(learnerIndex);
  const opponents = ordered((1 - learnerIndex) as 0 | 1);
  return team.map((ego) => ({ observation, attack, team, opponents, ego }));
}
