import type { MatchConfig, PlayerDefinition, TeamDefinition } from "../contracts/index.js";
import { createPlayerAbilities } from "./players.js";

function team(id: string): TeamDefinition {
  const positions = [
    { x: -27, y: 0 },
    { x: -18, y: -9 },
    { x: -18, y: 9 },
    { x: -10, y: 0 },
    { x: -5, y: 4 },
  ];
  const players = positions.map((formation, index): PlayerDefinition => {
    const keeper = index === 0;
    const abilities = createPlayerAbilities();
    if (keeper) abilities.technical.goalkeeping = 0.72;
    return {
      id: keeper ? `${id}-keeper` : `${id}-${index}`,
      role: keeper ? "goalkeeper" : "outfield",
      formation,
      bodyHeightM: 1.78,
      radiusM: 0.32,
      massKg: 72,
      preferredFoot: "right",
      abilities,
    };
  });
  return { id, players };
}

/** Returns an independent, editable configuration for a short five-a-side match. */
export function createArcadeConfig(): MatchConfig {
  return {
    pitch: {
      lengthM: 60,
      widthM: 40,
      goalWidthM: 5,
      goalHeightM: 2.1,
      postRadiusM: 0.05,
      goalAreaDepthM: 4,
      goalAreaWidthM: 10,
      keeperAreaDepthM: 8,
      keeperAreaWidthM: 16,
      centerCircleRadiusM: 3,
    },
    ball: {
      radiusM: 0.11,
      massKg: 0.43,
      gravityMps2: 9.81,
      groundDecelerationMps2: 1.6,
      groundRestitution: 0.55,
      contactRestitution: 0.65,
      maxSpeedMps: 45,
    },
    teams: [team("home"), team("away")],
    periodTicks: [3600, 3600],
    firstKickoffTeamId: "home",
    rules: {
      version: 1,
      restartPlacementTicks: 60,
      restartInputTicks: 300,
      goalPauseTicks: 120,
      halftimeTicks: 180,
      keeperHoldTicks: 360,
      restartExclusionM: 3,
    },
    tuning: { version: 2 },
  };
}
