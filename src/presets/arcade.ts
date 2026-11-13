import type { MatchConfig, PlayerDefinition, TeamDefinition, Vec2 } from "../contracts/index.js";
import { createPlayerAbilities } from "./players.js";

const quickFormation = [
  { x: -27, y: 0 },
  { x: -18, y: -9 },
  { x: -18, y: 9 },
  { x: -10, y: 0 },
  { x: -5, y: 4 },
];

// 4–3–3, in metres from midfield, for a team attacking toward positive x.
const fullSizeFormation = [
  { x: -49, y: 0 },
  { x: -32, y: -23 }, { x: -37, y: -8 }, { x: -37, y: 8 }, { x: -32, y: 23 },
  { x: -18, y: -13 }, { x: -13, y: 0 }, { x: -18, y: 13 },
  { x: -6, y: -21 }, { x: -1, y: 0 }, { x: -6, y: 21 },
];

function team(id: string, positions: readonly Vec2[]): TeamDefinition {
  const players = positions.map((formation, index): PlayerDefinition => {
    const keeper = index === 0;
    const abilities = createPlayerAbilities();
    if (keeper) abilities.technical.goalkeeping = 0.72;
    return {
      id: keeper ? `${id}-keeper` : `${id}-${index}`,
      role: keeper ? "goalkeeper" : "outfield",
      formation: { ...formation },
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
    teams: [team("home", quickFormation), team("away", quickFormation)],
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

/** Returns an editable eleven-a-side match with three-minute halves. */
export function createElevenAsideConfig(): MatchConfig {
  const config = createArcadeConfig();
  return {
    ...config,
    pitch: {
      ...config.pitch,
      lengthM: 105,
      widthM: 68,
      goalWidthM: 7.32,
      goalHeightM: 2.44,
      goalAreaDepthM: 5.5,
      goalAreaWidthM: 18.32,
      keeperAreaDepthM: 16.5,
      keeperAreaWidthM: 40.32,
      centerCircleRadiusM: 9.15,
    },
    teams: [team("home", fullSizeFormation), team("away", fullSizeFormation)],
    periodTicks: [10800, 10800],
    rules: { ...config.rules, restartExclusionM: 9.15, directionalKickoff: true },
  };
}
