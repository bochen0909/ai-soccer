// Where does a policy win and lose its points? Plays it against one opponent and reports, from the match events, how the
// goals against come about (shot distance, how the ball was won, how long before the goal), shots on target both ways,
// saves, and giveaways in our own third.
//   node scripts/match-report.mjs MODEL.json OPPONENT --matches 64 [--seed 1] [--out FILE]
// OPPONENT is "default", "tuned" or a model JSON path. Prints one JSON line per worker run; combine with --merge.
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { createArcadeConfig, createDefaultAIRegistry, MatchSession, profileTeamOptions, registerNeuralNetwork, decodeWeights } from "../dist/index.js";
import { SoccerEngine } from "../dist/engine/engine.js";

const { positionals, values } = parseArgs({ allowPositionals: true, options: { matches: { type: "string", default: "16" }, start: { type: "string", default: "0" }, seed: { type: "string", default: "1" } } });
const [modelPath, opponentSpec] = positionals;
const network = decodeWeights(JSON.parse(readFileSync(resolve(modelPath), "utf8")));
const opponentNetwork = opponentSpec.endsWith(".json") ? decodeWeights(JSON.parse(readFileSync(resolve(opponentSpec), "utf8"))) : undefined;
const config = createArcadeConfig();
const half = config.pitch.lengthM / 2;
const out = { matches: 0, goalsFor: 0, goalsAgainst: 0, points: 0, gaShotDepth: [], gaSinceGain: [], gaGainDepth: [], gaGainKind: {}, gaFromRestartWithin8s: 0,
  sotFor: 0, sotAgainst: 0, saves: 0, giveawaysOwnThird: 0, winsOppThird: 0, pickups: 0, losses: 0 };

for (let n = 0; n < Number(values.matches); n += 1) {
  const index = Number(values.start) + n, seed = 900_000_000 + Number(values.seed) * 100_000 + index;
  const ours = index % 2 === 0 ? "home" : "away", theirs = ours === "home" ? "away" : "home", ourIndex = ours === "home" ? 0 : 1;
  const registry = createDefaultAIRegistry();
  registerNeuralNetwork(registry, network, "us");
  if (opponentNetwork) registerNeuralNetwork(registry, opponentNetwork, "them");
  const engine = new SoccerEngine();
  const session = new MatchSession(registry, engine);
  session.reset(config, seed, { teams: { [ours]: { pluginId: "us" }, [theirs]: opponentNetwork ? { pluginId: "them" } : profileTeamOptions(opponentSpec) } });
  const ownDepth = (x) => { const attack = engine.observe(ours).attackDirections[ourIndex]; return x * attack + half; };   // metres from our own goal line
  let lastTouchTeam = null, gain = null, lastKick = null, restartTick = -1e9, tick = 0, finished = false;
  const shotOnTarget = (team, ballAfter) => {
    const attack = engine.observe(team).attackDirections[team === "home" ? 0 : 1];
    const speed = Math.hypot(ballAfter.velocity.x, ballAfter.velocity.y, ballAfter.velocity.z);
    if (speed < 6 || ballAfter.velocity.x * attack < 0.3 * speed) return false;
    const goalX = attack * half, t = (goalX - ballAfter.position.x) / ballAfter.velocity.x;
    return Math.abs(ballAfter.position.y + ballAfter.velocity.y * t) <= config.pitch.goalWidthM / 2 && Math.abs(goalX - ballAfter.position.x) <= 25;
  };
  while (!finished) {
    const result = session.step({});
    tick = result.tick; finished = result.status === "finished";
    const ball = engine.observe(ours).ball;
    for (const e of result.events) {
      if (e.kind === "phaseChanged" && e.phase.kind !== "live") restartTick = tick;
      if (e.kind === "ballExit") restartTick = tick;
      const playerId = e.kind === "actionContact" ? e.action.playerId : e.kind === "ballTouch" ? e.playerId : undefined;
      const team = playerId?.startsWith("home") ? "home" : playerId?.startsWith("away") ? "away" : undefined;
      if (e.kind === "actionContact" && e.outcome === "kicked") {
        lastKick = { team, depth: team === theirs ? ownDepth(e.position.x) : null, tick };
        if (shotOnTarget(team, ball)) team === ours ? out.sotFor += 1 : out.sotAgainst += 1;
      }
      if (e.kind === "actionContact" && (e.outcome === "caught" || e.outcome === "parried") && team === ours) out.saves += 1;
      if (team && (e.kind === "ballTouch" || (e.kind === "actionContact" && e.outcome !== "missed"))) {
        if (lastTouchTeam && lastTouchTeam !== team) {
          const kind = e.kind === "actionContact" && e.outcome === "wonBall" ? "tackle" : e.kind === "actionContact" ? e.outcome : "touch";
          gain = { team, tick, depth: ownDepth(e.position.x), kind };
          if (team === theirs && gain.depth < 20) out.giveawaysOwnThird += 1;
          if (team === ours && gain.depth > half * 2 - 20) out.winsOppThird += 1;
          team === theirs ? (out.pickups += 0, out.losses += 1) : (out.pickups += 1);
        }
        lastTouchTeam = team;
      }
      if (e.kind === "goal") {
        if (e.teamId === ours) out.goalsFor += 1;
        else {
          out.goalsAgainst += 1;
          if (lastKick?.team === theirs && lastKick.depth !== null) out.gaShotDepth.push(+lastKick.depth.toFixed(1));
          if (gain?.team === theirs) { out.gaSinceGain.push(+((tick - gain.tick) / 60).toFixed(1)); out.gaGainDepth.push(+gain.depth.toFixed(1)); out.gaGainKind[gain.kind] = (out.gaGainKind[gain.kind] ?? 0) + 1; }
          if (tick - restartTick < 480) out.gaFromRestartWithin8s += 1;
        }
        lastTouchTeam = null; gain = null; lastKick = null;
      }
    }
  }
  const score = engine.observe(ours).score;
  const [gf, ga] = ourIndex === 0 ? score : [score[1], score[0]];
  out.matches += 1; out.points += gf > ga ? 3 : gf === ga ? 1 : 0;
}
console.log(JSON.stringify(out));
