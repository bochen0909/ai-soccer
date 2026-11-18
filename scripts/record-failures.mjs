import { readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import {
  AI_PROFILES, createArcadeConfig, createDefaultAIRegistry, decodeWeights, MatchSession, mirrorScenario,
  profileTeamOptions, registerNeuralNetwork,
} from "../dist/index.js";
import { withKeeperReach } from "./keeper-reach.mjs";

const usage = `Usage: node scripts/record-failures.mjs --model PATH [--opponents default,tuned,model:PATH] [--matches N] [--lead TICKS] [--out PATH]

Plays the model against opponents and, for every goal it concedes, saves the state --lead ticks (default 300, 5 s) before
the goal as a scenario with the model as the home team. Training can start episodes from these (env option replayFile).`;
const { values } = parseArgs({ options: { help: { type: "boolean" }, model: { type: "string" }, opponents: { type: "string" },
  matches: { type: "string" }, lead: { type: "string" }, out: { type: "string" }, seed: { type: "string" } } });
if (values.help || !values.model) { console.log(usage); process.exit(values.help ? 0 : 2); }
const matches = Number(values.matches ?? 60);
const lead = Number(values.lead ?? 300);
const opponents = (values.opponents ?? "tuned,default").split(",");
const out = values.out ?? "demos/failures.json";
const baseSeed = Number(values.seed ?? 7_000_000);
const config = withKeeperReach(createArcadeConfig());
const network = decodeWeights(JSON.parse(readFileSync(values.model, "utf8")));

const scenarios = [];
const STRIDE = 20;
let conceded = 0;
for (let match = 0; match < matches; match += 1) {
  const learnerIndex = match % 2;
  const spec = opponents[match % opponents.length];
  const registry = createDefaultAIRegistry();
  registerNeuralNetwork(registry, network, "learner");
  const rival = spec.startsWith("model:")
    ? (registerNeuralNetwork(registry, decodeWeights(JSON.parse(readFileSync(spec.slice(6), "utf8"))), "rival"), { pluginId: "rival" })
    : profileTeamOptions(spec);
  const learnerId = config.teams[learnerIndex].id;
  const session = new MatchSession(registry);
  session.reset(config, baseSeed + match, { teams: { [learnerId]: { pluginId: "learner" }, [config.teams[1 - learnerIndex].id]: rival } });
  const ring = [];
  let result;
  do {
    result = session.step();
    const snap = session.snapshot();
    if (snap.tick % STRIDE === 0 && snap.phase.kind === "live") ring.push(structuredClone(snap));
    if (ring.length > Math.ceil(lead / STRIDE) + 2) ring.shift();
    for (const event of result.events) {
      if (event.kind !== "goal" || event.teamId === learnerId) continue;
      conceded += 1;
      const before = ring.find((s) => event.tick - s.tick <= lead);
      if (!before) continue;
      const scenario = {
        version: 1, id: `fail-${match}-${event.tick}`, period: before.period, remainingPeriodTicks: before.remainingPeriodTicks,
        score: before.score,
        players: before.players.map(({ playerId, position, facingRad, velocity, stamina, balance }) =>
          ({ playerId, position, facingRad, velocity, stamina, balance })),
        ball: { position: before.ball.position, velocity: before.ball.velocity,
          lastTouchPlayerId: before.ball.lastTouchPlayerId ?? before.players[0].playerId, control: before.ball.control },
        phase: { kind: "live" },
      };
      scenarios.push(learnerIndex === 0 ? scenario : mirrorScenario(scenario));
    }
  } while (result.status !== "finished");
  session.dispose();
}
writeFileSync(out, JSON.stringify(scenarios));
console.log(`${matches} matches, ${conceded} goals conceded, ${scenarios.length} situations saved to ${out}`);
