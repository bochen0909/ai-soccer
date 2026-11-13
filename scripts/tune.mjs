import { availableParallelism } from "node:os";
import { readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";
import { Worker } from "node:worker_threads";

const usage = `Usage:
  npm run --silent tune -- variance [--matches N] [--seed N] [--size 5|11]
  npm run --silent tune -- search [--generations N] [--population N] [--matches N] [--seed N]
                                  [--size 5|11] [--opponent PATH] [--out PATH]
  npm run --silent tune -- compare --params PATH [--matches N] [--seed N] [--size 5|11] [--opponent PATH]

variance  Plays the default baseline against itself on both sides to measure noise and home bias.
search    Evolution strategy over baseline traits and weights. Fitness is goal difference against the
          opponent, alternating sides. Writes PATH (full history) and PATH.teams.json for the evaluate CLI.
compare   Plays saved parameters against the opponent on held-out seeds and reports a confidence interval.

Options:
  --matches N      Matches per candidate (search, default 24) or in total (variance, compare; default 200).
  --generations N  Search generations (default 20).     --population N  Candidates per generation (default 16).
  --seed N         Random seed for sampling (default 1). --workers N  Worker threads (default CPUs - 1).
  --opponent PATH  Parameters file for the opponent (default: baseline defaults).
`;

const options = {
  help: { type: "boolean" }, matches: { type: "string" }, generations: { type: "string" },
  population: { type: "string" }, seed: { type: "string" }, size: { type: "string" }, out: { type: "string" },
  params: { type: "string" }, opponent: { type: "string" }, workers: { type: "string" },
};

// Search space. Traits are linear in [0, 1]. Weights are multipliers searched on a log scale.
const TRAITS = ["aggression", "riskTolerance", "awareness"];
const WEIGHTS = ["passLane", "passSpace", "passProgress", "passRange", "shotRange", "dribbleAhead", "pressureRadius"];
const NAMES = [...TRAITS, ...WEIGHTS];
const WEIGHT_MIN = 0.25;
const WEIGHT_MAX = 3;
const DEFAULT_UNIT = [0.55, 0.5, 0.65, ...WEIGHTS.map(() => Math.log(1 / WEIGHT_MIN) / Math.log(WEIGHT_MAX / WEIGHT_MIN))];

function toConfig(unit) {
  const config = {};
  NAMES.forEach((name, index) => {
    const u = Math.min(1, Math.max(0, unit[index]));
    config[name] = index < TRAITS.length ? u : WEIGHT_MIN * (WEIGHT_MAX / WEIGHT_MIN) ** u;
  });
  return config;
}

function toUnit(config) {
  return NAMES.map((name, index) => index < TRAITS.length ? config[name]
    : Math.log(config[name] / WEIGHT_MIN) / Math.log(WEIGHT_MAX / WEIGHT_MIN));
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;
const stdev = (values) => {
  const m = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - m) ** 2, 0) / Math.max(1, values.length - 1));
};

class Pool {
  constructor(size) {
    this.workers = Array.from({ length: size }, () => new Worker(new URL("./tune-worker.mjs", import.meta.url)));
  }

  /** Runs every task and returns results in task order. */
  run(tasks) {
    return new Promise((resolve, reject) => {
      const results = new Array(tasks.length);
      let next = 0;
      let pending = tasks.length;
      if (pending === 0) { resolve(results); return; }
      const feed = (worker) => {
        if (next >= tasks.length) return;
        const id = next++;
        worker.postMessage({ id, ...tasks[id] });
      };
      for (const worker of this.workers) {
        worker.removeAllListeners("message");
        worker.removeAllListeners("error");
        worker.on("error", reject);
        worker.on("message", (message) => {
          if (message.error) { reject(new Error(message.error)); return; }
          results[message.id] = message;
          pending -= 1;
          if (pending === 0) resolve(results); else feed(worker);
        });
        feed(worker);
      }
    });
  }

  close() {
    return Promise.all(this.workers.map((worker) => worker.terminate()));
  }
}

function integer(value, name, fallback, min = 1) {
  if (value === undefined) return fallback;
  const number = Number(value);
  if (!/^\d+$/.test(value) || number < min) throw new RangeError(`${name} must be an integer of at least ${min}`);
  return number;
}

async function readParams(path) {
  const data = JSON.parse(await readFile(path, "utf8"));
  const config = data.config ?? data.best?.config ?? data;
  for (const name of NAMES) {
    if (typeof config[name] !== "number") throw new Error(`${path} is missing "${name}"`);
  }
  return Object.fromEntries(NAMES.map((name) => [name, config[name]]));
}

/** Matches alternate sides so home advantage cancels. Candidate plays home on even match indices. */
function matchTasks(candidate, opponent, seeds, size) {
  return seeds.map((seed, index) => ({ seed, size, side: index % 2 === 0 ? "home" : "away", candidate, opponent }));
}

function summarize(results) {
  const diffs = results.map(({ goalsFor, goalsAgainst }) => goalsFor - goalsAgainst);
  return {
    matches: diffs.length,
    meanDiff: mean(diffs),
    standardError: stdev(diffs) / Math.sqrt(diffs.length),
    wins: diffs.filter((d) => d > 0).length,
    draws: diffs.filter((d) => d === 0).length,
    losses: diffs.filter((d) => d < 0).length,
    goalsFor: mean(results.map((r) => r.goalsFor)),
    goalsAgainst: mean(results.map((r) => r.goalsAgainst)),
    unfinished: results.filter((r) => !r.finished && !r.crashed).length,
    crashed: results.filter((r) => r.crashed).length,
  };
}

const f = (value, digits = 3) => value.toFixed(digits);

async function variance(pool, values, size) {
  const matches = integer(values.matches, "--matches", 200, 4);
  const base = integer(values.seed, "--seed", 1, 0) * 100_000;
  const seeds = Array.from({ length: matches }, (_, index) => base + index);
  const results = await pool.run(seeds.map((seed, index) => ({
    seed, size, side: index % 2 === 0 ? "home" : "away", candidate: {}, opponent: {},
  })));
  const homeDiffs = results.map(({ goalsFor, goalsAgainst }, index) =>
    index % 2 === 0 ? goalsFor - goalsAgainst : goalsAgainst - goalsFor);
  const diffs = results.map(({ goalsFor, goalsAgainst }) => goalsFor - goalsAgainst);
  const sd = stdev(diffs);
  console.log(`Default baseline vs itself, ${matches} matches (${size}v${size}), candidate alternates sides`);
  const crashed = results.filter((r) => r.crashed);
  if (crashed.length > 0) console.log(`  ${crashed.length} crashed (counted as draws): seeds ${crashed.map((r) => r.seed).join(", ")}`);
  console.log(`  goals per match        ${f(mean(results.map((r) => r.goalsFor + r.goalsAgainst)), 2)}`);
  console.log(`  home-side advantage    ${f(mean(homeDiffs))} goals (mean of home minus away)`);
  console.log(`  candidate mean diff    ${f(mean(diffs))} (should be near 0)`);
  console.log(`  per-match std dev      ${f(sd)}`);
  for (const effect of [0.5, 0.3, 0.2, 0.1]) {
    console.log(`  matches to resolve a ${effect} goal edge at 2 standard errors: ${Math.ceil((2 * sd / effect) ** 2)}`);
  }
}

async function search(pool, values, size) {
  const generations = integer(values.generations, "--generations", 20);
  const population = integer(values.population, "--population", 16, 4);
  const matches = integer(values.matches, "--matches", 24, 2);
  const seed = integer(values.seed, "--seed", 1, 0);
  const out = values.out ?? "tuning-result.json";
  const opponent = values.opponent ? await readParams(values.opponent) : {};
  const random = mulberry32(seed);
  const elites = Math.max(2, Math.floor(population / 4));
  let center = [...DEFAULT_UNIT];
  let sigma = 0.2;
  const history = [];
  let best;

  console.log(`Search: ${generations} generations x ${population} candidates x ${matches} matches (${size}v${size})`);
  for (let generation = 0; generation < generations; generation += 1) {
    // Shared seeds per generation make candidates comparable. Fresh seeds each generation resist overfitting.
    const seeds = Array.from({ length: matches }, (_, index) => 10_000_000 + seed * 1_000_000 + generation * 1_000 + index);
    const candidates = [center, ...Array.from({ length: population - 1 }, () =>
      center.map((value) => Math.min(1, Math.max(0, value + sigma * gaussian(random)))))];
    const tasks = candidates.flatMap((unit) => matchTasks(toConfig(unit), opponent, seeds, size));
    const results = await pool.run(tasks);
    const scored = candidates.map((unit, index) => ({
      unit, ...summarize(results.slice(index * matches, (index + 1) * matches)),
    }));
    const ranked = [...scored].sort((a, b) => b.meanDiff - a.meanDiff);
    const topMean = mean(ranked.slice(0, elites).map(({ meanDiff }) => meanDiff));
    const entry = {
      generation, sigma, centerDiff: scored[0].meanDiff, populationMean: mean(scored.map(({ meanDiff }) => meanDiff)),
      topMean, center: toConfig(center),
    };
    history.push(entry);
    console.log(`gen ${String(generation + 1).padStart(2)}  sigma ${f(sigma, 2)}  center ${f(scored[0].meanDiff, 2)}` +
      `  pop mean ${f(entry.populationMean, 2)}  top ${elites} ${f(topMean, 2)}`);
    // Averaging the elites, not following the single best, keeps lucky candidates from steering the search.
    center = NAMES.map((_, dimension) => mean(ranked.slice(0, elites).map(({ unit }) => unit[dimension])));
    sigma = Math.max(0.04, sigma * 0.93);
    best = { config: toConfig(center), unit: center };
    await writeFile(out, `${JSON.stringify({ size, seed, matches, population, opponent, best, history }, null, 2)}\n`);
  }
  await writeFile(`${out}.teams.json`, `${JSON.stringify({ home: { pluginId: "baseline", config: best.config } }, null, 2)}\n`);
  console.log(`\nFinal center written to ${out} and ${out}.teams.json`);
  for (const name of NAMES) console.log(`  ${name.padEnd(15)} ${f(best.config[name])}`);
  console.log("This is a training-seed estimate. Confirm it with: tune compare --params " + out);
}

async function compare(pool, values, size) {
  if (!values.params) throw new Error("compare requires --params PATH");
  const candidate = await readParams(values.params);
  const opponent = values.opponent ? await readParams(values.opponent) : {};
  const matches = integer(values.matches, "--matches", 200, 4);
  const seeds = Array.from({ length: matches }, (_, index) => 900_000_000 + integer(values.seed, "--seed", 1, 0) * 100_000 + index);
  const results = await pool.run(matchTasks(candidate, opponent, seeds, size));
  const total = summarize(results);
  const side = (parity) => summarize(results.filter((_, index) => index % 2 === parity));
  console.log(`Candidate vs ${values.opponent ?? "default baseline"}, ${matches} held-out matches (${size}v${size})`);
  console.log(`  mean goal difference  ${f(total.meanDiff)} +/- ${f(1.96 * total.standardError)} (95% interval)`);
  console.log(`  as home ${f(side(0).meanDiff)}   as away ${f(side(1).meanDiff)}`);
  console.log(`  wins ${total.wins}  draws ${total.draws}  losses ${total.losses}`);
  console.log(`  goals for ${f(total.goalsFor, 2)}  against ${f(total.goalsAgainst, 2)} per match`);
  if (total.unfinished > 0) console.log(`  ${total.unfinished} matches hit the tick limit`);
  if (total.crashed > 0) console.log(`  ${total.crashed} matches crashed in the engine and count as draws`);
}

const { positionals, values } = parseArgs({ options, allowPositionals: true });
if (values.help || positionals.length !== 1 || !["variance", "search", "compare"].includes(positionals[0])) {
  console.log(usage);
  process.exit(values.help ? 0 : 2);
}
const size = values.size === "11" ? 11 : values.size === undefined || values.size === "5" ? 5 : NaN;
if (Number.isNaN(size)) throw new RangeError("--size must be 5 or 11");
const pool = new Pool(integer(values.workers, "--workers", Math.max(1, availableParallelism() - 1)));
try {
  await { variance, search, compare }[positionals[0]](pool, values, size);
} finally {
  await pool.close();
}
