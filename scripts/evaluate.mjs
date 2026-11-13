import { link, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { arch, availableParallelism, cpus, platform, release, totalmem } from "node:os";
import { dirname, join, resolve } from "node:path";
import { parseArgs } from "node:util";
import { benchmarkMatches, evaluateMatch, verifyReplay } from "../dist/index.js";

const usage = `Usage:
  npm run --silent evaluate -- match [--seed N] [--matches N] [--replay PATH]
  npm run --silent evaluate -- verify PATH
  npm run --silent evaluate -- benchmark [--seed N] [--matches N] [--warmup N]

Match and benchmark options:
  --seed N          First unsigned 32-bit seed (default 17). Multiple matches use successive seeds.
  --config PATH     Match configuration JSON (default arcade configuration).
  --scenario PATH   Scenario definition JSON.
  --teams PATH      Team controller options JSON keyed by team ID. The CLI provides the baseline plugin.
  --max-ticks N     Stop each match after N simulation ticks.

The match command defaults to one match. --replay writes a single match to a new file.
The verify command plays back recorded actions and checks the state hashes.
The benchmark runs matches serially and excludes replay file I/O.
`;

const sharedOptions = {
  help: { type: "boolean" },
  seed: { type: "string" },
  matches: { type: "string" },
  config: { type: "string" },
  scenario: { type: "string" },
  teams: { type: "string" },
  "max-ticks": { type: "string" },
};

function integer(value, name, minimum, maximum = Number.MAX_SAFE_INTEGER) {
  if (value === undefined) return undefined;
  const number = Number(value);
  if (!/^\d+$/.test(value) || !Number.isSafeInteger(number) || number < minimum || number > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} to ${maximum}`);
  }
  return number;
}

async function readJson(path) {
  if (path === undefined) return undefined;
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch (error) {
    throw new Error(`Cannot read JSON from ${path}: ${error.message}`);
  }
}

async function readOptions(path) {
  const value = await readJson(path);
  if (path !== undefined && (!value || typeof value !== "object" || Array.isArray(value))) {
    throw new TypeError(`Expected a JSON object in ${path}`);
  }
  return value;
}

async function writeReplay(path, replay) {
  const destination = resolve(path);
  const stagingDirectory = await mkdtemp(join(dirname(destination), ".ai-soccer-replay-"));
  try {
    const stagingFile = join(stagingDirectory, "replay.json");
    await writeFile(stagingFile, `${JSON.stringify(replay)}\n`);
    // Publishing by hard link exposes the complete replay and fails if the destination exists.
    try {
      await link(stagingFile, destination);
    } catch (error) {
      if (error.code === "EEXIST") throw new Error(`Replay file already exists at ${destination}`);
      throw error;
    }
  } finally {
    await rm(stagingDirectory, { recursive: true, force: true });
  }
  return destination;
}

function hardware() {
  const processors = cpus();
  return {
    nodeVersion: process.version,
    platform: platform(),
    osRelease: release(),
    architecture: arch(),
    cpuModel: processors[0]?.model ?? null,
    logicalCpuCount: processors.length,
    availableParallelism: availableParallelism(),
    totalMemoryBytes: totalmem(),
  };
}

async function main([command, ...args]) {
  if (command === undefined || command === "--help" || command === "-h" || command === "help") {
    process.stdout.write(usage);
    return;
  }

  if (command === "verify") {
    const { values, positionals } = parseArgs({ args, options: { help: { type: "boolean" } }, allowPositionals: true });
    if (values.help) {
      process.stdout.write(usage);
      return;
    }
    if (positionals.length !== 1) throw new Error("verify requires exactly one replay path");
    printJson(verifyReplay(await readJson(positionals[0])));
    return;
  }

  if (command !== "match" && command !== "benchmark") throw new Error(`Unknown command ${command}. Use --help for usage.`);
  const options = command === "match"
    ? { ...sharedOptions, replay: { type: "string" } }
    : { ...sharedOptions, warmup: { type: "string" } };
  const { values } = parseArgs({ args, options });
  if (values.help) {
    process.stdout.write(usage);
    return;
  }

  const seed = integer(values.seed, "--seed", 0, 0xffff_ffff) ?? 17;
  const matches = integer(values.matches, "--matches", 1);
  const maxTicks = integer(values["max-ticks"], "--max-ticks", 1);
  const warmup = integer(values.warmup, "--warmup", 0);
  const replayPath = values.replay;
  if (replayPath !== undefined && matches !== undefined && matches !== 1) {
    throw new Error("--replay requires exactly one match");
  }
  const [config, scenario, teams] = await Promise.all([
    readOptions(values.config), readOptions(values.scenario), readOptions(values.teams),
  ]);
  const evaluation = { config, scenario, teams, maxTicks };

  if (command === "benchmark") {
    const result = benchmarkMatches({ ...evaluation, seed, matches, warmup });
    const { rss, heapUsed } = process.memoryUsage();
    printJson({
      ...result, hardware: hardware(), processMemoryAfterBenchmark: { rssBytes: rss, heapUsedBytes: heapUsed },
    });
    return;
  }

  if ((matches ?? 1) === 1) {
    const { replay, ...result } = evaluateMatch({ ...evaluation, seed, recordReplay: replayPath !== undefined });
    if (replayPath === undefined) printJson(result);
    else {
      if (replay === undefined) throw new Error("The evaluation did not produce a replay");
      printJson({ ...result, replayPath: await writeReplay(replayPath, replay) });
    }
    return;
  }

  const results = Array.from({ length: matches }, (_, index) => evaluateMatch({ ...evaluation, seed: (seed + index) >>> 0 }));
  printJson({ matches: results });
}

function printJson(value) {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

try {
  await main(process.argv.slice(2));
} catch (error) {
  process.stderr.write(`Evaluation failed: ${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
}
