// Round-robin tournament between controllers on shared seeds, both sides each, with Elo-style ratings.
//   npm run --silent arena -- default tuned model:/abs/a.json model:/abs/b.json --matches 128
// Players: "default", "tuned", or "model:<absolute path to an exported policy JSON>". A bare path to a .json is
// treated as a model. Use --vs NAME to play only pairs that include that player (cheaper than all pairs).
import { availableParallelism } from "node:os";
import { basename, resolve } from "node:path";
import { parseArgs } from "node:util";
import { Worker } from "node:worker_threads";

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: { matches: { type: "string", default: "128" }, seed: { type: "string", default: "1" }, vs: { type: "string" }, workers: { type: "string" } },
});
if (positionals.length < 2) { console.log("Usage: arena PLAYER PLAYER... [--matches N] [--seed N] [--vs PLAYER]"); process.exit(2); }

const players = positionals.map((raw) => {
  const greedy = raw.startsWith("greedy:");
  const p = greedy ? raw.slice(7) : raw;
  const base = p.endsWith(".json") ? `model:${resolve(p.replace(/^model:/, ""))}` : p.startsWith("model:") ? `model:${resolve(p.slice(6))}` : p;
  const spec = (greedy ? "greedy:" : "") + base;
  return { spec, name: (greedy ? "greedy " : "") + (spec.includes("model:") ? basename(spec, ".json") : spec.replace("greedy:", "")) };
});
const matches = Number(values.matches);
const seeds = Array.from({ length: matches }, (_, i) => 900_000_000 + Number(values.seed) * 100_000 + i);
const pairs = [];
for (let i = 0; i < players.length; i += 1) for (let j = i + 1; j < players.length; j += 1) {
  if (values.vs && ![players[i].name, players[j].name].includes(values.vs)) continue;
  pairs.push([i, j]);
}

const workers = Array.from({ length: Number(values.workers ?? Math.max(1, availableParallelism() - 1)) }, () => new Worker(new URL("./arena-worker.mjs", import.meta.url)));
const tasks = pairs.flatMap(([i, j]) => seeds.map((seed, k) => ({ pair: [i, j], seed, a: players[i].spec, b: players[j].spec, side: k % 2 === 0 ? "home" : "away" })));
const results = await new Promise((done, fail) => {
  const out = new Array(tasks.length); let next = 0; let left = tasks.length;
  for (const worker of workers) {
    worker.on("error", fail);
    worker.on("message", (m) => { out[m.id] = m; if (--left === 0) done(out); else feed(worker); });
    feed(worker);
  }
  function feed(worker) { if (next < tasks.length) { const id = next++; worker.postMessage({ id, ...tasks[id] }); } }
});
await Promise.all(workers.map((w) => w.terminate()));

// Per pair: mean goal difference of the first player, with a 95% interval.
const stats = pairs.map(([i, j]) => {
  const rs = results.filter((_, k) => tasks[k].pair[0] === i && tasks[k].pair[1] === j);
  const d = rs.map((r) => r.goalsFor - r.goalsAgainst);
  const mean = d.reduce((s, x) => s + x, 0) / d.length;
  const sd = Math.sqrt(d.reduce((s, x) => s + (x - mean) ** 2, 0) / (d.length - 1));
  return { i, j, mean, half: 1.96 * sd / Math.sqrt(d.length), w: d.filter((x) => x > 0).length, dr: d.filter((x) => x === 0).length, l: d.filter((x) => x < 0).length, crashed: rs.filter((r) => r.crashed).length };
});
console.log(`${matches} matches per pair on shared seeds, sides alternate (goal difference is from the first player's view)\n`);
for (const s of stats) {
  console.log(`${players[s.i].name.padEnd(18)} vs ${players[s.j].name.padEnd(18)} ${s.mean >= 0 ? "+" : ""}${s.mean.toFixed(2)} +/- ${s.half.toFixed(2)}   W${s.w} D${s.dr} L${s.l}${s.crashed ? `  (${s.crashed} crashed)` : ""}`);
}
// Ratings: least squares on mean goal differences (rating gap = expected goal difference), mean fixed at 0.
const n = players.length; const r = new Array(n).fill(0);
for (let it = 0; it < 2000; it += 1) {
  const grad = new Array(n).fill(0);
  for (const s of stats) { const e = r[s.i] - r[s.j] - s.mean; grad[s.i] += e; grad[s.j] -= e; }
  for (let k = 0; k < n; k += 1) r[k] -= 0.05 * grad[k];
  const m = r.reduce((a, b) => a + b, 0) / n; for (let k = 0; k < n; k += 1) r[k] -= m;
}
console.log("\nRating (expected goal difference against an average opponent in this field):");
players.map((p, k) => [p.name, r[k]]).sort((a, b) => b[1] - a[1]).forEach(([name, v]) => console.log(`  ${name.padEnd(20)} ${v >= 0 ? "+" : ""}${v.toFixed(2)}`));
