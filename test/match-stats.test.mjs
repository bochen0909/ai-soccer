import assert from "node:assert/strict";
import test from "node:test";
import { createArcadeConfig, createDefaultAIRegistry, MatchSession, profileTeamOptions } from "../dist/index.js";
import { MatchStatsRecorder } from "../dist/runners/match-stats.js";

test("the match report adds up: goals match the score, possession sums to one, passes and shots are counted", () => {
  const config = createArcadeConfig();
  const session = new MatchSession(createDefaultAIRegistry());
  session.reset(config, 7, { teams: { home: profileTeamOptions("tuned"), away: profileTeamOptions("default") } });
  const recorder = new MatchStatsRecorder(config, 7,
    [{ name: "Ember FC", control: "CPU Hard" }, { name: "Tidal FC", control: "CPU", model: "test@1" }], "5-a-side");
  let snapshot = session.snapshot();
  for (let done = false; !done;) {
    const result = session.step({});
    snapshot = session.snapshot();
    recorder.record(result, snapshot);
    done = result.status === "finished";
  }
  const report = recorder.report(snapshot.score);
  assert.deepEqual(report.score, snapshot.score);
  assert.equal(report.goals.length, snapshot.score[0] + snapshot.score[1]);
  for (const [index, team] of report.teams.entries()) {
    assert.equal(team.stats.goals, snapshot.score[index]);
    assert.ok(team.stats.shots >= team.stats.shotsOnTarget - team.stats.goals);
    assert.ok(team.stats.passes > 0 && team.stats.passesCompleted <= team.stats.passes, JSON.stringify(team.stats));
    assert.ok(team.stats.tacklesWon <= team.stats.tackles);
    assert.ok(team.stats.distanceM > 500, `${team.stats.distanceM}`);
  }
  const possession = report.teams[0].stats.possession + report.teams[1].stats.possession;
  assert.ok(Math.abs(possession - 1) < 0.01, `${possession}`);
  assert.equal(report.teams[1].model, "test@1");
  assert.equal(report.durationSeconds, Math.round(snapshot.tick / 60));
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(report)));
});
