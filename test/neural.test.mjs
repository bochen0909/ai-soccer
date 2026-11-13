import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
  createArcadeConfig, createDefaultAIRegistry, createElevenAsideConfig, decodeWeights, evaluateMatch, forward,
  MatchSession, registerNeuralPolicy, TeamAIRegistry,
} from '../dist/index.js';

const weights = JSON.parse(readFileSync(new URL('../models/trained-5v5.json', import.meta.url), 'utf8'));
const fixture = JSON.parse(readFileSync(new URL('./fixtures/policy-check.json', import.meta.url), 'utf8'));

function registry() {
  return registerNeuralPolicy(createDefaultAIRegistry(), weights);
}

/** PyTorch's Categorical reports normalized log-probabilities, so normalize ours the same way before comparing. */
function logProbabilities(logits, legal = () => true) {
  const kept = Array.from(logits, (value, i) => legal(i) ? value : -Infinity);
  const top = Math.max(...kept);
  const total = Math.log(kept.reduce((sum, value) => sum + Math.exp(value - top), 0)) + top;
  return kept.map((value) => value - total);
}

test('the network gives the same probabilities as PyTorch on real observations', () => {
  const network = decodeWeights(weights);
  assert.equal(network.obsSize, fixture.obs[0].length);
  fixture.obs.forEach((observation, row) => {
    const output = forward(network, observation);
    const compare = (ours, theirs, label, legal = () => true) => ours.forEach((value, i) => {
      if (legal(i)) assert.ok(Math.abs(value - theirs[i]) < 1e-3, `${label} ${row}.${i}: ${value} vs ${theirs[i]}`);
    });
    compare(logProbabilities(output.move), fixture.moveLogits[row], 'move');
    compare(logProbabilities(output.sprint), fixture.sprintLogits[row], 'sprint');
    // PyTorch hides illegal acts with a huge negative number, so compare only the legal ones.
    const legal = (i) => fixture.mask[row][i] === 1;
    compare(logProbabilities(output.act, legal), fixture.actLogits[row], 'act', legal);
  });
});

test('the trained controller plays a whole match without failures and replays exactly', () => {
  const options = {
    config: Object.assign(createArcadeConfig(), { periodTicks: [600, 600] }),
    seed: 21,
    registry: registry(),
    teams: { home: { pluginId: 'neural' }, away: { pluginId: 'baseline' } },
  };
  const first = evaluateMatch(options);
  const second = evaluateMatch({ ...options, registry: registry() });
  assert.equal(first.status, 'finished');
  assert.equal(first.finalHash, second.finalHash);
  assert.deepEqual(first.score, second.score);
});

test('the trained controller only commands the players it is assigned', () => {
  const config = createArcadeConfig();
  const session = new MatchSession(registry());
  const failures = session.reset(config, 3, {
    human: { teamId: 'home', playerId: 'home-3' },
    teams: { home: { pluginId: 'neural' }, away: { pluginId: 'baseline' } },
  });
  assert.deepEqual(failures, []);
  for (let tick = 0; tick < 300; tick += 1) assert.deepEqual(session.step({}).controllerFailures, []);
});

test('a policy trained for five players refuses a larger roster', () => {
  const session = new MatchSession(registry());
  const failures = session.reset(createElevenAsideConfig(), 3, {
    teams: { home: { pluginId: 'neural' } },
  });
  assert.equal(failures.length, 1);
  assert.match(failures[0].message, /trained for 5 players/);
  assert.ok(TeamAIRegistry);
});
