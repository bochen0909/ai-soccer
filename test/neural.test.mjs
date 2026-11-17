import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
  createArcadeConfig, createDefaultAIRegistry, createElevenAsideConfig, decodeWeights, evaluateMatch, forward,
  MatchSession, NeuralTeamAI, registerNeuralPolicy, SoccerTrainingEnv, TeamAIRegistry,
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

test('a separate goalkeeper network drives only the goalkeeper, and files without one still play with one network', () => {
  const env = new SoccerTrainingEnv();
  env.reset(1);
  const observation = env.observation();
  const home = env.config.teams[0].players;
  const intentsFor = (policy) => {
    const ai = new NeuralTeamAI(decodeWeights(policy));
    ai.reset({
      teamId: 'home', teamIndex: 0, players: home, opponents: env.config.teams[1].players, config: { greedy: true },
      seed: 1, contractVersion: 1,
    });
    return ai.decide(observation, { playerIds: home.map(({ id }) => id), revision: 0 }, { actionResults: [], passIntention: null });
  };
  // A keeper network that always picks move 3: a quarter turn from the attack direction, which is +y for the home side.
  const moveBias = new Float32Array(weights.moveCount);
  moveBias[3] = 1000;
  const keeperTensors = { ...weights.tensors, moveB: { shape: [weights.moveCount], data: Buffer.from(moveBias.buffer).toString('base64') } };
  const plain = intentsFor(weights);
  const split = intentsFor({ ...weights, keeper: { tensors: keeperTensors } });
  const keeperId = home.find(({ role }) => role === 'goalkeeper').id;
  for (const [index, intent] of split.entries()) {
    if (intent.playerId === keeperId) {
      assert.ok(Math.abs(intent.movement.x) < 1e-6 && intent.movement.y > 0.99, JSON.stringify(intent.movement));
      assert.notDeepEqual(intent.movement, plain[index].movement);
    } else {
      assert.deepEqual(intent, plain[index]);
    }
  }
});
