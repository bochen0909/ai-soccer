import assert from 'node:assert/strict';
import test from 'node:test';
import {
  AI_PROFILES, createArcadeConfig, createDefaultAIRegistry, createElevenAsideConfig, evaluateMatch,
  isAIProfileId, MatchSession, profileSupports, profileTeamOptions,
} from '../dist/index.js';

function shortMatch() {
  const config = createArcadeConfig();
  config.periodTicks = [1800, 1800];
  return config;
}

test('AI profiles have valid settings and know which formats they cover', () => {
  assert.ok(isAIProfileId('tuned') && isAIProfileId('default') && !isAIProfileId('hasOwnProperty'));
  for (const profile of Object.values(AI_PROFILES)) {
    for (const value of Object.values(profile.config)) assert.ok(Number.isFinite(value) && value >= 0);
  }
  assert.equal(profileSupports('default', 11), true);
  assert.equal(profileSupports('tuned', 5), true);
  assert.equal(profileSupports('tuned', 11), false);
  assert.equal(createElevenAsideConfig().teams[0].players.length, 11);
  const options = profileTeamOptions('tuned');
  options.config.aggression = 0;
  assert.notEqual(AI_PROFILES.tuned.config.aggression, 0, 'options are copies');
});

test('a computer-versus-computer session runs without a human player', () => {
  const config = shortMatch();
  const session = new MatchSession(createDefaultAIRegistry());
  const teams = {
    home: profileTeamOptions('tuned'),
    away: profileTeamOptions('default'),
  };
  assert.deepEqual(session.reset(config, 7, { teams }), []);
  let last;
  for (let i = 0; i < 600; i += 1) last = session.step({});
  assert.deepEqual(last.controllerFailures, []);
  assert.equal(session.snapshot().selectedPlayerId, null);
  assert.throws(() => session.step({ movement: { x: 1, y: 0 } }), /no human player/);
});

test('the tuned profile beats the default profile from either side', () => {
  let difference = 0;
  for (const [index, seed] of [101, 102, 103, 104].entries()) {
    const tunedHome = index % 2 === 0;
    const { score } = evaluateMatch({
      config: shortMatch(), seed,
      teams: {
        home: profileTeamOptions(tunedHome ? 'tuned' : 'default'),
        away: profileTeamOptions(tunedHome ? 'default' : 'tuned'),
      },
    });
    difference += tunedHome ? score[0] - score[1] : score[1] - score[0];
  }
  assert.ok(difference > 0, `tuned goal difference ${difference}`);
});

test('the trained profile runs as a team through the same wiring the browser app uses', async () => {
  const { readFileSync } = await import('node:fs');
  const { registerNeuralPolicy } = await import('../dist/index.js');
  const weights = JSON.parse(readFileSync(new URL('../models/trained-5v5.json', import.meta.url), 'utf8'));
  assert.equal(AI_PROFILES.trained.plugin, 'neural');
  assert.equal(profileTeamOptions('trained').pluginId, 'neural');
  assert.equal(profileSupports('trained', 5), true);
  assert.equal(profileSupports('trained', 11), false);
  const session = new MatchSession(registerNeuralPolicy(createDefaultAIRegistry(), weights));
  assert.deepEqual(session.reset(createArcadeConfig(), 4, {
    teams: { home: profileTeamOptions('trained'), away: profileTeamOptions('tuned') },
  }), []);
  for (let tick = 0; tick < 400; tick += 1) assert.deepEqual(session.step().controllerFailures, []);
});
