// Exercise the exact pilot path offline using deliberately synthetic scores.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { loadPilotCase, judgePilotCase } from './stylist6_nvidia_pilot.mjs';

const directory = fs.realpathSync(process.argv[2]);
const scenarios = JSON.parse(fs.readFileSync(path.join(directory, 'scenarios.json'), 'utf8'));
assert.equal(new Set(scenarios.map(({ id }) => id)).size, 12);

const results = [];
for (const { id } of scenarios) {
  const started = performance.now();
  const fixture = loadPilotCase(directory, id);
  const preferred = fixture.candidates.at(-1).id;
  const seed = fixture.candidates.find(({ origin }) => origin !== 'v5').id;
  const mockCalls = [];
  const result = await judgePilotCase(fixture, {
    apiKey: 'offline-mock-key-no-network',
    fetchImpl: async (url, init) => {
      assert.equal(url, 'https://integrate.api.nvidia.com/v1/chat/completions');
      const body = JSON.parse(init.body);
      const [prompt, ...media] = body.messages[0].content;
      assert.equal(media.filter((part) => part.type === 'image_url').length, 5);
      const candidateId = prompt.text.match(/candidateId must be exactly "([^"]+)"/)[1];
      mockCalls.push(candidateId);
      return Response.json({ choices: [{ finish_reason: 'stop', message: { content: JSON.stringify({
        candidateId, slotObservations: { top: 'light striped shirt', outerwear: 'short dark jacket', bottom: 'grey straight trousers', shoes: 'dark round toe shoes' },
        visualCohesion: 0.9, referenceFidelity: candidateId === preferred ? 1 : 0.7,
        styleFit: 0.9, occasionFit: 0.9, formalityFit: 0.9, confidence: 0.95, reject: candidateId === seed,
        relationships: [
          { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'mock top and layer match' },
          { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'mock footwear repeats bottom color' },
          { kind: 'reference', slots: ['top'], observation: 'mock reference repeats the top pattern' },
        ],
      }) } }] });
    },
  });
  assert.equal(result.providerCalls, 4);
  assert.equal(new Set(mockCalls).size, 4);
  assert.equal(result.v6Diagnostics.visualValid, fixture.candidates.filter(({ origin }) => origin !== 'v5').length);
  assert.equal(result.v6Diagnostics.visualRejected, 1);
  assert.equal(result.v6VisualPick, preferred);
  assert.notEqual(result.v6VisualPick, result.v6RetrievalPick);
  results.push({
    id, mockCalls: mockCalls.length, fixturePlusMockMs: performance.now() - started, mockVisualMs: result.wallMs,
    retrievalPick: result.v6RetrievalPick, mockSelection: result.v6VisualPick,
    hardConstraintsValidated: true,
  });
}
function percentiles(field) {
  const sorted = results.map((row) => row[field]).sort((a, b) => a - b);
  return { p50: sorted[Math.ceil(sorted.length * 0.50) - 1], p95: sorted[Math.ceil(sorted.length * 0.95) - 1] };
}
console.log(JSON.stringify({
  status: 'MOCK_ONLY', realProviderCalls: 0, independentRatings: 0,
  mockCalls: results.reduce((total, row) => total + row.mockCalls, 0),
  fixturePlusMockMs: percentiles('fixturePlusMockMs'), mockVisualMs: percentiles('mockVisualMs'), results,
}, null, 2));
