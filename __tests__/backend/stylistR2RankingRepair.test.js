/** @jest-environment node */
// Stylist 5 r2 — NVIDIA ranking contract reliability (workstream 1).
// The November 9 production log showed a 12/12 response discarded as
// `rank_contract_mismatch` because one candidate ID was repeated: correct
// count, one duplicate. A single repeated ID must no longer destroy all
// provider evidence when a safe repair is possible.
const {
  validateLookRanking,
  repairLookRanking,
  rankNvidiaLooks,
  nvidiaRankingBody,
} = require('../../backend/supabase/functions/_shared/nvidiaLookRanking');

const SHORTLIST = Array.from({ length: 12 }, (_, i) => `look${String(i + 1).padStart(3, '0')}`);

test('a valid provider ranking is fully accepted as AI ranking', () => {
  const received = [...SHORTLIST].reverse();
  const result = repairLookRanking(JSON.stringify({ rankedLookIds: received }), SHORTLIST);
  expect(result.status).toBe('ok');
  expect(result.ids).toEqual(received);
  expect(result.repaired).toBe(false);
});

test('the observed production failure (12 received, 1 duplicate) is repaired, not discarded', () => {
  // look005 appears twice; look009 is missing — exactly the observed shape.
  const received = ['look001', 'look002', 'look003', 'look004', 'look005', 'look006',
    'look005', 'look007', 'look008', 'look010', 'look011', 'look012'];
  const strict = validateLookRanking(JSON.stringify({ rankedLookIds: received }), SHORTLIST);
  expect(strict.ids).toBeNull();
  expect(strict.status).toBe('rank_contract_mismatch');

  const repaired = repairLookRanking(JSON.stringify({ rankedLookIds: received }), SHORTLIST);
  expect(repaired.status).toBe('repaired_hybrid');
  expect(repaired.repaired).toBe(true);
  expect(repaired.ids).toHaveLength(12);
  // Every valid unique ID keeps its provider order; the missing shortlist
  // member is filled deterministically at the tail.
  expect(repaired.ids.slice(0, 11)).toEqual(
    ['look001', 'look002', 'look003', 'look004', 'look005', 'look006',
      'look007', 'look008', 'look010', 'look011', 'look012']);
  expect(repaired.ids[11]).toBe('look009');
  expect(new Set(repaired.ids).size).toBe(12);
});

test('unknown (invented) provider IDs are dropped and never accepted', () => {
  const received = [...SHORTLIST.slice(0, 10), 'look007', 'invented-product-1'];
  const repaired = repairLookRanking(JSON.stringify({ rankedLookIds: received }), SHORTLIST);
  expect(repaired.status).toBe('repaired_hybrid');
  expect(repaired.ids).toHaveLength(12);
  expect(repaired.ids).not.toContain('invented-product-1');
  expect(new Set(repaired.ids).size).toBe(12);
  for (const id of SHORTLIST) expect(repaired.ids).toContain(id);
});

test('a response with too little valid signal falls back fully instead of repairing', () => {
  // Only 5 of 12 shortlist IDs are valid — below the half-shortlist floor.
  const received = ['look001', 'look002', 'look003', 'look004', 'look005',
    'ghost-1', 'ghost-2', 'ghost-3', 'ghost-4', 'ghost-5', 'ghost-6', 'ghost-7'];
  const repaired = repairLookRanking(JSON.stringify({ rankedLookIds: received }), SHORTLIST);
  expect(repaired.ids).toBeNull();
  expect(repaired.status).toBe('rank_contract_mismatch');
});

test('malformed and empty responses are never repaired', () => {
  expect(repairLookRanking('not json at all', SHORTLIST).status).toBe('invalid_json');
  expect(repairLookRanking('not json at all', SHORTLIST).ids).toBeNull();
  expect(repairLookRanking(JSON.stringify({ somethingElse: [] }), SHORTLIST).status)
    .toBe('invalid_schema');
  expect(repairLookRanking(JSON.stringify({ rankedLookIds: 'look001' }), SHORTLIST).status)
    .toBe('invalid_schema');
});

test('JSON wrapped in prose or fences is still parsed consistently', () => {
  const body = `Here is the ranking:\n\`\`\`json\n${JSON.stringify({ rankedLookIds: [...SHORTLIST].reverse() })}\n\`\`\`\nThanks!`;
  const result = repairLookRanking(body, SHORTLIST);
  expect(result.status).toBe('ok');
  expect(result.ids).toEqual([...SHORTLIST].reverse());
});

test('end-to-end: a duplicated provider response yields hybrid provenance, not fallback', async () => {
  const received = ['look001', 'look002', 'look003', 'look004', 'look005', 'look006',
    'look005', 'look007', 'look008', 'look010', 'look011', 'look012'];
  const fetchMock = jest.fn(async () => new Response(JSON.stringify({
    choices: [{ finish_reason: 'stop', message: { content: JSON.stringify({ rankedLookIds: received }) } }],
  })));
  const result = await rankNvidiaLooks('fixture-key', 'nvidia/nemotron-3.5-lightning-30b-a3b',
    'Rank existing looks', {}, SHORTLIST, fetchMock);
  expect(fetchMock).toHaveBeenCalledTimes(1);
  expect(result.ids).toHaveLength(12);
  expect(result.attempt.status).toBe('repaired_hybrid');
  expect(result.attempt.repaired).toBe(true);
  expect(result.attempt.duplicates).toBe(1);
  expect(result.attempt.filledMissing).toBe(1);
  expect(JSON.stringify(result)).not.toContain('fixture-key');
});

test('end-to-end: an unrepairable response stays a truthful deterministic fallback', async () => {
  const fetchMock = jest.fn(async () => new Response(JSON.stringify({
    choices: [{ message: { content: '{"rankedLookIds":["look001","look002"]}' } }],
  })));
  const result = await rankNvidiaLooks('fixture-key', 'fixture-model', 'Rank', {},
    ['look127', 'look208', 'look009'], fetchMock);
  expect(result.ids).toBeNull();
  expect(result.attempt).toMatchObject({ status: 'rank_contract_mismatch', expected: 3, received: 2, unknown: 2 });
});

test('rank prompts forbid repeating, omitting, or inventing IDs', () => {
  const prompt = nvidiaRankingBody('fixture-model', 'Rank', {}, ['look127', 'look009']).messages[0].content;
  expect(prompt).toContain('never repeat');
  expect(prompt).toContain('invent');
  expect(prompt).toContain(JSON.stringify(['look127', 'look009']));
  expect(prompt).not.toContain('look001');
});
