import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { compareAuthenticatedPlan } from './development_shadow.mjs';

const now = Date.parse('2026-11-10T12:00:00Z');
const slots = ['top', 'outerwear', 'bottom', 'shoes'];
function fixture() {
  const bytes = Buffer.alloc(256, 5);
  Buffer.from([0xff, 0xd8, 0xff, 0xc0, 0, 8, 8, 0, 16, 0, 16, 3]).copy(bytes);
  bytes.writeUInt16BE(0xffd9, bytes.length - 2);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  const catalog = new Map(), photos = new Map(), offers = [], assets = [];
  const grant = {
    rightsHolder: 'Fixture owner', reviewedBy: 'Fixture reviewer', status: 'active',
    validFrom: '2026-01-01T00:00:00Z', expiresAt: null,
    evidence: { reference: 'test', sha256 },
    nvidiaTrial: { submit: true, territory: 'global', providerServiceImprovement: true,
      terms: 'nvidia-api-trial', purposes: ['internal_outfit_evaluation'] },
  };
  const reference = { id: 'ref', bytes, sha256, sourceUrl: 'https://example.test/ref' };
  assets.push({ ...grant, kind: 'reference', id: reference.id, sha256, sourceUrl: reference.sourceUrl });
  const candidates = Array.from({ length: 4 }, (_, index) => ({
    id: 'look-' + index,
    concept: { id: 'concept-' + index },
    items: slots.map((slot) => {
      const sku = slot + '-' + index;
      const imageUrl = 'https://example.test/' + sku + '.jpg';
      const url = 'https://example.test/' + sku;
      catalog.set(sku, { id: sku, source_url: url, image_url: imageUrl, is_active: true, price: 90, currency: 'UZS' });
      photos.set(sku, { bytes, sha256 });
      assets.push({ ...grant, kind: 'product', id: sku, sha256, sourceUrl: imageUrl });
      offers.push({ sku, variantId: sku + '-M', url, size: 'M', sizeSystem: 'merchant-alpha',
        country: 'UZ', shippingConfirmed: true, stock: 'in_stock', currency: 'UZS', price: 90,
        landedPrice: 100, observedAt: '2026-11-10T10:00:00Z', source: 'merchant_api',
        evidenceRef: 'fixture', evidenceSha256: sha256 });
      return { id: 'shop_' + sku, imageUrl, macroCategory: slot, isShopItem: true, subtype: slot, price: 90, currency: 'UZS' };
    }),
  }));
  return {
    mode: 'private-stylist6-shadow', scenario: { synthetic: true, style: 'classic', occasion: 'Work',
      brief: 'Private benchmark', userEmail: 'should-not-leave@example.test' }, now,
    apiKey: 'mock-key-no-network-0123456', runId: 'synthetic-run', claimRun: () => true,
    plan: { candidates }, baselineDeck: candidates.slice(0, 3), catalog, photos, reference,
    rights: { version: 1, assets }, offers, purchase: { country: 'UZ', currency: 'UZS', budget: 400,
      sizes: Object.fromEntries(slots.map((slot) => [slot, { size: 'M', sizeSystem: 'merchant-alpha' }])) },
  };
}

function mock(responseIds, rejects = []) {
  return async (url, init) => {
    assert.equal(url, 'https://integrate.api.nvidia.com/v1/chat/completions');
    assert.doesNotMatch(init.body, /should-not-leave|example\.test/);
    const prompt = JSON.parse(init.body).messages[0].content[0].text;
    const id = prompt.match(/candidateId must be exactly "([^"]+)"/)[1];
    responseIds.push(id);
    return Response.json({ choices: [{ finish_reason: 'stop', message: { content: JSON.stringify({
      candidateId: id, visualCohesion: .9, referenceFidelity: .9, styleFit: .9, occasionFit: .9,
      formalityFit: .9, confidence: .9, reject: rejects.includes(id),
      slotObservations: { top: 'light striped shirt', outerwear: 'short dark jacket',
        bottom: 'grey straight trousers', shoes: 'dark round toe shoes' },
      relationships: [
        { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'mock top and layer match' },
        { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'mock footwear repeats bottom color' },
        { kind: 'reference', slots: ['top'], observation: 'mock reference repeats the top pattern' },
      ],
    }) } }] });
  };
}

test('shadow judges only rights-cleared, purchase-verified exact looks and reuses original looks', async () => {
  const input = fixture(), calls = [];
  input.offers.find(({ sku }) => sku === 'top-0').stock = 'out_of_stock';
  const result = await compareAuthenticatedPlan({ ...input, fetchImpl: mock(calls) });
  assert.deepEqual(new Set(calls), new Set(['look-1', 'look-2', 'look-3']));
  assert.equal(result.baselineDeck, input.baselineDeck);
  assert.equal(result.visualDeck.length, 3);
  assert.ok(result.visualDeck.every((look) => input.plan.candidates.includes(look)));
});

test('shadow withholds incomplete, unlicensed and over-budget decks without replacing the baseline', async () => {
  const input = fixture(), calls = [];
  const incomplete = await compareAuthenticatedPlan({ ...input, fetchImpl: mock(calls, ['look-0', 'look-1']) });
  assert.equal(incomplete.reason, 'no_complete_visual_deck');
  assert.equal(calls.length, 4);
  assert.equal(incomplete.visualDeck, null);
  for (const override of [
    { mode: 'production' }, { rights: null }, { purchase: { ...input.purchase, budget: 399 } },
  ]) {
    const result = await compareAuthenticatedPlan({ ...input, ...override, fetchImpl: () => assert.fail('unexpected dispatch') });
    assert.equal(result.visualDeck, null);
    assert.equal(result.baselineDeck, input.baselineDeck);
  }
});

test('changed merchant prices or switched photo URLs cannot be shown as exact verified offers', async () => {
  const input = fixture(), calls = [];
  input.offers.find(({ sku }) => sku === 'top-0').price = 95;
  input.plan.candidates[1].items[0].imageUrl = 'https://example.test/different-variant.jpg';
  const result = await compareAuthenticatedPlan({ ...input, fetchImpl: mock(calls) });
  assert.equal(result.reason, 'insufficient_verified_candidates');
  assert.equal(result.diagnostics.eligible, 2);
  assert.deepEqual(calls, []);
});


test('only one admitted shadow call can judge a plan with the same run identity', async () => {
  const input = fixture(), seen = new Set(), calls = [];
  input.claimRun = (id) => {
    if (seen.has(id)) return false;
    seen.add(id);
    return true;
  };
  const results = await Promise.all(Array.from({ length: 2 }, () =>
    compareAuthenticatedPlan({ ...input, fetchImpl: mock(calls) })));
  assert.equal(calls.length, 4);
  assert.equal(results.filter((row) => row.reason === 'shadow_already_claimed').length, 1);
  assert.equal(results.filter((row) => row.visualDeck?.length === 3).length, 1);
});
