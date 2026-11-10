import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { proposeVisualFirstDeck, shortlistRetrievalDeck } from './prototype.mjs';
import { createHostedVisualCritic } from './visual_critic_hosted.mjs';

// Isolated Stylist 6 + NVIDIA integration: retrieval -> real NVIDIA critic
// adapter (stubbed transport, recorded provider shape) -> deck selection.
// Proves actual outfit generation through the NVIDIA path with zero spend.
const catalog = [];
function add(slot, subtype, colors, price = 170000) {
  for (const [n, color] of colors.entries()) {
    const id = slot + '-' + subtype + '-' + n;
    catalog.push({
      id, name: color + ' ' + subtype, macroCategory: slot, subtype,
      color, texture: 'smooth',
      imageUrl: 'https://example.com/p/' + id + '.jpg',
      shopUrl: 'https://example.com/buy/' + id,
      price: price + n * 7000, currency: 'UZS', isShopItem: true,
    });
  }
}
add('top', 'fine-knit', ['grey', 'brown']);
add('outerwear', 'coat', ['camel', 'charcoal']);
add('bottom', 'tailored-trouser', ['brown', 'charcoal']);
add('shoes', 'boot', ['brown', 'black']);

const briefs = [
  ['rt_005', 'Brown tonal layering', 'rt_005', ['fine-knit'], ['coat'], ['tailored-trouser'], ['boot'], 'required', ['brown'], 'balanced'],
  ['house_1', 'Shirt and blazer', undefined, ['shirt'], ['blazer'], ['tailored-trouser'], ['loafer'], 'required', ['navy'], 'tailored'],
  ['house_2', 'Knit and cardigan', undefined, ['fine-knit'], ['cardigan'], ['chino'], ['loafer'], 'required', ['grey'], 'soft'],
  ['house_3', 'Polo and overshirt', undefined, ['polo'], ['overshirt'], ['jeans'], ['minimal-sneaker'], 'required', ['beige'], 'relaxed'],
  ['house_4', 'Tee and bomber', undefined, ['t-shirt'], ['bomber'], ['jeans'], ['minimal-sneaker'], 'required', ['black'], 'relaxed'],
  ['house_5', 'Shirt no layer', undefined, ['shirt'], [], ['chino'], ['derby'], 'absent', ['white'], 'tailored'],
].map(([id, name, referenceId, top, outerwear, bottom, shoes, layerPolicy, palette, silhouette]) => ({
  id, name, ...(referenceId ? { referenceId } : {}),
  roles: { top, outerwear, bottom, shoes },
  layerPolicy, palette, silhouette,
}));

const jpeg = Buffer.alloc(256, 7);
Buffer.from([0xff, 0xd8, 0xff, 0xc0, 0, 8, 8, 0, 16, 0, 16, 3]).copy(jpeg);
jpeg.writeUInt16BE(0xffd9, jpeg.length - 2);
const asset = { bytes: jpeg, sha256: createHash('sha256').update(jpeg).digest('hex') };

test('NVIDIA verdict reverses the feasible seed winner; rejected and unknown looks never surface', async () => {
  const base = {
    request: { style: 'old_money', occasion: 'Work', sourceMode: 'catalog', referenceId: 'rt_005',
      budget: { amount: 708000, currency: 'UZS' } },
    catalog,
    creativeDirector: async () => briefs,
    hardGate: async (items) => ({ ok: !items.some((i) => i.id === 'top-fine-knit-1') }),
    priceToBudgetCurrency: (i) => i.price,
    options: { target: 1, maxHostedCandidates: 3 },
  };
  const retrieval = await shortlistRetrievalDeck(base);
  const [first, second, third] = retrieval.shortlist;
  assert.ok(first && second && third);
  const seen = [];
  const critic = createHostedVisualCritic({
    apiKey: 'test-nvidia-key-0123456789abcdef', provider: 'nvidia',
    resolveProduct: ({ id }) => {
      assert.notEqual(id, 'top-fine-knit-1');
      return asset;
    },
    resolveReference: (id) => { assert.equal(id, 'rt_005'); return asset; },
    fetchImpl: async (url, init) => {
      assert.equal(url, 'https://integrate.api.nvidia.com/v1/chat/completions');
      const { content } = JSON.parse(init.body).messages[0];
      assert.equal(content.filter((part) => part.type === 'image_url').length, 5);
      const id = content[0].text.match(/candidateId must be exactly "([^"]+)"/)[1];
      seen.push(id);
      return Response.json({ choices: [{ message: { content: JSON.stringify({
        slotObservations: { top: 'light striped shirt', outerwear: 'short dark jacket', bottom: 'grey straight trousers', shoes: 'dark round toe shoes' },
        candidateId: id === third.id ? 'unknown' : id, visualCohesion: 0.9, referenceFidelity: 0.9,
        styleFit: 0.9, occasionFit: 0.9, formalityFit: 0.9, confidence: 0.95, reject: id === first.id,
        relationships: [
          { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'mock top and layer match' },
          { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'mock footwear repeats bottom color' },
          { kind: 'reference', slots: ['top'], observation: 'mock reference repeats the top pattern' },
        ],
      }) } }] });
    },
  });
  const result = await proposeVisualFirstDeck({ ...base, visualCritic: critic });
  assert.deepEqual(new Set(seen), new Set([first.id, second.id, third.id]));
  assert.equal(result.outfits[0].id, second.id);
  assert.equal(result.outfits[0].items.length, 4);
  assert.ok(result.outfits[0].items.reduce((sum, i) => sum + i.price, 0) <= base.request.budget.amount);
  assert.equal(result.diagnostics.visualValid, 2);
  assert.equal(result.diagnostics.visualRejected, 1);
  assert.equal(result.diagnostics.visualAbstained, 1);
  assert.equal(result.diagnostics.visualApproved, 1);
});
