import test from 'node:test';
import assert from 'node:assert/strict';
import { proposeVisualFirstDeck } from './prototype.mjs';
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

const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, ...new Array(2040).fill(7)]);
const approved = JSON.stringify({
  candidateId: 'REPLACE', visualCohesion: 0.9, styleFit: 0.88, occasionFit: 0.91,
  formalityFit: 0.93, confidence: 0.95, reject: false,
  evidence: ['tonal brown register holds', 'coat weight suits autumn', 'boot formality fits'],
});

test('NVIDIA critic verdicts drive real outfit selection (stubbed transport)', async () => {
  let providerCalls = 0;
  // Product photos only; provider calls are recorded below.
  const stubFetch = async (url) => {
    assert.ok(!String(url).includes('integrate.api.nvidia.com'));
    return { ok: true, status: 200, arrayBuffer: async () => jpeg };
  };
  const seen = [];
  const recordingFetch = async (url, init) => {
    if (String(url).includes('integrate.api.nvidia.com')) {
      providerCalls += 1;
      const body = JSON.parse(init.body);
      assert.equal(body.model, 'nvidia/nemotron-3.5-super-vl-120b-a12b');
      assert.equal(body.stream, false);
      const parts = body.messages[0].content;
      const images = parts.filter((p) => p.type === 'image_url');
      assert.ok(images.length >= 2, 'reference + product photos required');
      assert.ok(images.every((p) => p.image_url.url.startsWith('data:image/')));
      const text = parts.find((p) => p.type === 'text').text;
      const id = text.match(/candidateId must be exactly "([^"]+)"/)[1];
      seen.push(id);
      return {
        ok: true, status: 200,
        json: async () => ({
          choices: [{ message: { content: approved.replace('REPLACE', id) } }],
        }),
      };
    }
    return stubFetch(url, init);
  };
  const critic = createHostedVisualCritic({
    apiKey: 'test-nvidia-key-0123456789abcdef', provider: 'nvidia',
    fetchImpl: recordingFetch,
    resolveReference: async () => ({ base64: jpeg.toString('base64'), mimeType: 'image/jpeg' }),
  });
  const result = await proposeVisualFirstDeck({
    request: { style: 'old_money', occasion: 'Work', sourceMode: 'catalog' },
    catalog,
    creativeDirector: async () => briefs,
    visualCritic: critic,
    hardGate: async () => ({ ok: true }),
    priceToBudgetCurrency: (i) => i.price,
    options: { briefLimit: 6, perRole: 6, maxVisual: 6, perBrief: 20, minVisualQuality: 0.5 },
  });
  // Engine floors maxVisual at 12; 8 candidates -> 8 judged calls.
  assert.ok(providerCalls > 0 && providerCalls <= 12);
  assert.ok(result.outfits.length > 0, 'NVIDIA approvals must yield real outfits');
  for (const outfit of result.outfits) {
    assert.equal(outfit.items.length, 4);
    assert.match(outfit.concept.referenceId || '', /^rt_\d{3}$/);
    assert.ok(outfit.visualQuality >= 0.5);
  }
  assert.ok(result.diagnostics.visualApproved > 0);
});

test('NVIDIA provider rejects unconfigured keys without transport', () => {
  assert.throws(() => createHostedVisualCritic({ provider: 'nvidia', apiKey: 'short' }),
    /hosted_visual_critic_not_configured/);
  assert.throws(() => createHostedVisualCritic({ provider: 'nvidia-nope', apiKey: 'test-nvidia-key-0123456789abcdef' }),
    /unsupported_visual_critic_provider/);
});
