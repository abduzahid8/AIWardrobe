/** @jest-environment node */
import { describeGarment, prepareGarment } from '../../src/domain/garmentDescriptor';
import {
  visualAdmissionReason,
  currentVisual,
  type VisualFashionProfile,
} from '../../src/domain/visualFashion';
import { conceptGarmentAffinity, outfitConcepts } from '../../src/domain/outfitConcepts';
import { stylingContext } from '../../src/domain/stylingContext';
import { createOutfitPlan } from '../../src/domain/validatedQuartets';
import { productIdentityKeys } from '../../src/domain/outfitDiversity';
import {
  validateLookRanking,
  rankNvidiaLooks,
  nvidiaRankingBody,
} from '../../backend/supabase/functions/_shared/nvidiaLookRanking';
const observation = (value: string, confidence = 0.95) => ({
  value,
  confidence,
  margin: 0.2,
  evidence: 'image-model' as const,
});
const image = 'https://catalog.example/product.jpg',
  hash = 'a'.repeat(32);
const profile = (attributes: VisualFashionProfile['attributes']): VisualFashionProfile => ({
  version: 1,
  sourceHash: hash,
  imageUrl: image,
  imageSha256: 'b'.repeat(64),
  model: 'siglip2',
  revision: 'c'.repeat(40),
  attributes,
});
it('uses current confident image evidence while preserving unknowns and source slot authority', () => {
  const item = {
    id: 'real-shirt',
    name: 'Oxford shirt',
    macroCategory: 'top',
    imageUrl: image,
    visualSourceHash: hash,
    visualFashion: profile({
      subtype: observation('dress-shirt'),
      color: observation('navy'),
      texture: observation('tweed', 0.4),
    }),
  };
  const d = describeGarment(prepareGarment(item));
  expect(d.subtype).toBe('dress-shirt');
  expect(d.color).toBe('navy');
  expect(d.texture).not.toBe('tweed');
  expect(describeGarment({ ...item, visualSourceHash: 'd'.repeat(32) }).color).not.toBe('navy');
  expect(currentVisual({ ...item, imageUrl: image + 'changed' })).toBeUndefined();
  expect(
    describeGarment({ ...item, visualFashion: profile({ subtype: observation('oxford') }) }).subtype
  ).toBe('dress-shirt');
  expect(
    visualAdmissionReason({ ...item, visualFashion: profile({ slot: observation('shoes') }) })
  ).toBe('visual_source_category_conflict');
});
it('keeps base-versus-layer role ambiguity from discarding a real knit', () => {
  expect(
    visualAdmissionReason({
      macroCategory: 'outerwear',
      imageUrl: image,
      visualSourceHash: hash,
      visualFashion: profile({ slot: observation('top'), color: observation('brown') }),
    })
  ).toBeNull();
});
it('quarantines diagrams only when verified and retains provenance', () => {
  const p = {
    ...profile({}),
    imageKind: { ...observation('diagram'), evidence: 'image-ocr' as const },
  };
  expect(visualAdmissionReason({ imageUrl: image, visualSourceHash: hash, visualFashion: p })).toBe(
    'visual_image_unusable'
  );
  expect(
    visualAdmissionReason({
      imageUrl: image,
      visualSourceHash: hash,
      visualFashion: {
        ...p,
        imageKind: { ...observation('diagram', 0.6), evidence: 'image-ocr' as const },
      },
    })
  ).toBeNull();
});
it('defines different work, date and travel construction identities', () => {
  const concepts = (style: string, occasion: string) =>
    outfitConcepts(stylingContext({ style, occasion }));
  expect(concepts('old_money', 'work').map((c) => c.id)).toEqual([
    'refined-ivy',
    'textured-tailoring',
    'smart-knit-layering',
  ]);
  expect(concepts('classic', 'work').map((c) => c.id)).not.toEqual(
    concepts('semi_classic', 'date night').map((c) => c.id)
  );
  expect(concepts('casual', 'travel').every((c) => c.silhouette === 'relaxed')).toBe(true);
});
it('validates exact complete-look rankings without legacy garment parsing', () => {
  const allowed = ['look247', 'look009', 'look121'];
  expect(
    validateLookRanking('{"rankedLookIds":["look121","look247","look009"]}', allowed).ids
  ).toEqual(['look121', 'look247', 'look009']);
  for (const raw of [
    '{"rankedLookIds":["look001","look002"]}',
    '{"rankedLookIds":["look247","look247","look121"]}',
    '{"outfits":[{"items":[{"id":"shirt"}]}]}',
    '{"rankedLookIds":["look247"],"items":[]}',
  ])
    expect(validateLookRanking(raw, allowed).ids).toBeNull();
  const body = nvidiaRankingBody('nvidia/nemotron-3.5-lightning-30b-a3b', 'Rank', {}, allowed);
  expect(body.response_format).toEqual({ type: 'json_object' });
  expect(body.chat_template_kwargs).toEqual({ enable_thinking: false });
  expect(body.messages[0].content).toContain(JSON.stringify(allowed));
  expect(body.messages[0].content).not.toContain('look001');
});
it('makes one provider call and rejects unknown products without retrying', async () => {
  const fetcher = jest.fn(
    async () =>
      new Response(
        JSON.stringify({
          choices: [
            { finish_reason: 'stop', message: { content: '{"rankedLookIds":["invented"]}' } },
          ],
        })
      )
  );
  const result = await rankNvidiaLooks('secret', 'test', 'Rank', {}, ['look247'], fetcher);
  expect(fetcher).toHaveBeenCalledTimes(1);
  expect(result.ids).toBeNull();
  expect(result.attempt).toMatchObject({
    status: 'rank_contract_mismatch',
    unknown: 1,
    expected: 1,
    received: 1,
  });
  expect(JSON.stringify(result)).not.toContain('secret');
  expect(JSON.stringify(result)).not.toContain('invented');
});
it.each(['headers', 'body'])(
  'bounds a stalled provider %s even when transport ignores abort',
  async (stage) => {
    jest.useFakeTimers();
    try {
      const never = () => new Promise<any>(() => {});
      const fetcher = jest.fn(
        stage === 'headers' ? never : async () => ({ ok: true, json: never }) as any
      );
      const pending = rankNvidiaLooks('secret', 'test', 'Rank', {}, ['look247'], fetcher);
      await jest.advanceTimersByTimeAsync(12001);
      expect((await pending).attempt.status).toBe('timeout');
      expect(fetcher).toHaveBeenCalledTimes(1);
      expect(jest.getTimerCount()).toBe(0);
    } finally {
      jest.useRealTimers();
    }
  }
);
it('maintains physical uniqueness, anchors and caller budget validation with a bounded planner', () => {
  const names = [
    'Blue cotton shirt',
    'Navy cardigan',
    'Grey tailored trousers',
    'Brown leather loafers',
  ];
  const items = names.flatMap((name, s) =>
    Array.from({ length: 5 }, (_, i) => ({
      id: `${s}-${i}`,
      name: name + ' version ' + i,
      macroCategory: ['top', 'outerwear', 'bottom', 'shoes'][s],
      price: i + 1,
    }))
  );
  const context = stylingContext({ style: 'old_money', occasion: 'work' });
  let calls = 0;
  const plan = createOutfitPlan(
    items,
    context,
    (parts) => {
      calls++;
      return { valid: parts.reduce((sum, p) => sum + p.price, 0) <= 14, score: 0.9 };
    },
    {
      requiredIds: ['0-0'],
      target: 3,
      priority: () => [],
      cost: (i) => i.price,
      budget: 14,
      maxEvaluations: 64,
    }
  );
  expect(plan.seed).toHaveLength(1);
  expect(calls).toBeLessThanOrEqual(64);
  const keys = new Set<string>();
  for (const look of plan.seed) {
    expect(look.items.some((i) => i.id === '0-0')).toBe(true);
    expect(look.items.reduce((sum, i) => sum + i.price, 0)).toBeLessThanOrEqual(14);
    for (const item of look.items.filter((i) => i.id !== '0-0'))
      for (const key of productIdentityKeys(item)) {
        expect(keys.has(key)).toBe(false);
        keys.add(key);
      }
  }
});

it('uses cached color families and visual hues in concept palette scoring', () => {
  const base = describeGarment({ name: 'Navy cotton shirt', macroCategory: 'top' });
  const concept = outfitConcepts(stylingContext({ style: 'old_money', occasion: 'work' }))[0];
  for (const d of [base, { ...base, color: 'navy' }]) {
    expect(conceptGarmentAffinity(d, { ...concept, palette: ['navy'] })).toBeGreaterThan(
      conceptGarmentAffinity(d, { ...concept, palette: ['pink'] })
    );
  }
});
