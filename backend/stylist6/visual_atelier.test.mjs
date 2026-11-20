import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  VISUAL_ATELIER_VERSION,
  selectShowroomSlotPools,
  validateVisualAtelierOutfits,
  composeOutfitsVisually,
  selectVisualSlotReplacement,
} from './visual_atelier.mjs';
import { createPhotoByteCache, renderEditorialLookBoard } from './editorial_board_renderer.mjs';
import {
  createProviderRouter,
  PROVIDER_ENDPOINTS,
  DEFAULT_OPENROUTER_VISION_MODEL,
  DEFAULT_NVIDIA_VISION_MODEL,
} from './ai_provider_router.mjs';

async function makeJpeg(hex = '#4A5D73', w = 160, h = 220) {
  return sharp({ create: { width: w, height: h, channels: 3, background: hex } })
    .jpeg({ quality: 75 })
    .toBuffer();
}

function buildTestDirectionPools(photoBytes) {
  const slots = ['top', 'outerwear', 'bottom', 'shoes'];
  const subtypesByDir = [
    { top: 'dress-shirt', outerwear: 'blazer', bottom: 'tailored-trouser', shoes: 'oxford' },
    { top: 'fine-knit', outerwear: 'cardigan', bottom: 'chino', shoes: 'loafer' },
    { top: 'polo', outerwear: 'overshirt', bottom: 'trouser', shoes: 'derby' },
  ];
  const colorsByDir = [
    ['white', 'navy', 'charcoal', 'black'],
    ['ivory', 'camel', 'beige', 'brown'],
    ['sky', 'olive', 'stone', 'tan'],
  ];
  const brands = ['Canali', 'Eton', 'Incotex', 'Velasca', 'COS', 'Suitsupply', 'Massimo Dutti', 'Carmina', 'Reiss'];

  const directions = [0, 1, 2].map((dIdx) => ({
    id: `ai-director-${dIdx + 1}`,
    directionIndex: dIdx,
    name: ['Sharp Navy Tailoring', 'Warm Cashmere Ease', 'Textural Olive Smart'][dIdx],
    directionType: ['signature_mastery', 'silhouette_texture_shift', 'editorial_statement'][dIdx],
    artDirection: `Direction ${dIdx + 1} balances structured shoulders, breathable weave, and sleek footwear.`,
    silhouette: ['tailored', 'relaxed', 'balanced'][dIdx],
    palette: colorsByDir[dIdx],
    roles: Object.fromEntries(slots.map((s) => [s, [subtypesByDir[dIdx][s]]])),
  }));

  const dirPools = directions.map((dir, dIdx) => {
    const slotsObj = {};
    for (let sIdx = 0; sIdx < slots.length; sIdx++) {
      const slot = slots[sIdx];
      slotsObj[slot] = [0, 1, 2].map((rank) => {
        const id = `SECRET_SKU_D${dIdx}_${slot}_${rank}`;
        return {
          token: `P_${dIdx}_${sIdx}_${rank}`,
          slot,
          item: {
            id,
            macroCategory: slot,
            isShopItem: true,
            subtype: subtypesByDir[dIdx][slot],
            color: rank === 0 ? colorsByDir[dIdx][sIdx] : ['cream', 'grey', 'olive'][rank],
            brand: brands[(dIdx * 3 + rank) % brands.length],
            name: `${colorsByDir[dIdx][sIdx]} ${subtypesByDir[dIdx][slot]} D${dIdx} R${rank}`,
            price: 110 + dIdx * 30 + rank * 25,
            currency: 'USD',
            imageUrl: `https://static.zara.net/atelier-${dIdx}-${slot}-${rank}.jpg`,
            shopUrl: `https://www.zara.com/us/product/${id}`,
            bytes: photoBytes,
          },
        };
      });
    }
    return {
      directionIndex: dIdx,
      directionName: dir.name,
      slots: slotsObj,
      pool: Object.values(slotsObj).flat(),
    };
  });

  return {
    directions,
    perDirectionPools: {
      directions: dirPools,
      unifiedPool: dirPools.flatMap((d) => d.pool),
    },
  };
}

test('selectShowroomSlotPools draws up to 8 diverse budget-feasible candidates evenly across 3 orthogonal directions', async () => {
  assert.equal(VISUAL_ATELIER_VERSION, 's6-visual-atelier-v1');
  const jpg = await makeJpeg('#3D4E60');
  const { directions, perDirectionPools } = buildTestDirectionPools(jpg);

  // Inject an avoided color, quarantined item, and over-budget luxury coat into Direction 0
  perDirectionPools.directions[0].slots.outerwear.unshift(
    {
      slot: 'outerwear',
      item: {
        id: 'blocked-red-blazer',
        macroCategory: 'outerwear',
        subtype: 'blazer',
        color: 'red',
        brand: 'LoudHouse',
        price: 180,
        currency: 'USD',
        bytes: jpg,
      },
    },
    {
      slot: 'outerwear',
      item: {
        id: 'overbudget-vicuna-coat',
        macroCategory: 'outerwear',
        subtype: 'coat',
        color: 'camel',
        brand: 'UltraLux',
        price: 4500,
        currency: 'USD',
        bytes: jpg,
      },
    },
    {
      slot: 'outerwear',
      item: {
        id: 'quarantined-blazer',
        macroCategory: 'outerwear',
        subtype: 'blazer',
        color: 'navy',
        brand: 'BadPhoto',
        price: 200,
        currency: 'USD',
        quarantined: true,
        bytes: jpg,
      },
    },
  );

  const selected = selectShowroomSlotPools({
    directions,
    perDirectionPools,
    context: { budget: { amount: 1000, currency: 'USD' }, preferences: { avoidColors: ['red'] } },
    maxPerSlot: 8,
  });

  for (const slot of ['top', 'outerwear', 'bottom', 'shoes']) {
    assert.ok(selected.slotPools[slot].length >= 6 && selected.slotPools[slot].length <= 8);
    // First 3 items in each rack should draw 1 from each of the 3 orthogonal directions!
    const firstThreeIds = selected.slotPools[slot].slice(0, 3).map((it) => it.id);
    assert.ok(firstThreeIds.some((id) => id.includes('_D0_')), `${slot} must include Direction 0`);
    assert.ok(firstThreeIds.some((id) => id.includes('_D1_')), `${slot} must include Direction 1`);
    assert.ok(firstThreeIds.some((id) => id.includes('_D2_')), `${slot} must include Direction 2`);
  }

  const outerIds = selected.slotPools.outerwear.map((it) => it.id);
  assert.ok(!outerIds.includes('blocked-red-blazer'), 'avoidColors excluded');
  assert.ok(!outerIds.includes('overbudget-vicuna-coat'), 'over-budget slot outlier excluded');
  assert.ok(!outerIds.includes('quarantined-blazer'), 'quarantined item excluded');
});

test('composeOutfitsVisually renders 1600x1600 showroom board, sends JPEG + Style DNA to VLM, and maps exact authentic SKUs without leaking PII/URLs', async () => {
  const jpg = await makeJpeg('#526377');
  const { directions, perDirectionPools } = buildTestDirectionPools(jpg);
  let capturedVisionReq = null;

  const mockRouter = {
    isProviderConfigured: (p) => p === 'openrouter' || p === 'nvidia',
    callVisionModel: async (req) => {
      capturedVisionReq = req;
      return {
        ok: true,
        provider: 'openrouter',
        model: DEFAULT_OPENROUTER_VISION_MODEL,
        failedOver: false,
        attempts: [{ provider: 'openrouter', model: DEFAULT_OPENROUTER_VISION_MODEL, ok: true }],
        parsedJson: {
          outfits: [
            {
              directionIndex: 0,
              title: 'Sharp Milanese Navy Tailoring',
              top: 'T1',
              outerwear: 'L1',
              bottom: 'B1',
              shoes: 'S1',
              visualHarmonyRationale: 'Crisp white dress shirt illuminates the deep navy blazer above charcoal tailored wool trousers and sleek black oxfords.',
              stylingNote: 'A refined 3-value boardroom ensemble built on clean drape and tonal contrast.',
              whyItFlattersYou: 'High-contrast cool navy and white directly flatter your Cool Winter undertone.',
              stylingTips: ['Leave the top button open for relaxed polish after work', 'Match a matte black calfskin belt to the oxfords'],
            },
            {
              directionIndex: 1,
              title: 'Warm Cashmere Studio Ease',
              top: 'T2',
              outerwear: 'L2',
              bottom: 'B2',
              shoes: 'S2',
              visualHarmonyRationale: 'Ivory merino base bridges the warm camel cardigan with sand chinos and dark brown suede loafers.',
              stylingNote: 'Soft tactile layering for creative meetings without rigid suiting.',
              whyItFlattersYou: 'Elongated tonal flow keeps the silhouette clean and modern.',
              stylingTips: ['Half-tuck the fine knit to highlight trouser drape', 'Wear invisible no-show socks with the suede loafers'],
            },
            {
              directionIndex: 2,
              title: 'Textural Olive Gallery Smart',
              top: 'T3',
              outerwear: 'L3',
              bottom: 'B3',
              shoes: 'S3',
              visualHarmonyRationale: 'Sky blue knit polo brightens the structured olive overshirt paired with stone trousers and burnished tan derbies.',
              stylingNote: 'Contemporary utility-tailored balance for smart-casual evenings.',
              whyItFlattersYou: 'Structured overshirt shoulder frames the torso cleanly.',
              stylingTips: ['Roll cuffs once to expose a subtle watch', 'Keep overshirt unbuttoned to show vertical color contrast'],
            },
          ],
        },
      };
    },
  };

  const result = await composeOutfitsVisually({
    directions,
    perDirectionPools,
    context: { style: 'old_money', occasion: 'Work', weather: { temp: 16 }, budget: { amount: 1500, currency: 'USD' } },
    styleDnaPromptBlock: 'Cool Winter high contrast; prefers sharp tailored lines; avoid bright red.',
    tastePromptBlock: 'User prefers refined suede loafers and unstructured blazers.',
    providerRouter: mockRouter,
  });

  assert.equal(result.status, 'applied');
  assert.equal(result.provider, 'openrouter');
  assert.equal(result.model, DEFAULT_OPENROUTER_VISION_MODEL);
  assert.equal(result.outfits.length, 3);
  assert.ok(result.showroomBoard?.ok);
  assert.equal(result.showroomBoard.width, 1600);
  assert.equal(result.showroomBoard.height, 1600);
  assert.ok(Buffer.isBuffer(result.showroomBoard.jpegBuffer));

  // Verify VLM received the actual 1600x1600 board JPEG + Style DNA + Directions, and ZERO raw SKU IDs / URLs!
  assert.ok(capturedVisionReq);
  assert.equal(capturedVisionReq.images.length, 1);
  assert.equal(capturedVisionReq.images[0].bytes.equals(result.showroomBoard.jpegBuffer), true);
  assert.match(capturedVisionReq.prompt, /Cool Winter high contrast/);
  assert.match(capturedVisionReq.prompt, /Sharp Navy Tailoring/);
  assert.doesNotMatch(capturedVisionReq.prompt, /SECRET_SKU_|static\.zara\.net|www\.zara\.com/);

  // Verify strict SKU identity mapping & zero shared SKUs across looks
  const allItemIds = result.outfits.flatMap((o) => o.items.map((it) => it.id));
  assert.equal(allItemIds.length, 12);
  assert.equal(new Set(allItemIds).size, 12, 'all 12 SKUs across 3 outfits must be strictly disjoint');
  assert.equal(result.outfits[0].top.id, result.showroomBoard.tokens.T1.id);
  assert.equal(result.outfits[1].outerwear.id, result.showroomBoard.tokens.L2.id);
  assert.equal(result.outfits[2].shoes.id, result.showroomBoard.tokens.S3.id);
  assert.equal(result.outfits[0].composedVisually, true);
  assert.match(result.outfits[0].whyItFlattersYou, /Cool Winter/);
  assert.equal(result.outfits[0].stylingTips.length, 2);
});

test('validateVisualAtelierOutfits & composeOutfitsVisually strictly reject hallucinated, wrong-slot, or duplicate tokens', async () => {
  const jpg = await makeJpeg('#455565');
  const { directions, perDirectionPools } = buildTestDirectionPools(jpg);

  // Hallucinated token T99
  const hallucinated = await composeOutfitsVisually({
    directions,
    perDirectionPools,
    context: { style: 'classic', occasion: 'Work' },
    providerRouter: {
      isProviderConfigured: () => true,
      callVisionModel: async () => ({
        ok: true,
        provider: 'openrouter',
        model: DEFAULT_OPENROUTER_VISION_MODEL,
        parsedJson: {
          outfits: [
            { directionIndex: 0, title: 'A', top: 'T99', outerwear: 'L1', bottom: 'B1', shoes: 'S1', visualHarmonyRationale: 'x', stylingNote: 'x', whyItFlattersYou: 'x', stylingTips: ['tip'] },
            { directionIndex: 1, title: 'B', top: 'T2', outerwear: 'L2', bottom: 'B2', shoes: 'S2', visualHarmonyRationale: 'x', stylingNote: 'x', whyItFlattersYou: 'x', stylingTips: ['tip'] },
            { directionIndex: 2, title: 'C', top: 'T3', outerwear: 'L3', bottom: 'B3', shoes: 'S3', visualHarmonyRationale: 'x', stylingNote: 'x', whyItFlattersYou: 'x', stylingTips: ['tip'] },
          ],
        },
      }),
    },
  });
  assert.equal(hallucinated.status, 'fallback');
  assert.equal(hallucinated.outfits.length, 0);
  assert.ok(hallucinated.diagnostics.rejectedReasons.some((r) => r.includes('hallucinated_token_T99')));

  // Duplicate token across outfits (S1 reused in outfit 0 & outfit 1)
  const dupCheck = validateVisualAtelierOutfits(
    {
      outfits: [
        { directionIndex: 0, title: 'A', top: 'T1', outerwear: 'L1', bottom: 'B1', shoes: 'S1' },
        { directionIndex: 1, title: 'B', top: 'T2', outerwear: 'L2', bottom: 'B2', shoes: 'S1' },
        { directionIndex: 2, title: 'C', top: 'T3', outerwear: 'L3', bottom: 'B3', shoes: 'S3' },
      ],
    },
    hallucinated.showroomBoard.tokens,
    { targetCount: 3, minValidOutfits: 3 },
  );
  assert.equal(dupCheck.ok, false);
  assert.ok(dupCheck.rejectedReasons.some((r) => r.includes('duplicate_token_S1')));

  // Wrong-slot token (L1 placed into top slot)
  const wrongSlotCheck = validateVisualAtelierOutfits(
    {
      outfits: [
        { directionIndex: 0, title: 'A', top: 'L1', outerwear: 'L2', bottom: 'B1', shoes: 'S1' },
      ],
    },
    hallucinated.showroomBoard.tokens,
    { targetCount: 1, minValidOutfits: 1 },
  );
  assert.equal(wrongSlotCheck.ok, false);
  assert.ok(wrongSlotCheck.rejectedReasons.some((r) => r.includes('invalid_or_wrong_slot_token_top')));
});

test('composeOutfitsVisually automatically fails over from OpenRouter 503 to NVIDIA 120B VL via ai_provider_router', async () => {
  const jpg = await makeJpeg('#6B5A4C');
  const { directions, perDirectionPools } = buildTestDirectionPools(jpg);
  const endpointsCalled = [];

  const router = createProviderRouter({
    openrouterApiKey: 'sk-or-v1-test-visual-atelier-0123456',
    nvidiaApiKey: 'nvapi-test-visual-atelier-0123456',
    fetchImpl: async (url, init) => {
      const body = JSON.parse(init.body);
      endpointsCalled.push({ url: String(url), model: body.model });
      if (url === PROVIDER_ENDPOINTS.openrouter) {
        return new Response('openrouter busy', { status: 503 });
      }
      assert.equal(url, PROVIDER_ENDPOINTS.nvidia);
      assert.equal(body.model, DEFAULT_NVIDIA_VISION_MODEL);
      return Response.json({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              outfits: [
                { directionIndex: 0, title: 'Look 1', top: 'T1', outerwear: 'L1', bottom: 'B1', shoes: 'S1', visualHarmonyRationale: 'Balanced tones', stylingNote: 'Clean tailoring', whyItFlattersYou: 'Sharp shoulder', stylingTips: ['Tip 1'] },
                { directionIndex: 1, title: 'Look 2', top: 'T2', outerwear: 'L2', bottom: 'B2', shoes: 'S2', visualHarmonyRationale: 'Warm drape', stylingNote: 'Relaxed knit', whyItFlattersYou: 'Soft palette', stylingTips: ['Tip 2'] },
                { directionIndex: 2, title: 'Look 3', top: 'T3', outerwear: 'L3', bottom: 'B3', shoes: 'S3', visualHarmonyRationale: 'Crisp contrast', stylingNote: 'Smart overshirt', whyItFlattersYou: 'Balanced frame', stylingTips: ['Tip 3'] },
              ],
            }),
          },
        }],
      });
    },
  });

  const res = await composeOutfitsVisually({
    directions,
    perDirectionPools,
    context: { style: 'classic', occasion: 'Date night', weather: { temp: 18 } },
    providerRouter: router,
  });

  assert.equal(res.status, 'applied');
  assert.equal(res.provider, 'nvidia');
  assert.equal(res.model, DEFAULT_NVIDIA_VISION_MODEL);
  assert.equal(res.diagnostics.failedOver, true);
  assert.ok(endpointsCalled.some((e) => e.url === PROVIDER_ENDPOINTS.openrouter));
  assert.ok(endpointsCalled.some((e) => e.url === PROVIDER_ENDPOINTS.nvidia));
  assert.equal(res.outfits.length, 3);
});

test('selectVisualSlotReplacement renders 1400x900 Swap Board (reusing photoCache for locked items) and returns visually selected R-token SKU', async () => {
  const lockedBuf = await makeJpeg('#2B3648', 160, 220);
  const candBuf = await makeJpeg('#5B3A29', 180, 140);
  const cache = createPhotoByteCache({ maxEntries: 32 });
  const fetches = [];
  const fetchImpl = async (url) => {
    fetches.push(url);
    return new Response(url.includes('cand') ? candBuf : lockedBuf, { status: 200 });
  };

  const look = {
    id: 'look_swap_test',
    items: [
      { id: 'lock-coat', macroCategory: 'outerwear', subtype: 'blazer', brand: 'Canali', price: 540, currency: 'USD', imageUrl: 'https://static.zara.net/sw-coat.jpg' },
      { id: 'lock-top', macroCategory: 'top', subtype: 'fine-knit', brand: 'COS', price: 110, currency: 'USD', imageUrl: 'https://static.zara.net/sw-top.jpg' },
      { id: 'lock-bottom', macroCategory: 'bottom', subtype: 'tailored-trouser', brand: 'Incotex', price: 210, currency: 'USD', imageUrl: 'https://static.zara.net/sw-bottom.jpg' },
      { id: 'weak-shoe', macroCategory: 'shoes', subtype: 'sport-sneaker', brand: 'RunCo', price: 130, currency: 'USD', imageUrl: 'https://static.zara.net/sw-old-shoe.jpg' },
    ],
  };

  // Pre-warm cache for the 3 locked pieces
  await renderEditorialLookBoard(look.items.slice(0, 3), { photoCache: cache, fetchImpl });
  assert.equal(fetches.length, 3);

  const candidates = [
    { id: 'shoe-r1', macroCategory: 'shoes', subtype: 'derby', brand: 'Berwick', price: 220, currency: 'USD', imageUrl: 'https://static.zara.net/cand-r1.jpg' },
    { id: 'shoe-r2-suede-loafer', macroCategory: 'shoes', subtype: 'loafer', brand: 'Velasca', price: 260, currency: 'USD', imageUrl: 'https://static.zara.net/cand-r2.jpg' },
    { id: 'shoe-r3', macroCategory: 'shoes', subtype: 'oxford', brand: 'Carmina', price: 390, currency: 'USD', imageUrl: 'https://static.zara.net/cand-r3.jpg' },
  ];

  let sentSwapImage = null;
  const res = await selectVisualSlotReplacement({
    look,
    targetSlot: 'shoes',
    repairDirective: 'Replace chunky sneaker with dark brown suede loafer to harmonize with tailored trousers',
    candidates,
    context: { occasion: 'Work', style: 'classic', weather: { temp: 17 } },
    styleDnaPromptBlock: 'Warm autumn undertone, refined tailored silhouette.',
    photoCache: cache,
    fetchImpl,
    providerRouter: {
      isProviderConfigured: () => true,
      callVisionModel: async (req) => {
        sentSwapImage = req.images[0].bytes;
        assert.match(req.prompt, /dark brown suede loafer/i);
        assert.doesNotMatch(req.prompt, /shoe-r2-suede-loafer|static\.zara\.net/);
        return {
          ok: true,
          provider: 'openrouter',
          model: DEFAULT_OPENROUTER_VISION_MODEL,
          parsedJson: {
            selectedToken: 'R2',
            visualRationale: 'R2 sleek dark brown suede loafer grounds the blazer and trousers without looking overly stiff.',
            confidence: 0.94,
          },
        };
      },
    },
  });

  assert.equal(res.status, 'selected');
  assert.equal(res.token, 'R2');
  assert.equal(res.selectedItem.id, 'shoe-r2-suede-loafer');
  assert.match(res.rationale, /suede loafer/i);
  assert.equal(res.swapBoard.width, 1400);
  assert.equal(res.swapBoard.height, 900);
  assert.equal(res.swapBoard.cacheHits, 3, '3 locked items reused from photoCache');
  assert.equal(fetches.length, 6, 'only 3 candidate shoe photos fetched');
  assert.ok(sentSwapImage.equals(res.swapBoard.jpegBuffer));

  // Verify hallucinated R9 returns honest fallback
  const badRes = await selectVisualSlotReplacement({
    look,
    targetSlot: 'shoes',
    repairDirective: 'Replace shoes',
    candidates,
    photoCache: cache,
    fetchImpl,
    providerRouter: {
      isProviderConfigured: () => true,
      callVisionModel: async () => ({
        ok: true,
        provider: 'openrouter',
        model: DEFAULT_OPENROUTER_VISION_MODEL,
        parsedJson: { selectedToken: 'R9', visualRationale: 'hallucinated', confidence: 0.5 },
      }),
    },
  });
  assert.equal(badRes.status, 'fallback');
  assert.equal(badRes.selectedItem, null);
  assert.equal(badRes.reason, 'invalid_selected_token');
});

test('Phase 10 Defect B: selectShowroomSlotPools excludes logo sweatshirts, sweatpants, jersey Henleys, and non-black-tie tuxedo jackets while building deeper backfill queue', () => {
  const catalog = [
    { id: 'ck-logo-sweat', macroCategory: 'top', subtype: 'sweatshirt', color: 'white', brand: 'Calvin Klein', name: 'CK Logo Crewneck Sweatshirt', price: 140, currency: 'USD', imageUrl: 'https://static.zara.net/ck.jpg' },
    { id: 'rl-jersey-henley', macroCategory: 'top', subtype: 'shirt', color: 'grey', brand: 'Ralph Lauren', name: 'Cotton Jersey Waffle Henley', price: 95, currency: 'USD', imageUrl: 'https://static.zara.net/henley.jpg' },
    { id: 'clean-merino', macroCategory: 'top', subtype: 'fine-knit', color: 'cream', brand: 'Canali', name: 'Fine Merino Crewneck Sweater', price: 240, currency: 'USD', imageUrl: 'https://static.zara.net/merino.jpg' },
    { id: 'blocked-host-shirt', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', brand: 'Barbour', name: 'White Poplin Shirt', price: 130, currency: 'USD', imageUrl: 'https://www.barbour.com/shirt.jpg' },
    { id: 'tux-jacket', macroCategory: 'outerwear', subtype: 'blazer', color: 'black', brand: 'J.Crew', name: 'Ludlow Satin Shawl Tuxedo Dinner Jacket', price: 698, currency: 'USD', imageUrl: 'https://static.zara.net/tux.jpg' },
    { id: 'navy-blazer', macroCategory: 'outerwear', subtype: 'blazer', color: 'navy', brand: 'Aurélien', name: 'Navy Tailored Wool Blazer', price: 495, currency: 'USD', imageUrl: 'https://static.zara.net/blazer.jpg' },
    { id: 'allsaints-sweatpant', macroCategory: 'bottom', subtype: 'athletic-trouser', color: 'navy', brand: 'AllSaints', name: 'Elastic Drawstring Lounge Pants', price: 189, currency: 'USD', imageUrl: 'https://static.zara.net/sweatpant.jpg' },
    { id: 'wool-trouser', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', brand: 'J.Crew', name: 'Tailored Fit Wool Suit Trousers', price: 220, currency: 'USD', imageUrl: 'https://static.zara.net/trouser.jpg' },
    { id: 'espresso-loafer', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', brand: 'Beckett Simonon', name: 'Espresso Calf Loafer', price: 249, currency: 'USD', imageUrl: 'https://static.zara.net/loafer.jpg' },
  ];

  const { slotPools } = selectShowroomSlotPools({
    catalog,
    context: { style: 'Old Money', occasion: 'Work' },
    authorizePhoto: (url) => url.startsWith('https://static.zara.net/'),
    maxPerSlot: 8,
    queuePerSlot: 16,
  });

  const allIds = Object.values(slotPools).flat().map((it) => it.id);
  assert.ok(!allIds.includes('ck-logo-sweat'), 'CK logo sweatshirt excluded from Old Money Work');
  assert.ok(!allIds.includes('rl-jersey-henley'), 'casual jersey Henley excluded from Old Money Work');
  assert.ok(!allIds.includes('blocked-host-shirt'), 'non-allowlisted host excluded before showroom fetch');
  assert.ok(!allIds.includes('tux-jacket'), 'satin shawl tuxedo dinner jacket excluded from Work');
  assert.ok(!allIds.includes('allsaints-sweatpant'), 'drawstring sweatpants excluded from Work');
  assert.ok(allIds.includes('clean-merino'));
  assert.ok(allIds.includes('navy-blazer'));
  assert.ok(allIds.includes('wool-trouser'));
  assert.ok(allIds.includes('espresso-loafer'));
});
