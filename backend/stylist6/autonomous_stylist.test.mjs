import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  AUTONOMOUS_STYLIST_VERSION,
  filterAutonomousHardGates,
  curateShowroomSelection,
  detectLayeringOrOccasionDefect,
  buildAct2EditorPrompt,
  verifyAutonomousOutfits,
  runAutonomousVisualStylist,
} from './autonomous_stylist.mjs';
import { createPhotoByteCache } from './editorial_board_renderer.mjs';

async function makeTestJpeg(r, g, b) {
  return sharp({
    create: { width: 220, height: 280, channels: 3, background: { r, g, b } },
  }).jpeg({ quality: 88 }).toBuffer();
}

function makeCatalogItem(id, slot, subtype, name, {
  color = 'navy',
  brand = 'Aurelien',
  price = 220,
  currency = 'USD',
  imageUrl = `https://media.frgroup.uz/images/${id}.jpg`,
  shopUrl = `https://aurelien-online.com/products/${id}`,
  isShopItem = true,
} = {}) {
  return {
    id,
    macroCategory: slot,
    subtype,
    fashion_subtype: subtype,
    name,
    color,
    brand,
    price,
    currency,
    imageUrl,
    shopUrl,
    isShopItem,
  };
}

function makeRichCatalog() {
  return [
    // Tops (5)
    makeCatalogItem('top-ivory-knit', 'top', 'fine-knit', 'Ivory Merino Turtleneck', { color: 'cream', brand: 'Loro Piana', price: 240 }),
    makeCatalogItem('top-white-shirt', 'top', 'dress-shirt', 'Crisp White Poplin Dress Shirt', { color: 'white', brand: 'Canali', price: 180 }),
    makeCatalogItem('top-navy-polo', 'top', 'knit-polo', 'Midnight Merino Knit Polo', { color: 'navy', brand: 'Aurelien', price: 165 }),
    makeCatalogItem('top-charcoal-knit', 'top', 'fine-knit', 'Charcoal Cashmere Crewneck', { color: 'charcoal', brand: 'Massimo Dutti', price: 195 }),
    makeCatalogItem('top-blue-stripe', 'top', 'shirt', 'Striped Blue Office Oxford Shirt', { color: 'blue', brand: 'Selected Homme', price: 95 }),
    // Outerwear (6 — includes workwear overshirt + heavy winter puffer for safety/curation tests)
    makeCatalogItem('outer-arket-chore', 'outerwear', 'overshirt', 'Overdyed Twill Workwear Overshirt - Green', { color: 'green', brand: 'ARKET', price: 65 }),
    makeCatalogItem('outer-navy-blazer', 'outerwear', 'blazer', 'Midnight Unstructured Wool Blazer', { color: 'navy', brand: 'Canali', price: 420 }),
    makeCatalogItem('outer-espresso-suede', 'outerwear', 'jacket', 'Espresso Suede Blouson Jacket', { color: 'brown', brand: 'Aurelien', price: 450 }),
    makeCatalogItem('outer-charcoal-cardigan', 'outerwear', 'cardigan', 'Charcoal Merino Tailored Cardigan', { color: 'charcoal', brand: 'Massimo Dutti', price: 260 }),
    makeCatalogItem('outer-camel-blazer', 'outerwear', 'blazer', 'Camel Soft Tailored Wool Blazer', { color: 'camel', brand: 'Giorgio Armani', price: 440 }),
    makeCatalogItem('outer-winter-puffer', 'outerwear', 'coat', 'Heavy Down Winter Puffer Parka', { color: 'black', brand: 'NorthPeak', price: 390 }),
    // Bottoms (5)
    makeCatalogItem('bot-charcoal-trouser', 'bottom', 'tailored-trouser', 'Charcoal Pleated Flannel Trousers', { color: 'charcoal', brand: 'Canali', price: 230 }),
    makeCatalogItem('bot-navy-trouser', 'bottom', 'tailored-trouser', 'Navy Tailored Wool Trousers', { color: 'navy', brand: 'Massimo Dutti', price: 190 }),
    makeCatalogItem('bot-stone-trouser', 'bottom', 'tailored-trouser', 'Stone Pleated Wool Trousers', { color: 'beige', brand: 'Aurelien', price: 210 }),
    makeCatalogItem('bot-olive-trouser', 'bottom', 'trouser', 'Dark Olive Tailored Trousers', { color: 'olive', brand: 'COS', price: 160 }),
    makeCatalogItem('bot-swim-trunks', 'bottom', 'shorts', 'Navy Swim Shorts Trunks', { color: 'navy', brand: 'ResortCo', price: 85 }),
    // Shoes (5)
    makeCatalogItem('shoe-navy-loafer', 'shoes', 'loafer', 'Navy Suede Penny Loafers', { color: 'navy', brand: 'Aurelien', price: 260 }),
    makeCatalogItem('shoe-espresso-chelsea', 'shoes', 'chelsea', 'Espresso Suede Chelsea Boots', { color: 'brown', brand: 'Beckett Simonon', price: 279 }),
    makeCatalogItem('shoe-black-loafer', 'shoes', 'loafer', 'Black Calfskin Penny Loafers', { color: 'black', brand: 'Velasca', price: 295 }),
    makeCatalogItem('shoe-oxblood-derby', 'shoes', 'derby', 'Oxblood Calf Dress Derbies', { color: 'burgundy', brand: 'TLB Mallorca', price: 310 }),
    makeCatalogItem('shoe-brown-oxford', 'shoes', 'oxford', 'Dark Brown Cap-Toe Oxfords', { color: 'brown', brand: 'Cobbler Union', price: 320 }),
  ];
}

test('filterAutonomousHardGates enforces objective real-world gates only and blocks hot-weather parkas, swimwear, bad PDPs, and explicit avoidColors', () => {
  assert.equal(AUTONOMOUS_STYLIST_VERSION, 's6-autonomous-atelier-v1');
  const catalog = [
    ...makeRichCatalog(),
    makeCatalogItem('bad-pdp', 'top', 'shirt', 'Homepage Shirt', { shopUrl: 'https://www.armani.com/en-us/' }),
    makeCatalogItem('bad-host', 'top', 'shirt', 'Unlisted Host Shirt', { imageUrl: 'https://evil.example.com/pic.jpg' }),
    makeCatalogItem('too-expensive', 'outerwear', 'blazer', 'Ultra Luxury Cashmere Blazer', { price: 2500, currency: 'USD' }),
  ];

  const eligibleHot = filterAutonomousHardGates(catalog, {
    context: {
      style: 'classic',
      occasion: 'Date night',
      weather: { temp: 30 },
      budget: { amount: 1500, currency: 'USD' },
      preferences: { avoidColors: ['green'] },
    },
  });
  const ids = new Set(eligibleHot.map((x) => x.id));
  assert.equal(ids.has('outer-winter-puffer'), false, 'heavy winter puffer parka must be blocked at 30C');
  assert.equal(ids.has('bot-swim-trunks'), false, 'swimwear must be blocked for Date night');
  assert.equal(ids.has('bad-pdp'), false, 'generic homepage PDP must be blocked');
  assert.equal(ids.has('bad-host'), false, 'non-allowlisted image host must be blocked');
  assert.equal(ids.has('too-expensive'), false, 'item exceeding single-slot budget headroom must be blocked');
  assert.equal(ids.has('outer-arket-chore'), false, 'explicit user avoidColor green must be respected');
  // Modern pieces stay eligible (zero subjective taste-cage bans)
  assert.equal(ids.has('bot-olive-trouser'), true);
  assert.equal(ids.has('shoe-black-loafer'), true);
  assert.equal(ids.has('outer-espresso-suede'), true);
});

test('curateShowroomSelection prioritizes evening-appropriate archetypes over alphabetical workwear overshirts', () => {
  const eligible = filterAutonomousHardGates(makeRichCatalog(), {
    context: { style: 'classic', occasion: 'Date night', weather: { temp: 18 } },
  });
  const { slotPools } = curateShowroomSelection(eligible, {
    context: { style: 'classic', occasion: 'Date night', weather: { temp: 18 } },
    maxPerSlot: 8,
  });
  // ARKET workwear overshirt must NOT sit at index 0 of Date night outerwear!
  assert.notEqual(slotPools.outerwear[0]?.id, 'outer-arket-chore');
  const top3Outer = slotPools.outerwear.slice(0, 3).map((x) => x.id);
  assert.ok(top3Outer.includes('outer-navy-blazer') || top3Outer.includes('outer-espresso-suede'));
  // Pinned anchor always placed first on its rack
  const pinned = curateShowroomSelection(eligible, {
    context: { style: 'classic', occasion: 'Date night' },
    requiredIds: ['outer-charcoal-cardigan'],
  });
  assert.equal(pinned.slotPools.outerwear[0]?.id, 'outer-charcoal-cardigan');
});

test('runAutonomousVisualStylist happy path completes in 2 cross-lab VLM calls, preserves bespoke titles, strips jargon, and corrects colors', async () => {
  const jpegBytes = await makeTestJpeg(48, 60, 88);
  const vlmCalls = [];
  const photoCache = createPhotoByteCache({ maxEntries: 128, maxBytes: 16 * 1024 * 1024 });

  const fetchImpl = async (url, init = {}) => {
    const u = new URL(String(url));
    if (u.hostname === 'media.frgroup.uz') {
      return new Response(jpegBytes, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    const payload = JSON.parse(init.body);
    vlmCalls.push({ host: u.hostname, model: payload.model, schema: payload.response_format?.json_schema?.name });

    if (vlmCalls.length === 1) {
      // Act 1 on OpenRouter Gemini 2.5 Flash: inspect Showroom Board tokens dynamically from ledger
      const promptText = payload.messages[0].content[0].text;
      const jsonLine = promptText.split('\n').reverse().find((l) => l.trim().startsWith('{'));
      const data = JSON.parse(jsonLine);
      const t = data.showroomRacks.top.map((x) => x.token);
      const l = data.showroomRacks.outerwear.map((x) => x.token);
      const b = data.showroomRacks.bottom.map((x) => x.token);
      const s = data.showroomRacks.shoes.map((x) => x.token);
      const shoeNavyIdx = data.showroomRacks.shoes.findIndex((x) => x.brand === 'Aurelien');
      const s0 = shoeNavyIdx >= 0 ? s[shoeNavyIdx] : s[0];
      const sRest = s.filter((tok) => tok !== s0);
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              outfits: [
                {
                  title: 'Midnight Soft Tailoring',
                  top: t[0], outerwear: l[0], bottom: b[0], shoes: s0,
                  // Intentionally include leaked jargon + misdescribed black loafers to prove sanitizer fixes both!
                  stylistNote: 'Soft tailoring anchored by black loafers and clean drape per shoe-to-palette temperature rules.',
                  whyItFlattersYou: 'The open jacket silhouette draws a clean vertical line across the frame.',
                  stylingTips: ['Wear the outer layer unfastened so the collar catches the light.', 'Keep a minimal trouser break over the loafers.'],
                },
                {
                  title: 'Warm Espresso & Ivory',
                  top: t[1], outerwear: l[1], bottom: b[1], shoes: sRest[0],
                  stylistNote: 'Rich tonal warmth balanced by crisp tailoring for a relaxed evening presence.',
                  whyItFlattersYou: 'Warm tactile texture adds quiet depth without visual clutter.',
                  stylingTips: ['Leave the top collar relaxed for effortless dinner polish.'],
                },
                {
                  title: 'Candlelit Charcoal Knit',
                  top: t[2], outerwear: l[2], bottom: b[2], shoes: sRest[1],
                  stylistNote: 'Soft fine-gauge layering creates understated evening elegance with modern ease.',
                  whyItFlattersYou: 'Balanced proportions keep the look sharp from dinner through late drinks.',
                  stylingTips: ['Let the knit neckline frame the outer layer cleanly.'],
                },
              ],
            }),
          },
        }],
        usage: { prompt_tokens: 900, completion_tokens: 320, total_tokens: 1220 },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    // Act 2 on cross-lab NVIDIA NIM (Kimi or Nemotron): 1 holistic Portfolio Sheet review
    return new Response(JSON.stringify({
      choices: [{
        finish_reason: 'stop',
        message: {
          content: JSON.stringify({
            reviews: [
              { lookToken: 'L00', score: 9.1, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'cream', outerwear: 'navy', bottom: 'charcoal', shoes: 'navy' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Cohesive evening silhouette with rich contrast.' },
              { lookToken: 'L01', score: 8.8, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'white', outerwear: 'brown', bottom: 'navy', shoes: 'brown' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Warm tactile layering with refined footwear.' },
              { lookToken: 'L02', score: 8.6, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'navy', outerwear: 'charcoal', bottom: 'beige', shoes: 'black' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Distinct relaxed knit architecture.' },
            ],
            portfolioNote: 'Three genuinely distinct, elevated evening outfits.',
          }),
        },
      }],
      usage: { prompt_tokens: 800, completion_tokens: 240, total_tokens: 1040 },
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  const result = await runAutonomousVisualStylist({
    request: { style: 'classic', occasion: 'Date night', weather: { temp: 18 }, language: 'en' },
    catalog: makeRichCatalog(),
    openrouterApiKey: 'sk-or-v1-test-autonomous-0123456789',
    nvidiaApiKey: 'nvapi-test-autonomous-0123456789abcdef',
    photoCache,
    fetchImpl,
    photoProvenance: 'host_allowlisted_staging',
    merchantRightsVerified: false,
  });

  assert.equal(result.status, 'applied');
  assert.equal(vlmCalls.length, 2, 'happy path must complete in exactly 2 VLM calls (Act 1 Showroom + Act 2 Portfolio Editor)');
  assert.equal(vlmCalls[0].host, 'openrouter.ai', 'Act 1 runs on primary OpenRouter vision');
  assert.equal(vlmCalls[1].host, 'integrate.api.nvidia.com', 'Act 2 Independent Editor runs cross-lab on NVIDIA NIM');
  assert.equal(result.diagnostics.crossLabVerified, true);
  assert.equal(result.diagnostics.photoProvenance, 'host_allowlisted_staging');
  assert.equal(result.diagnostics.merchantRightsVerified, false);
  assert.equal(result.deck.status, 'complete');
  assert.equal(result.deck.outfits.length, 3);

  // Bespoke titles preserved (zero "Clean separates"!), 0 duplicate SKUs across all 12 pieces
  const titles = result.deck.outfits.map((o) => o.concept.title);
  assert.deepEqual(titles, ['Midnight Soft Tailoring', 'Warm Espresso & Ivory', 'Candlelit Charcoal Knit']);
  const allSkuIds = new Set(result.deck.outfits.flatMap((o) => o.items.map((it) => it.id)));
  assert.equal(allSkuIds.size, 12);

  // Jargon stripped & misdescribed "black loafers" corrected to "navy loafers"
  assert.doesNotMatch(result.deck.outfits[0].stylistNote, /temperature rules/i);
  assert.doesNotMatch(result.deck.outfits[0].stylistNote, /\bblack loafers\b/i);
  assert.match(result.deck.outfits[0].stylistNote, /\bnavy loafers\b/i);
});

test('runAutonomousVisualStylist triggers 1400x900 Visual Swap Board and 1024x1024 re-verification when Editor flags a look < 8.0/10', async () => {
  const jpegBytes = await makeTestJpeg(70, 52, 42);
  let callIdx = 0;
  let sawSwapBoard = false;
  let sawRecheckBoard = false;

  const fetchImpl = async (url, init = {}) => {
    const u = new URL(String(url));
    if (u.hostname === 'media.frgroup.uz') {
      return new Response(jpegBytes, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    callIdx += 1;
    const payload = JSON.parse(init.body);
    const promptText = payload.messages[0].content[0].text;

    if (callIdx === 1) {
      const data = JSON.parse(promptText.split('\n').reverse().find((l) => l.trim().startsWith('{')));
      const t = data.showroomRacks.top.map((x) => x.token);
      const l = data.showroomRacks.outerwear.map((x) => x.token);
      const b = data.showroomRacks.bottom.map((x) => x.token);
      const s = data.showroomRacks.shoes.map((x) => x.token);
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              outfits: [
                { title: 'Midnight Soft Tailoring', top: t[0], outerwear: l[0], bottom: b[0], shoes: s[0], stylistNote: 'Refined evening tailoring.', whyItFlattersYou: '', stylingTips: ['Wear open.'] },
                { title: 'Casual Workwear Draft', top: t[1], outerwear: l[1], bottom: b[1], shoes: s[1], stylistNote: 'Relaxed daytime layer.', whyItFlattersYou: '', stylingTips: ['Roll cuffs.'] },
                { title: 'Evening Knit Depth', top: t[2], outerwear: l[2], bottom: b[2], shoes: s[2], stylistNote: 'Fine-gauge knit poise.', whyItFlattersYou: '', stylingTips: ['Clean trouser break.'] },
              ],
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    if (callIdx === 2) {
      // Act 2 Editor flags L01 as 6.9/10 off-occasion -> triggers Visual Swap on outerwear
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              reviews: [
                { lookToken: 'L00', score: 9.0, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'cream', outerwear: 'navy', bottom: 'charcoal', shoes: 'brown' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Sharp evening tailoring.' },
                { lookToken: 'L01', score: 6.9, approved: false, occasionFit: false, distinctFromOthers: true, photoClean: true, observedColors: { top: 'white', outerwear: 'green', bottom: 'navy', shoes: 'black' }, weakestSlot: 'outerwear', repairBrief: 'Replace casual layer with an elevated evening blazer or suede jacket.', critiqueSummary: 'Outerwear feels too daytime utilitarian for date night.' },
                { lookToken: 'L02', score: 8.7, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'navy', outerwear: 'charcoal', bottom: 'beige', shoes: 'burgundy' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Clean modern evening knit.' },
              ],
              portfolioNote: 'Look 2 needs an outerwear upgrade.',
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    if (callIdx === 3) {
      sawSwapBoard = promptText.includes('1400x900');
      const data = JSON.parse(promptText.split('\n').reverse().find((l) => l.trim().startsWith('{')));
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              selectedToken: data.candidates[0].token,
              updatedTitle: 'Camel Evening Tailoring',
              updatedStylistNote: 'Elevated soft tailoring gives the crisp shirt and dark trousers effortless evening warmth.',
              updatedWhyItFlattersYou: 'Structured shoulders frame the look with clean balance.',
              updatedStylingTips: ['Keep the jacket unbuttoned for relaxed evening poise.'],
              swapRationale: 'Replaces daytime outerwear with refined evening tailoring.',
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }

    sawRecheckBoard = promptText.includes('1024x1024');
    return new Response(JSON.stringify({
      choices: [{
        finish_reason: 'stop',
        message: {
          content: JSON.stringify({
            reviews: [
              { lookToken: 'L00', score: 9.1, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'white', outerwear: 'camel', bottom: 'navy', shoes: 'black' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Elevated evening silhouette.' },
            ],
            portfolioNote: 'Upgraded look earns 9.1/10.',
          }),
        },
      }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  const result = await runAutonomousVisualStylist({
    request: { style: 'classic', occasion: 'Date night', weather: { temp: 18 } },
    catalog: makeRichCatalog(),
    openrouterApiKey: 'sk-or-v1-test-swap-0123456789',
    nvidiaApiKey: 'nvapi-test-swap-0123456789abcdef',
    fetchImpl,
  });

  assert.equal(result.status, 'applied');
  assert.equal(sawSwapBoard, true, 'must render 1400x900 Visual Swap Board when look < 8.0/10');
  assert.equal(sawRecheckBoard, true, 'must re-verify swapped look on 1024x1024 Studio Look Board');
  assert.equal(result.diagnostics.surgicalSwapsAttempted, 1);
  assert.equal(result.diagnostics.surgicalSwapsAccepted, 1);
  assert.equal(result.deck.outfits[1].concept.title, 'Camel Evening Tailoring');
  assert.equal(result.deck.outfits[1].surgicalRefinement?.applied, true);
  assert.equal(result.deck.outfits[1].surgicalRefinement?.scoreAfter, 9.1);
  assert.equal(new Set(result.deck.outfits.flatMap((o) => o.items.map((it) => it.id))).size, 12, 'swapped deck must remain 100% SKU-disjoint');
});

test('verifyAutonomousOutfits enforces server CBU FX budget hard gate & rejects hallucinated tokens; falls back cleanly without API keys', async () => {
  const tokens = {
    T1: makeCatalogItem('t1', 'top', 'shirt', 'Shirt 1', { price: 100 }),
    L1: makeCatalogItem('l1', 'outerwear', 'blazer', 'Expensive Blazer', { price: 900 }),
    L2: makeCatalogItem('l2', 'outerwear', 'cardigan', 'Affordable Cardigan', { price: 150 }),
    B1: makeCatalogItem('b1', 'bottom', 'tailored-trouser', 'Trouser 1', { price: 120 }),
    S1: makeCatalogItem('s1', 'shoes', 'loafer', 'Loafer 1', { price: 130 }),
  };

  // Hallucinated token T99 rejected; over-budget L1 automatically budget-repaired to L2 within $550 cap!
  const { verified, rejected } = await verifyAutonomousOutfits([
    { title: 'Hallucinated', top: 'T99', outerwear: 'L1', bottom: 'B1', shoes: 'S1' },
    { title: 'Budget Checked', top: 'T1', outerwear: 'L1', bottom: 'B1', shoes: 'S1' },
  ], tokens, {
    context: { budget: { amount: 550, currency: 'USD' } },
  });

  assert.equal(rejected.length, 1);
  assert.match(rejected[0], /invalid_token_top/);
  assert.equal(verified.length, 1);
  assert.equal(verified[0].items.find((x) => x.macroCategory === 'outerwear')?.id, 'l2');
  assert.equal(verified[0].budgetCheck.ok, true);
  assert.equal(verified[0].budgetCheck.total, 500);

  // Without API keys, returns clean fallback without throwing
  const noKeys = await runAutonomousVisualStylist({
    request: { style: 'classic', occasion: 'Work' },
    catalog: makeRichCatalog(),
  });
  assert.equal(noKeys.status, 'fallback');
  assert.equal(noKeys.reason, 'not_configured');
});

test('Phase 12 showroom curation deprioritizes daytime work velvet and elevated-occasion leisure trousers, and rescues sweater-under-cardigan via Visual Swap', async () => {
  const velvetBlazer = makeCatalogItem('outer-velvet-burgundy', 'outerwear', 'blazer', 'Tailored Fit Velvet Single-Breasted Blazer', { color: 'burgundy', brand: 'Reiss', price: 498 });
  const beigeLeisure = makeCatalogItem('bot-leisure-jogger', 'bottom', 'trouser', 'Light Beige Leisure Trousers', { color: 'beige', brand: 'Aurelien', price: 355 });
  const extendedCatalog = [...makeRichCatalog(), velvetBlazer, beigeLeisure];

  const workPools = curateShowroomSelection(extendedCatalog, {
    context: { style: 'old_money', occasion: 'Work', weather: { temp: 16 } },
    maxPerSlot: 6,
  }).slotPools;
  const workTop3Ids = workPools.outerwear.slice(0, 3).map((x) => x.id);
  assert.equal(workTop3Ids.includes('outer-velvet-burgundy'), false, 'evening velvet blazer must not lead daytime Work showroom rack');

  const datePools = curateShowroomSelection(extendedCatalog, {
    context: { style: 'classic', occasion: 'Date night', weather: { temp: 16 } },
    maxPerSlot: 8,
  }).slotPools;
  assert.equal(datePools.outerwear.some((x) => x.id === 'outer-velvet-burgundy'), true, 'velvet blazer must remain eligible for evening Date night');
  assert.notEqual(datePools.bottom[0]?.id, 'bot-leisure-jogger', 'leisure/drawstring trousers must be deprioritized on Date night rack');
  assert.equal(datePools.bottom.slice(0, 4).some((x) => x.id === 'bot-leisure-jogger'), false);

  // Physical layering defect detector catches sweater-under-cardigan, daytime velvet, and leisure joggers
  const cardiganSweepDefect = detectLayeringOrOccasionDefect([
    makeCatalogItem('t-knit', 'top', 'fine-knit', 'AUR1 Sweater Navy'),
    makeCatalogItem('l-card', 'outerwear', 'cardigan', 'Macardomon Cardigan'),
    makeCatalogItem('b-trou', 'bottom', 'tailored-trouser', 'Cold wool suit trousers'),
    makeCatalogItem('s-loaf', 'shoes', 'loafer', 'Luteer', { color: 'brown' }),
  ], { occasion: 'Work' });
  assert.equal(cardiganSweepDefect?.code, 'sweater_under_cardigan');
  assert.equal(cardiganSweepDefect?.weakestSlot, 'top');

  // Even if Act 2 Editor rubber-stamps sweater-under-cardigan & leisure joggers at Date night, Act 2b swaps them out cleanly!
  const jpegBytes = await makeTestJpeg(54, 62, 74);
  let callIdx = 0;
  let swapCandidatesSent = [];
  const fetchImpl = async (url, init = {}) => {
    const u = new URL(String(url));
    if (u.hostname === 'media.frgroup.uz') {
      return new Response(jpegBytes, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    callIdx += 1;
    const payload = JSON.parse(init.body);
    const promptText = payload.messages[0].content[0].text;
    const jsonLine = promptText.split('\n').reverse().find((l) => l.trim().startsWith('{'));
    const data = jsonLine ? JSON.parse(jsonLine) : {};

    if (callIdx === 1) {
      const findTok = (slot, matchFn) => data.showroomRacks[slot].find((x) => matchFn(x))?.token || data.showroomRacks[slot][0].token;
      const tSweater = findTok('top', (x) => /crewneck/i.test(x.pieceName || ''));
      const tOther1 = data.showroomRacks.top.find((x) => x.token !== tSweater && /shirt/i.test(x.pieceName || ''))?.token || data.showroomRacks.top[1].token;
      const tOther2 = data.showroomRacks.top.find((x) => x.token !== tSweater && x.token !== tOther1)?.token || data.showroomRacks.top[2].token;
      const lCard = findTok('outerwear', (x) => x.garmentType === 'cardigan');
      const lOther1 = data.showroomRacks.outerwear.find((x) => x.token !== lCard)?.token;
      const lOther2 = data.showroomRacks.outerwear.find((x) => x.token !== lCard && x.token !== lOther1)?.token;
      const bLeisure = data.showroomRacks.bottom.find((x) => /leisure/i.test(x.pieceName || ''))?.token || data.showroomRacks.bottom[2].token;
      const bRest = data.showroomRacks.bottom.filter((x) => x.token !== bLeisure).map((x) => x.token);
      const s = data.showroomRacks.shoes.map((x) => x.token);
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              outfits: [
                { title: 'Navy Tailored Date', top: tOther1, outerwear: lOther1, bottom: bRest[0], shoes: s[0], stylistNote: 'Refined evening tailoring.', whyItFlattersYou: '', stylingTips: ['Keep collar crisp.'] },
                { title: 'Double Knit Flaw', top: tSweater, outerwear: lCard, bottom: bRest[1], shoes: s[1], stylistNote: 'Layered knitwear.', whyItFlattersYou: '', stylingTips: ['Wear open.'] },
                { title: 'Leisure Date Look', top: tOther2, outerwear: lOther2, bottom: bLeisure, shoes: s[2], stylistNote: 'Relaxed suede look.', whyItFlattersYou: '', stylingTips: ['Minimal break.'] },
              ],
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    if (callIdx === 2) {
      // Lenient rubber-stamp score 8.5 even though Look 2 has sweater-under-cardigan and Look 3 has leisure joggers
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              reviews: [
                { lookToken: 'L00', score: 9.0, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'white', outerwear: 'navy', bottom: 'charcoal', shoes: 'brown' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Clean tailoring.' },
                { lookToken: 'L01', score: 8.5, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'charcoal', outerwear: 'charcoal', bottom: 'navy', shoes: 'black' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Looks warm.' },
                { lookToken: 'L02', score: 8.5, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'cream', outerwear: 'brown', bottom: 'beige', shoes: 'navy' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Soft beige bottom.' },
              ],
              portfolioNote: 'Portfolio check.',
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    if (promptText.includes('1400x900')) {
      swapCandidatesSent.push(data.candidates || []);
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              selectedToken: data.candidates[0].token,
              updatedTitle: 'Clean Refined Look',
              updatedStylistNote: 'Clean tailored proportion with refined textures for evening plans.',
              updatedWhyItFlattersYou: 'Open framing elongates the torso.',
              updatedStylingTips: ['Keep outer layer unbuttoned for clean lines.'],
              swapRationale: 'Fixes physical/occasion pairing.',
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    return new Response(JSON.stringify({
      choices: [{
        finish_reason: 'stop',
        message: {
          content: JSON.stringify({
            reviews: [
              { lookToken: 'L00', score: 9.2, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'blue', outerwear: 'charcoal', bottom: 'navy', shoes: 'black' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Refined.' },
            ],
            portfolioNote: 'Solid.',
          }),
        },
      }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  // Fixture where beige leisure trousers enter Date night pool (< 8 bottoms total in test fixture)
  const rescueRes = await runAutonomousVisualStylist({
    request: { style: 'classic', occasion: 'Date night', weather: { temp: 16 } },
    catalog: [...makeRichCatalog(), beigeLeisure],
    openrouterApiKey: 'sk-or-v1-test-rescue-0123456789',
    nvidiaApiKey: 'nvapi-test-rescue-0123456789abcdef',
    fetchImpl,
  });

  assert.equal(rescueRes.status, 'applied');
  assert.equal(rescueRes.diagnostics.surgicalSwapsAttempted, 2, 'both physical sweater-under-cardigan & leisure-trouser defects must trigger Visual Swap even if editor rubber-stamped');
  assert.equal(rescueRes.diagnostics.surgicalSwapsAccepted, 2);
  // Look 2 top candidates offered for cardigan must prioritize shirts/polos over turtleneck/sweater!
  assert.ok(swapCandidatesSent[0].length >= 1);
  assert.notEqual(swapCandidatesSent[0][0].garmentType, 'fine-knit', 'swap rack under cardigan must not rank another fine-knit sweater #1');
  // Final Look 2 & Look 3 have zero sweater-under-cardigan and zero leisure joggers!
  for (const look of rescueRes.deck.outfits) {
    assert.equal(detectLayeringOrOccasionDefect(look.items, { occasion: 'Date night' }), null, `final look "${look.concept.title}" must have 0 layering/lounge defects`);
  }

  // Phase 12b: daytime Work deprioritizes boxy oversized camp-collar shirts, and Editor prompt includes explicit "reviews" JSON keys
  const oversizedShortSleeve = makeCatalogItem('top-archer-oversized', 'top', 'shirt', 'Archer Oversized Viscose Shirt', { color: 'navy', brand: 'AllSaints', price: 129 });
  const workTopPools = curateShowroomSelection([...makeRichCatalog(), oversizedShortSleeve], {
    context: { style: 'old_money', occasion: 'Work', weather: { temp: 16 } },
    maxPerSlot: 6,
  }).slotPools;
  assert.equal(
    workTopPools.top.slice(0, 4).some((x) => x.id === 'top-archer-oversized'),
    false,
    'boxy oversized shirt must not displace crisp dress shirts/fine knits on Work showroom rack',
  );
  const editorPrompt = buildAct2EditorPrompt({
    context: { style: 'classic', occasion: 'Date night' },
    looks: rescueRes.deck.outfits.map((o) => ({ items: o.items, autonomousMeta: { title: o.concept.title, stylingTips: o.stylingTips } })),
  });
  assert.match(editorPrompt, /"reviews":\[/, 'Act 2 prompt must embed explicit JSON keys for Kimi/Gemma json_object mode');

  // Phase 12c: zip/quarter-zip cashmere knit sweater under cardigan is also caught cleanly
  const zipKnitUnderCardigan = detectLayeringOrOccasionDefect([
    makeCatalogItem('t-zip-knit', 'top', 'fine-knit', 'Air force blue cashmere zippered sweater'),
    makeCatalogItem('l-card-zip', 'outerwear', 'cardigan', 'Knitted cardigan'),
    makeCatalogItem('b-trou-2', 'bottom', 'tailored-trouser', 'Navy Tailored Trousers'),
    makeCatalogItem('s-loaf-2', 'shoes', 'loafer', 'Velasca Luteer Loafers', { color: 'black' }),
  ], { occasion: 'Date night' });
  assert.equal(zipKnitUnderCardigan?.code, 'sweater_under_cardigan');
});

test('Act 2 Editor recovers numeric editorScores with crossLabVerified:false when cross-lab Kimi fails over to same-lab Nemotron', async () => {
  const jpegBytes = await makeTestJpeg(56, 62, 72);
  let callCount = 0;
  const fetchImpl = async (url, init = {}) => {
    const u = new URL(String(url));
    if (u.hostname === 'media.frgroup.uz') {
      return new Response(jpegBytes, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    const payload = JSON.parse(init.body);
    callCount += 1;
    if (callCount === 1) {
      // Act 1 on Nemotron
      const data = JSON.parse(payload.messages[0].content[0].text.split('\n').reverse().find((l) => l.trim().startsWith('{')));
      const t = data.showroomRacks.top.map((x) => x.token);
      const l = data.showroomRacks.outerwear.map((x) => x.token);
      const b = data.showroomRacks.bottom.map((x) => x.token);
      const s = data.showroomRacks.shoes.map((x) => x.token);
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              outfits: [
                { title: 'Navy Wool Contrast', top: t[0], outerwear: l[0], bottom: b[0], shoes: s[0], stylistNote: 'Tailored contrast layers for the office.', whyItFlattersYou: '', stylingTips: ['Wear jacket open.'] },
                { title: 'Espresso Work Suede', top: t[1], outerwear: l[1], bottom: b[1], shoes: s[1], stylistNote: 'Warm suede texture paired cleanly.', whyItFlattersYou: '', stylingTips: ['Clean trouser hem.'] },
                { title: 'Charcoal Soft Balance', top: t[2], outerwear: l[2], bottom: b[2], shoes: s[2], stylistNote: 'Restrained tonal workwear.', whyItFlattersYou: '', stylingTips: ['Keep collar crisp.'] },
              ],
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    if (payload.model === 'moonshotai/kimi-k3.1-pro') {
      return new Response('upstream error', { status: 503 });
    }
    // Same-lab Nemotron fallback rescues Act 2 cleanly
    return new Response(JSON.stringify({
      choices: [{
        finish_reason: 'stop',
        message: {
          content: JSON.stringify({
            reviews: [
              { lookToken: 'L00', score: 8.9, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'white', outerwear: 'navy', bottom: 'charcoal', shoes: 'navy' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Clean office tailoring.' },
              { lookToken: 'L01', score: 8.6, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'cream', outerwear: 'brown', bottom: 'navy', shoes: 'brown' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Warm suede texture.' },
              { lookToken: 'L02', score: 8.5, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'blue', outerwear: 'charcoal', bottom: 'beige', shoes: 'black' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Distinct silhouette.' },
            ],
            portfolioNote: 'Recovered via same-lab fallback.',
          }),
        },
      }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  const res = await runAutonomousVisualStylist({
    request: { style: 'old_money', occasion: 'Work', weather: { temp: 16 } },
    catalog: makeRichCatalog(),
    nvidiaApiKey: 'nvapi-test-failover-0123456789abcdef',
    fetchImpl,
  });
  assert.equal(res.status, 'applied');
  assert.equal(res.diagnostics.crossLabVerified, false, 'same-lab Nemotron rescue must report crossLabVerified:false honestly');
  assert.deepEqual(res.diagnostics.editorScores, [8.9, 8.6, 8.5]);
});

test('Phase 13 parallel swaps keep candidates disjoint and recover single-slot duplicate into 3 unique verified looks', async () => {
  const jpegBytes = await makeTestJpeg(74, 82, 90);
  const swapBoardsSent = [];
  const fetchImpl = async (url, init = {}) => {
    const u = new URL(String(url));
    if (u.hostname === 'media.frgroup.uz') {
      return new Response(jpegBytes, { status: 200, headers: { 'Content-Type': 'image/jpeg' } });
    }
    const payload = JSON.parse(init.body);
    const promptText = payload.messages[0].content[0].text;
    const jsonLine = promptText.split('\n').reverse().find((l) => l.trim().startsWith('{'));
    const data = jsonLine ? JSON.parse(jsonLine) : {};

    if (data.showroomRacks) {
      const t = data.showroomRacks.top.map((x) => x.token);
      const l = data.showroomRacks.outerwear.map((x) => x.token);
      const b = data.showroomRacks.bottom.map((x) => x.token);
      const s = data.showroomRacks.shoes.map((x) => x.token);
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              outfits: [
                { title: 'Classic Work Navy', top: t[0], outerwear: l[0], bottom: b[0], shoes: s[0], stylistNote: 'Sharp navy blazer pairing for work.', whyItFlattersYou: '', stylingTips: ['Wear open.'] },
                { title: 'Warm Taupe Meeting', top: t[1], outerwear: l[1], bottom: b[1], shoes: s[1], stylistNote: 'Soft neutral tailoring.', whyItFlattersYou: '', stylingTips: ['Clean trouser drape.'] },
                // Look 3 accidentally reuses shoes s[0] from Look 1 -> single-slot duplicate rescue!
                { title: 'Structured Ivory Look', top: t[2], outerwear: l[2], bottom: b[2], shoes: s[0], stylistNote: 'Ivory and grey layering.', whyItFlattersYou: '', stylingTips: ['Keep collar neat.'] },
              ],
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    if (promptText.includes('1400x900')) {
      swapBoardsSent.push(data.candidates || []);
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              selectedToken: data.candidates[0].token,
              updatedTitle: 'Complete Tailored Quartet',
              updatedStylistNote: 'Refined four-piece work tailoring with balanced contrast.',
              updatedWhyItFlattersYou: '',
              updatedStylingTips: ['Keep blazer open over crisp collar.'],
              swapRationale: 'Upgraded footwear for distinctness.',
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    if (data.looks?.length === 3) {
      // Flag BOTH Look 1 and Look 2 on shoes, while Look 3 was flagged needsDuplicateSlotSwap
      return new Response(JSON.stringify({
        choices: [{
          finish_reason: 'stop',
          message: {
            content: JSON.stringify({
              reviews: [
                { lookToken: 'L00', score: 7.4, approved: false, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'white', outerwear: 'navy', bottom: 'charcoal', shoes: 'brown' }, weakestSlot: 'shoes', repairBrief: 'Upgrade shoes on L00.', critiqueSummary: 'Needs stronger footwear.' },
                { lookToken: 'L01', score: 7.5, approved: false, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'cream', outerwear: 'brown', bottom: 'navy', shoes: 'black' }, weakestSlot: 'shoes', repairBrief: 'Upgrade shoes on L01.', critiqueSummary: 'Needs distinct loafers.' },
                { lookToken: 'L02', score: 8.6, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'blue', outerwear: 'beige', bottom: 'grey', shoes: 'burgundy' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Distinct look.' },
              ],
              portfolioNote: 'Two looks need footwear upgrade plus duplicate rescue.',
            }),
          },
        }],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }
    return new Response(JSON.stringify({
      choices: [{
        finish_reason: 'stop',
        message: {
          content: JSON.stringify({
            reviews: [
              { lookToken: 'L00', score: 8.8, approved: true, occasionFit: true, distinctFromOthers: true, photoClean: true, observedColors: { top: 'white', outerwear: 'navy', bottom: 'charcoal', shoes: 'black' }, weakestSlot: 'none', repairBrief: '', critiqueSummary: 'Elevated.' },
            ],
            portfolioNote: 'Approved.',
          }),
        },
      }],
    }), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };

  const catalog = [
    ...makeRichCatalog(),
    makeCatalogItem('shoe-tan-monk', 'shoes', 'monk', 'Tan Burnished Single Monk Straps', { color: 'tan', brand: 'Carmina', price: 340 }),
    makeCatalogItem('shoe-cognac-loafer', 'shoes', 'loafer', 'Cognac Calf Penny Loafers', { color: 'brown', brand: 'Meermin', price: 220 }),
    makeCatalogItem('shoe-navy-oxford', 'shoes', 'oxford', 'Midnight Museum Calf Oxfords', { color: 'navy', brand: 'Carlos Santos', price: 310 }),
  ];

  const res = await runAutonomousVisualStylist({
    request: { style: 'old_money', occasion: 'Work', weather: { temp: 16 } },
    catalog,
    nvidiaApiKey: 'nvapi-test-parallel-0123456789abcdef',
    timeoutMs: 45_000,
    fetchImpl,
  });
  assert.equal(res.status, 'applied');
  assert.equal(res.deck.outfits.length, 3, 'single-slot duplicate must recover into all 3 verified looks');
  assert.equal(res.diagnostics.surgicalSwapsAttempted, 3);
  assert.equal(res.diagnostics.surgicalSwapsAccepted, 3);
  assert.equal(new Set(res.deck.outfits.flatMap((o) => o.items.map((it) => it.id))).size, 12, 'parallel swaps must maintain 12 distinct SKUs');
});
