import test from 'node:test';
import assert from 'node:assert/strict';
import sharp from 'sharp';
import {
  AUTONOMOUS_STYLIST_VERSION,
  filterAutonomousHardGates,
  curateShowroomSelection,
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
