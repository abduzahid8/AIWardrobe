import test from 'node:test';
import assert from 'node:assert/strict';
import {
  FASHION_COLOR_DICTIONARY,
  getColorMetrics,
  deltaE76,
  scoreColorUndertoneFit,
  scoreOutfitColorTheory,
} from '../../src/domain/colorTheory.ts';

test('fashion color dictionary covers 80+ real fashion colors with valid hex pigments', () => {
  const keys = Object.keys(FASHION_COLOR_DICTIONARY);
  assert.ok(keys.length >= 80, `expected >= 80 fashion colors, got ${keys.length}`);
  for (const key of keys) {
    const m = getColorMetrics(key);
    assert.equal(m.known, true, `expected known metrics for ${key}`);
    assert.match(m.hex, /^#[0-9A-F]{6}$/);
    assert.ok(m.L >= 0 && m.L <= 100, `L out of range for ${key}: ${m.L}`);
    assert.ok(m.chroma >= 0, `chroma negative for ${key}`);
    assert.ok(m.hue >= 0 && m.hue < 360, `hue out of range for ${key}`);
  }
});

test('getColorMetrics computes true sRGB->CIELAB (D65) for reference and arbitrary hex colors', () => {
  const white = getColorMetrics('#FFFFFF');
  assert.equal(white.canonical, 'white');
  assert.equal(white.hex, '#FFFFFF');
  assert.ok(Math.abs(white.L - 100) < 0.05);
  assert.ok(Math.abs(white.a) < 0.05);
  assert.ok(Math.abs(white.b) < 0.05);
  assert.equal(white.valueTier, 'crisp_white');
  assert.equal(white.chromaTier, 'achromatic');
  assert.equal(white.temperature, 'neutral');

  const black = getColorMetrics('#000');
  assert.equal(black.hex, '#000000');
  assert.equal(black.L, 0);
  assert.equal(black.valueTier, 'deep');
  assert.equal(black.chromaTier, 'achromatic');

  // Pure sRGB red (#FF0000) reference D65 CIELAB: L* ≈ 53.23, a* ≈ 80.11, b* ≈ 67.22
  const pureRed = getColorMetrics('#FF0000');
  assert.ok(Math.abs(pureRed.L - 53.23) < 0.2, `unexpected L for #FF0000: ${pureRed.L}`);
  assert.ok(Math.abs(pureRed.a - 80.11) < 0.2, `unexpected a for #FF0000: ${pureRed.a}`);
  assert.ok(Math.abs(pureRed.b - 67.22) < 0.2, `unexpected b for #FF0000: ${pureRed.b}`);
  assert.equal(pureRed.chromaTier, 'vivid');
  assert.equal(pureRed.valueTier, 'mid');

  // Legacy Midnight Blue hex maps to navy while preserving exact #0A1931 CIELAB coordinates
  const midnight = getColorMetrics('#0A1931');
  assert.equal(midnight.canonical, 'navy');
  assert.equal(midnight.hex, '#0A1931');
  assert.equal(midnight.temperature, 'cool');
  assert.equal(midnight.valueTier, 'deep');
  assert.ok(midnight.b < -10, 'midnight blue must have negative Lab b*');

  // Arbitrary unlisted warm leather hex resolves to nearest warm fashion pigment
  const customCognac = getColorMetrics('#A65329');
  assert.equal(customCognac.known, true);
  assert.equal(customCognac.hex, '#A65329');
  assert.equal(customCognac.temperature, 'warm');
  assert.ok(['cognac', 'cinnamon', 'rust', 'saddle', 'caramel', 'copper'].includes(customCognac.canonical));

  // American spelling & unknown inputs
  assert.equal(getColorMetrics('gray').canonical, 'grey');
  assert.equal(getColorMetrics('chernyj').canonical, 'black');
  assert.equal(getColorMetrics('definitely_not_a_color').known, false);
  assert.equal(getColorMetrics('definitely_not_a_color').canonical, 'unknown');
});

test('deltaE76 measures exact Euclidean CIELAB separation and detects near-miss dark shades', () => {
  assert.equal(deltaE76('#FFFFFF', '#000000'), 100);
  assert.equal(deltaE76('navy', 'navy'), 0);
  assert.equal(deltaE76('camel', 'espresso'), deltaE76('espresso', 'camel'));

  const nearMissNavies = deltaE76('#0A1931', '#122340');
  assert.ok(nearMissNavies > 3 && nearMissNavies < 12, `expected near-miss ΔE, got ${nearMissNavies}`);

  const tonalSeparation = deltaE76('cream', 'espresso');
  assert.ok(tonalSeparation > 65, `expected strong tonal separation, got ${tonalSeparation}`);
});

test('scoreColorUndertoneFit aligns high-contrast Winter vs low-contrast Warm Autumn', () => {
  const highContrastWinter = {
    appearance: {
      skinUndertone: 'cool',
      skinDepth: 'light',
      contrastLevel: 'high',
      colorSeason: 'cool_winter',
      flatteringColors: ['navy', 'emerald', 'crisp_white'],
      powerNeutrals: ['black', 'charcoal'],
      avoidColors: ['mustard'],
    },
  };
  const lowContrastWarmAutumn = {
    appearance: {
      skinUndertone: 'warm',
      skinDepth: 'medium',
      contrastLevel: 'low',
      colorSeason: 'warm_autumn',
      flatteringColors: ['camel', 'olive', 'terracotta'],
      powerNeutrals: ['espresso', 'cream'],
      avoidColors: ['ice_blue'],
    },
  };

  // Winter thrives in crisp_white, black, emerald; struggles in camel/mustard
  assert.ok(scoreColorUndertoneFit('crisp_white', highContrastWinter).score >= 0.85);
  assert.ok(scoreColorUndertoneFit('emerald', highContrastWinter).score >= 0.85);
  assert.ok(scoreColorUndertoneFit('camel', highContrastWinter).score <= 0.35);
  assert.equal(scoreColorUndertoneFit('mustard', highContrastWinter).tier, 'avoided');

  // Warm Autumn thrives in camel, olive, espresso, cream; struggles in ice_blue / silver / crisp_white
  assert.ok(scoreColorUndertoneFit('camel', lowContrastWarmAutumn).score >= 0.85);
  assert.ok(scoreColorUndertoneFit('olive', lowContrastWarmAutumn).score >= 0.85);
  assert.ok(scoreColorUndertoneFit('espresso', lowContrastWarmAutumn).score >= 0.82);
  assert.equal(scoreColorUndertoneFit('ice_blue', lowContrastWarmAutumn).tier, 'avoided');
  assert.ok(scoreColorUndertoneFit('crisp_white', lowContrastWarmAutumn).score <= 0.30);

  // Unspecified DNA stays neutral
  const neutral = scoreColorUndertoneFit('navy', null);
  assert.equal(neutral.score, 0.55);
  assert.equal(neutral.tier, 'neutral');
});

test('scoreOutfitColorTheory rewards true fashion harmonies', () => {
  const warmAutumn = {
    appearance: {
      skinUndertone: 'warm',
      contrastLevel: 'low',
      colorSeason: 'warm_autumn',
    },
  };
  const coolWinter = {
    appearance: {
      skinUndertone: 'cool',
      contrastLevel: 'high',
      colorSeason: 'cool_winter',
    },
  };

  // 1. Warm earth analogous
  const earth = scoreOutfitColorTheory(
    { top: 'cream', outerwear: 'camel', bottom: 'olive', shoes: 'espresso' },
    warmAutumn,
  );
  assert.equal(earth.harmonyMode, 'warm_earth_analogous');
  assert.ok(earth.score >= 0.85, `expected high warm earth score, got ${earth.score}`);
  assert.equal(earth.flaws.length, 0);

  // 2. Cool architectural contrast
  const arch = scoreOutfitColorTheory(
    { top: 'crisp_white', outerwear: 'charcoal', bottom: 'slate', shoes: 'black' },
    coolWinter,
  );
  assert.equal(arch.harmonyMode, 'cool_architectural_contrast');
  assert.ok(arch.score >= 0.84, `expected high cool architectural score, got ${arch.score}`);

  // 3. Tonal monochrome gradient
  const tonal = scoreOutfitColorTheory(
    { top: 'white', outerwear: 'light_grey', bottom: 'charcoal', shoes: 'black' },
    null,
  );
  assert.ok(['tonal_monochrome_gradient', 'cool_architectural_contrast'].includes(tonal.harmonyMode));
  assert.ok(tonal.score >= 0.82);

  // 4. Neutral anchor + single accent
  const accent = scoreOutfitColorTheory(
    { top: 'white', outerwear: 'burgundy', bottom: 'charcoal', shoes: 'black' },
    null,
  );
  assert.ok(['neutral_anchor_plus_accent', 'cool_architectural_contrast'].includes(accent.harmonyMode));
  assert.ok(accent.score >= 0.80);

  // 5. Classic navy + warm cognac complementary (NOT a warm-vs-cool clash)
  const navyCognac = scoreOutfitColorTheory(
    { top: 'white', outerwear: 'navy', bottom: 'beige', shoes: 'cognac' },
    null,
  );
  assert.notEqual(navyCognac.harmonyMode, 'discordant');
  assert.ok(navyCognac.score >= 0.78, `expected strong classic navy+cognac score, got ${navyCognac.score}`);
  assert.equal(navyCognac.flaws.length, 0);
});

test('scoreOutfitColorTheory catches near-miss dark clashes, cognac-vs-icy-black, and multi-accent noise', () => {
  // 1. Two accidentally-close navies without tonal separation
  const nearMissNavies = scoreOutfitColorTheory(
    { top: 'white', outerwear: '#0A1931', bottom: '#14233F', shoes: 'brown' },
    null,
  );
  assert.ok(nearMissNavies.flaws.includes('near_miss_dark_tonal_clash'));
  assert.equal(nearMissNavies.harmonyMode, 'discordant');
  assert.ok(nearMissNavies.score <= 0.46);

  // 2. Navy jacket directly over black trousers (adjacent near-miss clash)
  const navyBlackAdjacent = scoreOutfitColorTheory(
    { top: 'white', outerwear: 'navy', bottom: 'black', shoes: 'black' },
    null,
  );
  assert.ok(navyBlackAdjacent.flaws.includes('near_miss_navy_black_clash'));
  assert.equal(navyBlackAdjacent.harmonyMode, 'discordant');

  // Separated navy blazer + mid_grey trousers + black Oxfords is classic & valid
  const navyGreyBlack = scoreOutfitColorTheory(
    { top: 'white', outerwear: 'navy', bottom: 'mid_grey', shoes: 'black' },
    null,
  );
  assert.equal(navyGreyBlack.flaws.includes('near_miss_navy_black_clash'), false);
  assert.ok(navyGreyBlack.score >= 0.75);

  // 3. Warm cognac leather shoes clashing against stark icy black + cool grey
  const cognacIcyBlack = scoreOutfitColorTheory(
    { top: 'crisp_white', outerwear: 'black', bottom: 'cool_grey', shoes: 'cognac' },
    null,
  );
  assert.ok(cognacIcyBlack.flaws.includes('warm_cognac_leather_clashes_with_icy_cool_palette'));
  assert.equal(cognacIcyBlack.harmonyMode, 'discordant');
  assert.ok(cognacIcyBlack.temperatureCoherence <= 0.50);

  // 4. Multi-accent visual noise
  const noisy = scoreOutfitColorTheory(
    { top: 'red', outerwear: 'cobalt', bottom: 'yellow', shoes: 'magenta' },
    null,
  );
  assert.ok(noisy.flaws.includes('multi_accent_visual_noise'));
  assert.equal(noisy.harmonyMode, 'discordant');
  assert.ok(noisy.score <= 0.45);

  // 5. Insufficient evidence degrades honestly
  const unknown = scoreOutfitColorTheory({ top: 'unknown', bottom: '' }, null);
  assert.equal(unknown.harmonyMode, 'insufficient_evidence');

  // 6. Phase 10 Defect D: Stark black formal shoes clash with warm caramel/orange/brown trousers
  const blackShoesWarmTrousers = scoreOutfitColorTheory(
    { top: 'grey', outerwear: 'brown', bottom: 'orange', shoes: 'black' },
    null,
  );
  assert.ok(blackShoesWarmTrousers.flaws.includes('stark_black_shoes_clash_with_warm_earth_trousers'));
  assert.equal(blackShoesWarmTrousers.harmonyMode, 'discordant');

  const brownShoesWarmTrousers = scoreOutfitColorTheory(
    { top: 'cream', outerwear: 'brown', bottom: 'camel', shoes: 'espresso' },
    null,
  );
  assert.ok(!brownShoesWarmTrousers.flaws.includes('stark_black_shoes_clash_with_warm_earth_trousers'));
  assert.ok(brownShoesWarmTrousers.score > blackShoesWarmTrousers.score + 0.15);

  // Modern pairings: sleek black loafers/boots with dark olive or rich espresso/chocolate trousers are NOT discordant
  const blackOnDarkOlive = scoreOutfitColorTheory(
    { top: 'cream', outerwear: 'charcoal', bottom: 'dark_olive', shoes: 'black' },
    null,
  );
  assert.equal(blackOnDarkOlive.flaws.includes('stark_black_shoes_clash_with_warm_earth_trousers'), false);
  assert.notEqual(blackOnDarkOlive.harmonyMode, 'discordant');

  const blackOnEspresso = scoreOutfitColorTheory(
    { top: 'ivory', outerwear: 'black', bottom: 'espresso', shoes: 'black' },
    null,
  );
  assert.equal(blackOnEspresso.flaws.includes('stark_black_shoes_clash_with_warm_earth_trousers'), false);
  assert.notEqual(blackOnEspresso.harmonyMode, 'discordant');
});
