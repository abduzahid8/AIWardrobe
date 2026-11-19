import test from 'node:test';
import assert from 'node:assert/strict';
import {
  resolveOutfitGrammar,
  scoreThermalSuitability,
} from '../../src/domain/outfitGrammar.ts';

test('resolveOutfitGrammar always preserves the binding 4-real-purchasable-item contract', () => {
  const fullCatalog = { top: 300, outerwear: 200, bottom: 180, shoes: 160, accessory: 45, dress: 40 };
  const currentCatalog = { top: 3082, outerwear: 2117, bottom: 1672, shoes: 1649, accessory: 0, dress: 0 };

  // Mild/cool default -> layered_separates
  const cool = resolveOutfitGrammar({ weather: { temp: 15 }, occasion: 'work', presentation: 'menswear' }, fullCatalog);
  assert.equal(cool.grammarId, 'layered_separates');
  assert.deepEqual(cool.slots, ['top', 'outerwear', 'bottom', 'shoes']);
  assert.equal(cool.fourItemContract, true);
  assert.equal(cool.fallbackApplied, false);

  // Warm/hot (>=26°C) WITH real accessory inventory -> warm_weather_separates
  const hotWithAccessories = resolveOutfitGrammar(
    { weather: { temp: 31, condition: 'sunny' }, occasion: 'casual', presentation: 'menswear' },
    fullCatalog,
  );
  assert.equal(hotWithAccessories.grammarId, 'warm_weather_separates');
  assert.deepEqual(hotWithAccessories.slots, ['top', 'bottom', 'shoes', 'accessory']);
  assert.equal(hotWithAccessories.fourItemContract, true);
  assert.equal(hotWithAccessories.fallbackApplied, false);

  // Warm/hot (>=26°C) WITHOUT accessory inventory (current catalog) -> honest fallback to layered_separates + breathable thermal constraints
  const hotCurrentCatalog = resolveOutfitGrammar(
    { weather: { temp: 31, condition: 'sunny' }, occasion: 'work', presentation: 'menswear' },
    currentCatalog,
  );
  assert.equal(hotCurrentCatalog.grammarId, 'layered_separates');
  assert.deepEqual(hotCurrentCatalog.slots, ['top', 'outerwear', 'bottom', 'shoes']);
  assert.equal(hotCurrentCatalog.fourItemContract, true);
  assert.equal(hotCurrentCatalog.fallbackApplied, true);
  assert.equal(hotCurrentCatalog.fallbackReason, 'no_accessory_inventory_for_warm_weather_separates');
  assert.equal(hotCurrentCatalog.thermalConstraints.maxOuterwearWeight, 'ultralight_breathable');
  assert.equal(hotCurrentCatalog.thermalConstraints.requireBreathableLayer, true);
  assert.ok(hotCurrentCatalog.thermalConstraints.forbiddenOuterwearTypes.includes('puffer'));

  // One-piece editorial WITH dress + accessory inventory
  const onePiece = resolveOutfitGrammar(
    { weather: { temp: 20 }, occasion: 'formal', presentation: 'womenswear' },
    fullCatalog,
  );
  assert.equal(onePiece.grammarId, 'one_piece_editorial');
  assert.deepEqual(onePiece.slots, ['dress', 'outerwear', 'shoes', 'accessory']);
  assert.equal(onePiece.fourItemContract, true);
  assert.equal(onePiece.fallbackApplied, false);

  // One-piece editorial WITHOUT dress/accessory inventory -> honest fallback
  const onePieceFallback = resolveOutfitGrammar(
    { weather: { temp: 20 }, occasion: 'formal', presentation: 'womenswear', request: 'silk midi dress' },
    currentCatalog,
  );
  assert.equal(onePieceFallback.grammarId, 'layered_separates');
  assert.equal(onePieceFallback.slots.length, 4);
  assert.equal(onePieceFallback.fallbackApplied, true);
  assert.match(String(onePieceFallback.fallbackReason), /one_piece_editorial/);
});

test('scoreThermalSuitability strongly prefers breathable layers at >=26°C and insulating coats at <=10°C', () => {
  const linenOvershirt = {
    macroCategory: 'outerwear',
    name: 'Unlined Linen Open-Weave Overshirt',
    material: '100% linen',
    bulk: 'light',
  };
  const cottonShirtJacket = {
    macroCategory: 'outerwear',
    name: 'Lightweight Cotton Shirt Jacket',
    material: 'cotton',
  };
  const woolOvercoat = {
    macroCategory: 'outerwear',
    name: 'Heavy Wool Double-Breasted Overcoat',
    material: 'heavy wool',
    bulk: 'heavy',
  };
  const downPuffer = {
    macroCategory: 'outerwear',
    name: 'Goose-Down Insulated Puffer Parka',
    material: 'down',
    insulated: true,
  };
  const leatherBoots = {
    macroCategory: 'shoes',
    name: 'Weatherproof Leather Chelsea Boots',
    material: 'leather',
  };
  const summerSandals = {
    macroCategory: 'shoes',
    name: 'Open-Toe Suede Sandals',
    material: 'suede',
  };

  // Hot 30°C
  const hot = { temp: 30, condition: 'sunny' };
  const linenHot = scoreThermalSuitability(linenOvershirt, hot);
  const cottonHot = scoreThermalSuitability(cottonShirtJacket, hot);
  const woolHot = scoreThermalSuitability(woolOvercoat, hot);
  const pufferHot = scoreThermalSuitability(downPuffer, hot);

  assert.equal(linenHot.suitable, true);
  assert.ok(linenHot.score >= 0.90, `linen overshirt hot score: ${linenHot.score}`);
  assert.equal(cottonHot.suitable, true);
  assert.ok(cottonHot.score >= 0.85, `cotton shirt-jacket hot score: ${cottonHot.score}`);
  assert.equal(woolHot.suitable, false);
  assert.ok(woolHot.score <= 0.25, `wool overcoat hot score: ${woolHot.score}`);
  assert.equal(pufferHot.suitable, false);
  assert.ok(pufferHot.score <= 0.12, `puffer hot score: ${pufferHot.score}`);
  assert.ok(linenHot.score > woolHot.score + 0.65);

  // Cold 5°C
  const cold = { temp: 5, condition: 'winter overcast' };
  const pufferCold = scoreThermalSuitability(downPuffer, cold);
  const woolCold = scoreThermalSuitability(woolOvercoat, cold);
  const linenCold = scoreThermalSuitability(linenOvershirt, cold);
  const bootsCold = scoreThermalSuitability(leatherBoots, cold);
  const sandalsCold = scoreThermalSuitability(summerSandals, cold);

  assert.equal(pufferCold.suitable, true);
  assert.ok(pufferCold.score >= 0.92);
  assert.equal(woolCold.suitable, true);
  assert.ok(woolCold.score >= 0.88);
  assert.equal(linenCold.suitable, false);
  assert.ok(linenCold.score <= 0.20);
  assert.equal(bootsCold.suitable, true);
  assert.ok(bootsCold.score >= 0.90);
  assert.equal(sandalsCold.suitable, false);
  assert.ok(sandalsCold.score <= 0.10);

  // Unknown weather stays neutral
  const unknown = scoreThermalSuitability(linenOvershirt, null);
  assert.equal(unknown.band, 'unknown');
  assert.equal(unknown.suitable, true);
  assert.equal(unknown.score, 0.65);
});
