/** @jest-environment node */
import {assessOutfitQuality, passesStyleJudge} from '../../src/domain/occasionRules';
import {describeGarment} from '../../src/domain/garmentDescriptor';
import {layerCompatibility} from '../../src/domain/layerCompatibility';
import {stylingContext} from '../../src/domain/stylingContext';
import {validatedQuartets} from '../../src/domain/validatedQuartets';
import {fixtureLook, stylingV193Fixtures} from '../fixtures/stylingV193';

test.each(stylingV193Fixtures)('$id: occasion compatibility from actual evidence', fixture => {
  const items = fixtureLook(fixture.names);
  const style = fixture.style || 'semi_classic', occasion = fixture.occasion || 'Date night';
  const assessment = assessOutfitQuality(items, style, occasion, null);
  expect({id: fixture.id, reason: assessment.acceptance.reason, accepted: assessment.acceptance.passed})
    .toMatchObject({accepted: fixture.accepted});
  expect(passesStyleJudge(items, style, occasion, null)).toBe(fixture.accepted);
});

test('missing temperature does not penalize a winter layer as if hot weather were known', () => {
  const [top, layer] = fixtureLook(['Plain poplin shirt', 'Heavyweight wool coat', 'Cotton chinos', 'Leather loafers']);
  const unknown = stylingContext({occasion: 'Travel'});
  const cold = stylingContext({occasion: 'Travel', weather: {temp: 4}});
  const hot = stylingContext({occasion: 'Travel', weather: {temp: 28}});
  const a = layerCompatibility(top, layer, unknown), b = layerCompatibility(top, layer, cold);
  expect(unknown.weather).toBeNull();
  expect(unknown.uncertainty).toContain('weather_unknown');
  expect(a.score).toBe(b.score);
  expect(a.evidence).not.toContain('heavy_layer_without_cold_need');
  expect(layerCompatibility(top, layer, hot).score).toBeLessThan(a.score);
});

test.each(['Washed black jeans', 'Garment-washed dark indigo jeans', 'Raw blue denim jeans'])(
  'ambiguous wash alone is not evidence of extreme fading: %s', name => {
    expect(describeGarment({name, macroCategory: 'bottom'}).fadedDenim).toBe(false);
  }
);

test('sparse material metadata does not invent winter weight or pattern prominence', () => {
  const d = describeGarment({name: 'Jacquard cardigan', material: 'wool cashmere', macroCategory: 'outerwear'});
  expect(d.bulk).toBe('unknown');
  expect(d.pattern).toBe('texture');
  expect(d.patternProminence).toBe('unknown');
  expect(describeGarment({name: 'Oversized cotton shirt with subtle micro checks', macroCategory: 'top'}).patternProminence).toBe('subtle');
  expect(describeGarment({name: 'Fine wool checked blazer', macroCategory: 'outerwear'}).patternProminence).toBe('unknown');
});

test('local quartet builder applies the same acceptance floor as the Edge path', () => {
  const bad = fixtureLook(stylingV193Fixtures[0].names, 'busy');
  const good = fixtureLook(stylingV193Fixtures.find(f => f.id === 'refined-neutral')!.names, 'quiet');
  const context = stylingContext({style: 'semi_classic', occasion: 'Date night'});
  const candidates = validatedQuartets([...bad, ...good], context, () => true);
  expect(candidates.length).toBeGreaterThan(0);
  for (const look of candidates)
    expect(assessOutfitQuality(look.items, context.style, context.occasion).acceptance.passed).toBe(true);
  const winter = fixtureLook(stylingV193Fixtures.find(f => f.id === 'airy-heavy-winter')!.names);
  expect(validatedQuartets(winter, context, () => true)).toEqual([]);
});
