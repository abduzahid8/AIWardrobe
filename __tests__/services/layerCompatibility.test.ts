import { layerCompatibility } from '../../src/domain/layerCompatibility';
import { describeGarment } from '../../src/domain/garmentDescriptor';
import { stylingContext } from '../../src/domain/stylingContext';
const top = (name: string) => ({ name, macroCategory: 'top' }),
  layer = (name: string) => ({ name, macroCategory: 'outerwear' });
const casual = stylingContext({ style: 'casual' });
it('rejects directional bulk and fit conflicts', () => {
  expect(
    layerCompatibility(top('Heavy chunky sweater'), layer('Thick cable-knit pullover'), casual)
      .reasons
  ).toContain('competing_heavy_layers');
  expect(
    layerCompatibility(top('Oversized cotton shirt'), layer('Slim fitted cardigan'), casual).valid
  ).toBe(false);
});
it('accepts light shirt / fine cardigan and tee / overshirt without inventing weather', () => {
  for (const pair of [
    [top('Lightweight poplin shirt'), layer('Fine merino cardigan')],
    [top('Cotton T-shirt'), layer('Linen overshirt')],
  ]) {
    const result = layerCompatibility(pair[0], pair[1], casual);
    expect(result.valid).toBe(true);
    expect(result.score).toBeGreaterThanOrEqual(0.8);
  }
});
it('strict Formal needs a structured layer and proper base', () => {
  const c = stylingContext({ style: 'semi_classic', occasion: 'Formal' });
  expect(layerCompatibility(top('Fine cotton polo'), layer('Merino cardigan'), c).valid).toBe(
    false
  );
  expect(layerCompatibility(top('White dress shirt'), layer('Wool blazer'), c).valid).toBe(true);
});
it('nominal categories do not erase evidence distinctions', () => {
  expect(
    describeGarment({ name: 'Waterproof hiking lug-sole boots', macroCategory: 'shoes' }).rugged
  ).toBe(true);
  expect(
    describeGarment({ name: 'Slim leather dress boots', macroCategory: 'shoes' }).subtype
  ).toBe('dress-boot');
  expect(describeGarment({ name: 'Stripe track pants', macroCategory: 'bottom' }).athletic).toBe(
    true
  );
  expect(
    describeGarment({ name: 'Tailored wool trousers', macroCategory: 'bottom' }).formality
  ).toBe('formal');
  expect(describeGarment(top('Field shirt')).subtype).toBe('field-shirt');
  expect(describeGarment(top('Cotton dress shirt')).subtype).toBe('dress-shirt');
  expect(describeGarment(layer('Fine cardigan')).bulk).toBe('light');
  expect(describeGarment(layer('Chunky cardigan')).bulk).toBe('heavy');
});
it('unknown evidence remains unknown', () => {
  const d = describeGarment({ name: 'Product 103', macroCategory: 'shoes' });
  expect(d.subtype).toBe('unknown');
  expect(d.formality).toBe('unknown');
  expect(d.pattern).toBe('unknown');
  expect(d.texture).toBe('unknown');
});
it('reports directional hem/closure evidence without declaring every cropped look impossible', () => {
  const result = layerCompatibility(
    top('Longline cotton shirt'),
    layer('Cropped fine cardigan'),
    casual
  );
  expect(result.valid).toBe(true);
  expect(result.evidence).toContain('long_base_below_cropped_layer');
  expect(describeGarment(layer('Cropped fine cardigan')).closure).toBe('openable');
  expect(describeGarment(layer('Fine pullover sweater')).closure).toBe('pullover');
});
