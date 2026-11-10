/** @jest-environment node */
import { outfitQualityFloor } from '../../src/domain/outfitQualityFloor';
import { assessOutfitQuality } from '../../src/domain/occasionRules';
import { describeGarment } from '../../src/domain/garmentDescriptor';
import { stylingContext } from '../../src/domain/stylingContext';
import type { OutfitCoherence } from '../../src/domain/outfitCoherence';

const item = (id: string, macroCategory: string, name: string) => ({id, macroCategory, name});
const tailored = (topName: string) => [
  item('top-1', 'top', topName),
  item('layer-1', 'outerwear', 'Navy structured wool blazer'),
  item('bottom-1', 'bottom', 'Grey tailored wool trousers'),
  item('shoe-1', 'shoes', 'Brown leather penny loafers with leather sole'),
];
const strongEvidence = {
  valid: true, score: 0.95, evidence: [],
  dimensions: {layer: 0.95, footwear: 0.95, style: 0.95, palette: 0.95},
} as unknown as OutfitCoherence;
const date = stylingContext({style:'semi_classic',occasion:'Date night'});

test('prominent resort print with short sleeves cannot pass as polished semi-classic date tailoring', () => {
  const look = tailored('Blue short-sleeve all-over tropical floral printed shirt');
  expect(describeGarment(look[0]).patternProminence).toBe('prominent');
  expect(describeGarment(look[1]).structured).toBe(true);
  expect(describeGarment(look[2]).structured).toBe(true);
  const result = outfitQualityFloor(look, date, strongEvidence);
  expect(result.passed).toBe(false);
  expect(result.reason).toBe('prominent_short_sleeve_print_conflicts_with_date_tailoring');
  expect(result.score).toBeLessThanOrEqual(0.49);
});

test('quiet stripes, ordinary short sleeves and knit polos do not trigger a blanket fashion veto', () => {
  for (const name of [
    'Blue fine striped long-sleeve poplin shirt',
    'Plain short-sleeve cotton shirt',
    'Fine merino short-sleeve knit polo',
    'Long-sleeve all-over tropical floral printed shirt',
  ]) {
    const look=tailored(name);
    const verdict=outfitQualityFloor(look,date,strongEvidence);
    expect(verdict.reason).not.toBe('prominent_short_sleeve_print_conflicts_with_date_tailoring');
  }
});

test('a verified user-owned locked printed shirt is not silently replaced by the extra preference gate', () => {
  const look=tailored('Blue short-sleeve all-over tropical floral printed shirt');
  const locked=stylingContext({style:'semi_classic',occasion:'Date night',anchorId:'top-1'});
  const verdict=outfitQualityFloor(look,locked,strongEvidence);
  expect(verdict.reason).not.toBe('prominent_short_sleeve_print_conflicts_with_date_tailoring');
});


test('whole-outfit assessor never accepts a pronounced resort shirt under a dress blazer for a polished date', () => {
  const loud=tailored('Blue short-sleeve all-over tropical floral printed shirt');
  const result=assessOutfitQuality(loud,'semi_classic','Date night');
  expect(result.acceptance.passed).toBe(false);
  expect(result.acceptance.reason).toBeTruthy();
  const restrained=tailored('Blue fine striped long-sleeve poplin shirt');
  const accepted=assessOutfitQuality(restrained,'semi_classic','Date night');
  expect(accepted.acceptance.passed).toBe(true);
});
