import { stylingContext } from '../../src/domain/stylingContext';
it('preserves style and strict occasion independently', () => {
  const c = stylingContext({ style: 'Semi-Classic', occasion: 'Formal' });
  expect(c.dressCode).toBe('formal');
  expect(c.style).toBe('semi_classic');
  expect(c.weather).toBeNull();
  expect(c.uncertainty).toContain('weather_unknown');
});
it('does not turn a date into formal or manufacture a temperature', () => {
  expect(stylingContext({ style: 'minimalist', occasion: 'Date night' }).dressCode).toBe('smart');
  expect(stylingContext({ weather: { condition: 'rain' } }).weather?.temp).toBeUndefined();
});
it('season words do not fabricate a temperature', () => {
  const { resolveStylistWeather } = require('../../src/domain/outfitComposition');
  expect(resolveStylistWeather('summer travel')).toEqual({ condition: 'requested summer season' });
  expect(resolveStylistWeather('winter visit')).toEqual({ condition: 'requested winter season' });
  expect(resolveStylistWeather('destination 12°C')?.temp).toBe(12);
});
it('keeps a measured 32°C destination temperature despite a summer word', () => {
  const { resolveStylistWeather, resolveOutfitComposition } = require('../../src/domain/outfitComposition');
  const resolved = resolveStylistWeather('summer old money work, no extra jacket', { temp:32, condition:'sunny' });
  expect(resolved).toEqual({ temp:32, condition:'requested summer season' });
  expect(resolveOutfitComposition('old_money',resolved,'summer old money work, no extra jacket',true).layered)
    .toBe(false);
  expect(resolveStylistWeather('summer destination at 9°C',{temp:32})?.temp).toBe(9);
  expect(resolveStylistWeather('winter journey',{temp:32})).toEqual({temp:32,condition:'requested winter season'});
});

it('captures negative constraints instead of retaining a mutable caller array', () => {
  const avoidColors = ['red'];
  const c = stylingContext({ preferences: { avoidColors } });
  avoidColors.push('blue');
  expect(c.preferences?.avoidColors).toEqual(['red']);
});
