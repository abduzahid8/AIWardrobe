import { parseBudgetControl, strictestOutfitBudget } from '../../src/domain/outfitBudgetSelection';
import { validateOutfitGenerationRequest } from '../../src/domain/outfitGenerationRequest';

test('an empty budget control leaves the request unbudgeted', () => {
  expect(parseBudgetControl('  ', 'UZS')).toBeNull();
  expect(validateOutfitGenerationRequest({preferences: {}, allowPartialResults: true})).toBeTruthy();
});

test('a selected UZS cap is a complete-look budget and text can only tighten it', () => {
  const selected = parseBudgetControl('1 500 000', 'UZS');
  expect(selected).toEqual({amount: 1500000, currency: 'UZS'});
  expect(parseBudgetControl('1,500,000', 'UZS')).toEqual(selected);
  expect(strictestOutfitBudget(selected, {amount: 1400000, currency: 'UZS'}, b => b))
    .toEqual({amount: 1400000, currency: 'UZS'});
  expect(validateOutfitGenerationRequest({preferences: {maxShopBudget: selected}})).toBeTruthy();
});

test('invalid explicit amounts and request caps fail before generation', () => {
  expect(() => parseBudgetControl('1,50,000', 'UZS')).toThrow('invalid_outfit_budget');
  expect(() => parseBudgetControl('0', 'UZS')).toThrow('invalid_outfit_budget');
  expect(() => validateOutfitGenerationRequest({preferences: {maxShopBudget: {amount: -1, currency: 'USD'}}})).toThrow();
  expect(() => validateOutfitGenerationRequest({allowPartialResults: 'yes'})).toThrow();
});
