import { styleJudgeReason, rejectForStyleOccasion, normalizeGarmentText } from '../../features/outfit-generator/utils/occasionRules';
const item = (name: string, macroCategory: string) => ({ name, macroCategory });
const office = [item('White cotton shirt', 'top'), item('Navy trousers', 'bottom'), item('Brown loafers', 'shoes')];
describe('strict stylist request contract', () => {
  it('rejects Russian logo sweatshirts in professional looks', () => {
    expect(rejectForStyleOccasion(item('Свитшот Monologo', 'outerwear'), 'business_casual', 'Modern Professional')).not.toBeNull();
  });
  it('normalizes Russian footwear and colors consistently', () => {
    expect(normalizeGarmentText('Коричневые замшевые лоферы')).toMatch(/brown.*suede.*loafers/);
  });
  it.each(['gym workout, old money', 'тренировка в спортзале'])('rejects office garments for %s', occasion => {
    expect(styleJudgeReason(office, 'old_money', occasion)).toMatch(/workout/);
  });
  it('does not mistake an ordinary suit for black tie', () => {
    expect(styleJudgeReason([...office, item('Navy suit blazer', 'outerwear')], 'classic', 'black-tie')).toMatch(/tuxedo/);
  });
  it('accepts a documented black-tie combination', () => {
    expect(styleJudgeReason([item('Black tuxedo dinner jacket', 'outerwear'), item('White tuxedo formal shirt', 'top'), item('Black tuxedo trousers', 'bottom'), item('Black patent Oxford shoes', 'shoes')], 'classic', 'black tie')).toBeNull();
  });
  it('accepts verified athletic clothing for workouts', () => {
    expect(styleJudgeReason([item('Performance training tee', 'top'), item('Running athletic shorts', 'bottom'), item('Training athletic sneakers', 'shoes')], 'casual', 'gym workout')).toBeNull();
  });
  it('does not invent weather restrictions when weather is unavailable', () => {
    expect(styleJudgeReason(office, 'casual', 'everyday', undefined)).toBeNull();
  });
});

it('allows training shirts and shorts despite a classic aesthetic preference', () => {
  const items = [item('Performance training shirt', 'top'), item('Running shorts', 'bottom'), item('Athletic training sneakers', 'shoes')];
  expect(styleJudgeReason(items, 'old_money', 'sport')).toBeNull();
  expect(rejectForStyleOccasion(items[1], 'old_money', 'sport')).toBeNull();
});
it('does not treat a sport coat as verified workout clothing', () => {
  expect(styleJudgeReason([item('Sport coat', 'top'), item('Running shorts', 'bottom'), item('Athletic sneakers', 'shoes')], 'casual', 'gym')).toMatch(/athletic top/);
});
