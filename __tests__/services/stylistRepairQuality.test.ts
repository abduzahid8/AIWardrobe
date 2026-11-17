import { describeGarment } from '../../src/domain/garmentDescriptor';
import { passesStyleJudge } from '../../src/domain/occasionRules';
import { scoreGarmentForContext } from '../../src/domain/garmentContext';
import { stylingContext } from '../../src/domain/stylingContext';
import { stylingCopyParts } from '../../src/services/stylingCopy';
import i18n from 'i18next';

const look = (names: string[]) => names.map((name,i)=>({id:`piece-${i}`,name,macroCategory:['top','outerwear','bottom','shoes'][i]}));
test.each([
  ['Light blue button-down shirt','Dark cashmere sweater','Interlock trousers','Black leather Oxford shoes'],
  ['Cream long-sleeve polo','Black leather peacoat','Interlock trousers','Black leather dress Oxford shoes'],
  ['Beige military utility shirt with chest pockets','Navy formal blazer','White tailored trousers','Black leather derby shoes'],
  ['Brown polo','Bulky shearling biker jacket','Beige tailored trousers','Hybrid leather sneaker shoes'],
])('prevents the reported Classic combination: %s / %s', (...names) => {
  expect(passesStyleJudge(look(names),'classic','Travel')).toBe(false);
  expect(passesStyleJudge(look(names),'classic','Work')).toBe(false);
});
test('Interlock is knit construction evidence, not invented athletic or visual detailing', () => {
  const d = describeGarment({name:'Interlock trousers',macroCategory:'bottom'});
  expect(d.subtype).toBe('knit-trouser');expect(d.athletic).toBe(false);
  expect(describeGarment({name:'Tailored jersey trousers',macroCategory:'bottom'}).subtype).toBe('tailored-trouser');
});
test.each([
  ['White cotton T-shirt','Black leather peacoat','Straight dark jeans','Clean leather sneakers'],
  ['Cotton polo','Shearling biker jacket','Straight dark jeans','Leather boots'],
  ['White cotton T-shirt','Lightweight cardigan','Interlock trousers','Clean leather sneakers'],
  ['Beige military utility shirt','Cotton field jacket','Straight jeans','Leather boots'],
])('retains coherent casual uses of the same garment families', (...names) => {
  expect(passesStyleJudge(look(names),'casual','Everyday')).toBe(true);
});
test('Classic travel favors removable layers and walking evidence; business travel stays possible', () => {
  const travel = stylingContext({style:'classic',occasion:'Travel'});
  const score = (name:string,macroCategory:string, context=travel) => scoreGarmentForContext({name,macroCategory}, context);
  expect(score('Lightweight cotton cardigan','outerwear')).toBeGreaterThan(score('Navy wool blazer','outerwear'));
  expect(score('Flexible cushioned leather loafers','shoes')).toBeGreaterThan(score('Black leather Oxford shoes','shoes'));
  expect(score('Bulky shearling biker jacket','outerwear')).toBeLessThan(score('Lightweight cotton jacket','outerwear'));
  const business = stylingContext({style:'classic',occasion:'Travel',request:'business meeting'});
  expect(score('Navy wool blazer','outerwear',business)).toBeGreaterThan(score('Navy wool blazer','outerwear'));
  expect(passesStyleJudge(look(['White cotton shirt','Navy unlined blazer','Beige chinos','Flexible leather loafers']),'classic','Travel')).toBe(true);
});
test.each(['en','ru','uz'])('%s copy names the actual garments and separates occasion advice', async locale => {
  await i18n.init({lng:locale,resources:{[locale]:{translation:require(`../../i18n/locales/${locale}.json`)}}});
  const items=look(['White poplin dress shirt','Navy unlined wool blazer','Grey tailored trousers','Black leather Oxford shoes']);
  const copy=stylingCopyParts(items,stylingContext({style:'classic',occasion:'Formal'}));
  for(const item of items)expect(copy.description).toContain(item.name);
  expect(copy.tips.length).toBeGreaterThan(0);
  expect(copy.tips).not.toContain(copy.description);
  expect(copy.tips.join(' ')).not.toContain('To‘liq obraz');
  expect(copy.tips.join(' ')).not.toContain('outfitMaker.');
});
