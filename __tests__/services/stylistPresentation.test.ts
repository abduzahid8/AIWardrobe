import i18n from 'i18next';
import { outfitDescription, outfitConceptTitle } from '../../src/services/stylingCopy';
import { stylingContext } from '../../src/domain/stylingContext';
import { assessOutfitQuality } from '../../src/domain/occasionRules';

const items = ['White cotton shirt', 'Navy linen blazer', 'Grey tailored trousers', 'Black leather Oxford shoes']
  .map((name, i) => ({id: `item-${i}`, name, macroCategory: ['top','outerwear','bottom','shoes'][i]}));
const context = stylingContext({style:'classic',occasion:'Work'});

test.each(['en','ru','uz'])('versioned explanation renders grounded facts in %s despite server English', async lng => {
  const resources = {[lng]: {translation: require(`../../i18n/locales/${lng}.json`)}};
  await i18n.init({lng, resources});
  const description = outfitDescription({explanationVersion:1,description:'English server fallback must not win'}, items, context);
  expect(description).not.toContain('English server fallback');
  expect(description).toContain('Grey tailored trousers');
  expect(description).toContain('Black leather Oxford shoes');
  expect(description).not.toContain('outfitMaker.');
  if (lng !== 'en') expect(description).not.toContain('The line of');
  const spend = i18n.t('outfitMaker.lookSpend', {total:'400 UZS',cap:'500 UZS'});
  expect(spend).toContain('400 UZS');
  expect(spend).toContain('500 UZS');
  expect(spend).not.toContain('{{');
  expect(outfitConceptTitle('clean-lines-no-layer')).toBe(resources[lng].translation.outfitMaker.concepts['clean-lines'].title);
  expect(outfitConceptTitle('future-unknown-concept')).toBe(resources[lng].translation.outfitMaker.styledLook);
});

test('legacy prose remains compatible and versioned copy changes with actual garments', () => {
  expect(outfitDescription({description:'Legacy narrative'},items,context)).toBe('Legacy narrative');
  const changed = items.map(i => i.macroCategory === 'shoes' ? {...i,name:'Brown leather derby shoes'} : i);
  expect(outfitDescription({explanationVersion:1},changed,context)).toContain('Brown leather derby shoes');
  expect(outfitDescription({explanationVersion:1},changed,context)).not.toContain('Black leather Oxford shoes');
});

test('sparse date metadata cannot be rescued by lowering the quality floor', () => {
  const sparse = ['Cotton piqué polo shirt','Merino cardigan','Beige cotton chinos','Bexford Leather Oxfords - Black']
    .map((name,i)=>({id:String(i),name,macroCategory:['top','outerwear','bottom','shoes'][i]}));
  const result = assessOutfitQuality(sparse,'old_money','date',null,'3');
  expect(result.acceptance.minimum).toBe(0.75);
  expect(result.acceptance.passed).toBe(false);
});
