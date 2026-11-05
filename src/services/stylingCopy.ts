import i18n from 'i18next';
import en from '../../i18n/locales/en.json';
import { stylingExplanationKeys } from '../domain/stylingExplanation';
import type { GarmentEvidence } from '../domain/garmentDescriptor';
import type { StylingContext } from '../domain/stylingContext';

function translate(key: string, values: Record<string, string> = {}): string {
  const fallback = en.outfitMaker.qualityExplanation[
    key.split('.').at(-1) as keyof typeof en.outfitMaker.qualityExplanation
  ];
  const translated = i18n.t(key, {...values, defaultValue: fallback});
  const text = typeof translated === 'string' && translated && translated !== key ? translated : fallback;
  return text.replace(/{{(\w+)}}/g, (_, name: string) => values[name] || '');
}

/** Product names stay authentic; descriptions and advice serve distinct purposes. */
export function stylingCopyParts(items: GarmentEvidence[], context: StylingContext): {description: string; tips: string[]} {
  const values = Object.fromEntries(['top', 'outerwear', 'bottom', 'shoes'].map(slot =>
    [slot, items.find(item => item.macroCategory === slot)?.name || '']));
  const description = translate('outfitMaker.qualityExplanation.garmentSummary', values);
  const tips = stylingExplanationKeys(items, context).map(key => translate(key));
  return {description, tips: [...new Set(tips)].filter(tip => tip !== description)};
}

export function stylingCopy(items: GarmentEvidence[], context: StylingContext): string[] {
  return stylingCopyParts(items, context).tips;
}
