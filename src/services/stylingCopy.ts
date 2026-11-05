import i18n from 'i18next';
import en from '../../i18n/locales/en.json';
import { stylingExplanationKeys } from '../domain/stylingExplanation';
import type { GarmentEvidence } from '../domain/garmentDescriptor';
import type { StylingContext } from '../domain/stylingContext';
export function stylingCopy(items: GarmentEvidence[], context: StylingContext): string[] {
  return stylingExplanationKeys(items, context).map((key) => {
    const translated = i18n.t(key);
    return typeof translated === 'string' && translated && translated !== key
      ? translated
      : en.outfitMaker.qualityExplanation[
          key.split('.').at(-1) as keyof typeof en.outfitMaker.qualityExplanation
        ];
  });
}
