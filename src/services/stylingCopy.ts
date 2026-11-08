import { lookExplanationFacts } from '../domain/lookExplanation';
import i18n from 'i18next';
import en from '../../i18n/locales/en.json';
import { stylingExplanationKeys } from '../domain/stylingExplanation';
import type { GarmentEvidence } from '../domain/garmentDescriptor';
import type { StylingContext } from '../domain/stylingContext';

function translate(key: string, values: Record<string, string> = {}): string {
  const fallback =
    en.outfitMaker.qualityExplanation[
      key.split('.').at(-1) as keyof typeof en.outfitMaker.qualityExplanation
    ];
  const translated = i18n.t(key, { ...values, defaultValue: fallback });
  const text =
    typeof translated === 'string' && translated && translated !== key ? translated : fallback;
  return (text || '').replace(/{{(\w+)}}/g, (_, name: string) => values[name] || '');
}

/** Product names stay authentic; descriptions and advice serve distinct purposes. */
export function stylingCopyParts(
  items: GarmentEvidence[],
  context: StylingContext
): { description: string; tips: string[] } {
  const values = Object.fromEntries(
    ['top', 'outerwear', 'bottom', 'shoes'].map((slot) => [
      slot,
      items.find((item) => item.macroCategory === slot)?.name || '',
    ])
  );
  const description = translate('outfitMaker.qualityExplanation.garmentSummary', values);
  const tips = stylingExplanationKeys(items, context).map((key) => translate(key));
  return { description, tips: [...new Set(tips)].filter((tip) => tip !== description) };
}

export function stylingCopy(items: GarmentEvidence[], context: StylingContext): string[] {
  return stylingCopyParts(items, context).tips;
}

/** One localized, product-specific paragraph. Only known relationships are stated. */
export function editorialCopy(items: GarmentEvidence[], context: StylingContext): string {
  const facts = lookExplanationFacts(items, context);
  const values = {
    top: facts.top,
    layer: facts.layer,
    bottom: facts.bottom,
    shoes: facts.shoes,
    palette: facts.palette
      .split(' / ')
      .map((color) =>
        String(i18n.t('outfitMaker.editorialColors.' + color, { defaultValue: color }))
      )
      .join(' / '),
    shape: translate(`outfitMaker.qualityExplanation.shape_${facts.shape}`),
    texture: translate(`outfitMaker.qualityExplanation.texture_${facts.texture}`),
  };
  const relationship = facts.tailoredFootwear ? 'editorialTailoredFootwear'
    : facts.lightOpenLayer ? 'editorialLightBase'
    : facts.textureContrast ? 'editorialTextureContrast'
    : facts.casualFootwear ? 'editorialCasualFootwear'
    : facts.layer ? 'editorialLayered' : 'editorialUnlayered';
  const sentences = [translate(`outfitMaker.qualityExplanation.${relationship}`, values)];
  if (facts.palette && facts.knownColors >= 2)
    sentences.push(translate('outfitMaker.qualityExplanation.paletteStory', values));
  if (facts.shape !== 'unknown')
    sentences.push(translate('outfitMaker.qualityExplanation.shapeStory', values));
  return sentences.join(' ');
}

/** Version 1 describes grounded facts, not server-localized prose. Render those
 * facts in the active locale; legacy free-form descriptions remain compatible. */
export function outfitDescription(
  outfit: { explanationVersion?: number; description?: unknown },
  items: GarmentEvidence[],
  context: StylingContext
): string {
  if (outfit.explanationVersion === 1) return editorialCopy(items, context);
  return typeof outfit.description === 'string' && outfit.description.trim()
    ? outfit.description
    : stylingCopyParts(items, context).description;
}

/** Variant IDs are planner details; their base concept owns localized copy. */
export function outfitConceptTitle(id: string): string {
  const baseId = id.replace(/-no-layer$/, '');
  return String(i18n.t(`outfitMaker.concepts.${baseId}.title`, {
    defaultValue: i18n.t('outfitMaker.styledLook', {defaultValue:'Styled look'}),
  }));
}
