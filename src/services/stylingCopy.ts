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
  // A second sentence is added only when it contributes grounded information
  // the first sentence did not state: a verified multi-color palette first,
  // otherwise the verified trouser silhouette. Never a third recitation.
  if (facts.palette && facts.knownColors >= 2)
    sentences.push(translate('outfitMaker.qualityExplanation.paletteStory', values));
  else if (values.shape)
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

/** Truthful ranking provenance. Only a fully accepted provider ordering is
 * 'ai'; a repaired ordering is 'hybrid'; everything else is fallback.
 * Unknown and legacy values stay fallback, never AI. */
export function resolveSelectionSource(
  source: unknown
): 'ai' | 'hybrid' | 'deterministic_fallback' {
  return source === 'ai' ? 'ai'
    : source === 'hybrid' ? 'hybrid'
    : 'deterministic_fallback';
}

/** Card badge copy key. The AI badge renders for genuine provider acceptance
 * only; hybrid and fallback looks use the neutral curated label. */
export function selectionBadgeKey(source: unknown): string {
  return resolveSelectionSource(source) === 'ai'
    ? 'outfitMaker.aiRanked'
    : 'outfitMaker.styledLook';
}

/** Variant IDs are planner details; their base concept owns localized copy.
 * An unknown concept keeps the server-supplied title when present; only an
 * unidentifiable look falls back to the generic localized label. */
export function outfitConceptTitle(id: string, fallbackTitle?: string): string {
  const baseId = String(id || '').replace(/-no-layer$/, '');
  const key = `outfitMaker.concepts.${baseId}.title`;
  if (baseId && i18n.exists(key)) return String(i18n.t(key));
  const fallback = typeof fallbackTitle === 'string' ? fallbackTitle.trim() : '';
  return fallback || String(i18n.t('outfitMaker.styledLook', {defaultValue:'Styled look'}));
}

/**
 * The Stylist 6 server returns item-specific, unique titles. A known native
 * concept ID must not overwrite that title with its generic translation
 * ("Refined Tailoring" on multiple cards). Legacy engines keep the existing
 * localized concept name; absent S6 server copy keeps the normal fallback.
 */
export function outfitConceptDisplayTitle(
  id: string, suppliedTitle?: string, engineVersion?: string,
): string {
  const title = typeof suppliedTitle === 'string' ? suppliedTitle.trim() : '';
  if (/^stylist-6-/i.test(String(engineVersion || '')) && title) return title;
  return outfitConceptTitle(id, suppliedTitle);
}

/** Advice that adds information beyond the description: no empties, no
 * case/whitespace-only duplicates, and never the description restated. */
export function distinctStylingTips(description: string, tips: readonly unknown[]): string[] {
  const key = (value: string): string => value.replace(/\s+/g, ' ').trim().toLowerCase();
  const descriptionKey = key(String(description || ''));
  const seen = new Set<string>();
  return (tips || [])
    .filter((tip): tip is string => typeof tip === 'string')
    .map((tip) => tip.replace(/\s+/g, ' ').trim())
    .filter((tip) => {
      if (!tip) return false;
      const tipKey = key(tip);
      if (tipKey === descriptionKey || seen.has(tipKey)) return false;
      seen.add(tipKey);
      return true;
    });
}
