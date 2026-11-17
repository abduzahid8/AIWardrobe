/**
 * compatibility.ts — Outfit visual compatibility + diagnostics (4I, 4J).
 *
 * Deterministic hybrid: each dimension scored separately from structured
 * metadata, never collapsed into one opaque score. Structured constraints
 * (occasion/weather) are hard conflicts, not suggestions. Diagnoses are
 * actionable and hedged — never stated as universal laws.
 */
import type {
  CompatibilityDimension,
  OutfitCompatibility,
  OutfitDiagnosis,
} from '../../types/visualOS';

export interface CompatItem {
  id: string;
  category: string;
  primaryColor?: string;
  pattern?: string;
  formality01?: number;
  silhouette?: string;
  visualWeight?: 'light' | 'medium' | 'heavy';
}

export interface CompatContext {
  occasion?: string;
  tempC?: number;
  condition?: string;
  dnaFormality?: number;
}

const NEUTRALS = new Set(['black', 'white', 'grey', 'gray', 'navy', 'beige', 'brown', 'cream', 'charcoal']);

function dim(key: string, score: number, note: string, confidence: CompatibilityDimension['confidence'] = 'medium'): CompatibilityDimension {
  return { key, score: Math.round(score * 100) / 100, note, confidence };
}

export function scoreCompatibility(
  items: readonly CompatItem[],
  ctx: CompatContext = {},
): OutfitCompatibility {
  const dims: CompatibilityDimension[] = [];
  const strengths: string[] = [];
  const tensions: string[] = [];
  const hardConflicts: string[] = [];
  const improvements: string[] = [];

  // Palette harmony
  const colors = items.map((i) => (i.primaryColor || 'unknown').toLowerCase());
  const distinct = [...new Set(colors.filter((c) => c !== 'unknown'))];
  const nonNeutral = distinct.filter((c) => !NEUTRALS.has(c));
  const paletteScore = nonNeutral.length <= 1 ? 0.9 : nonNeutral.length === 2 ? 0.65 : 0.4;
  dims.push(dim('palette_harmony', paletteScore,
    nonNeutral.length <= 1 ? 'Cohesive restrained palette' : `${nonNeutral.length} strong colors compete`,
    distinct.length ? 'medium' : 'low'));
  if (paletteScore >= 0.8) strengths.push('Restrained, cohesive palette.');
  else { tensions.push('Multiple strong colors compete for attention.'); improvements.push('Keep one accent color; neutralize the rest.'); }

  // Formality coherence
  const forms = items.map((i) => i.formality01 ?? 0.5);
  const spread = Math.max(...forms) - Math.min(...forms);
  const formScore = spread <= 0.25 ? 0.9 : spread <= 0.45 ? 0.6 : 0.3;
  dims.push(dim('formality_coherence', formScore, `Formality spread ${spread.toFixed(2)}`));
  if (formScore < 0.5) {
    tensions.push('Formality levels clash across pieces.');
    improvements.push('Align shoes and outer layer to the outfit\u2019s formality anchor.');
  } else strengths.push('Consistent formality level.');

  // Footwear compatibility (shoe formality vs outfit anchor)
  const shoes = items.filter((i) => i.category === 'shoes');
  const tops = items.filter((i) => i.category === 'top' || i.category === 'dress');
  if (shoes.length && tops.length) {
    const gap = Math.abs((shoes[0].formality01 ?? 0.5) - (tops[0].formality01 ?? 0.5));
    dims.push(dim('footwear_compatibility', gap <= 0.3 ? 0.9 : 0.4,
      gap <= 0.3 ? 'Shoes match outfit formality' : 'Shoe formality mismatches the outfit anchor'));
    if (gap > 0.3) { tensions.push('Shoe formality mismatches the top layer.'); improvements.push('Swap to shoes within one formality step of the outfit anchor.'); }
  }

  // Silhouette balance / visual weight
  const weights = items.map((i) => i.visualWeight || 'medium');
  const heavy = weights.filter((w) => w === 'heavy').length;
  const light = weights.filter((w) => w === 'light').length;
  const silScore = heavy > 0 && light > 0 && heavy + light === items.length ? 0.5 : 0.8;
  dims.push(dim('silhouette_balance', silScore,
    heavy > 0 && light === items.length - heavy && heavy < items.length
      ? 'Mixed visual weights need balancing' : 'Balanced visual weight'));
  if (silScore < 0.7) { tensions.push('Visual weight is uneven across pieces.'); improvements.push('Add a structured middle layer to bridge heavy and light pieces.'); }

  // Pattern/texture
  const patterned = items.filter((i) => i.pattern && i.pattern !== 'solid').length;
  dims.push(dim('texture_pattern_balance', patterned <= 1 ? 0.9 : 0.45,
    patterned <= 1 ? 'Pattern discipline held' : `${patterned} patterns compete`));
  if (patterned > 1) { tensions.push('Too many focal patterns.'); improvements.push('Keep one patterned piece; render the rest solid.'); }

  // Context / climate — hard conflicts from structured constraints
  if (ctx.occasion) {
    const formalCtx = ['formal', 'office', 'interview'].includes(ctx.occasion);
    const avgForm = forms.reduce((a, b) => a + b, 0) / Math.max(1, forms.length);
    dims.push(dim('context_suitability', formalCtx ? (avgForm >= 0.55 ? 0.9 : 0.3) : 0.8,
      `Occasion: ${ctx.occasion}`, 'high'));
    if (formalCtx && avgForm < 0.55) hardConflicts.push(`Too casual for ${ctx.occasion}.`);
  }
  if (ctx.tempC !== undefined) {
    const hasOuter = items.some((i) => i.category === 'outerwear');
    if (ctx.tempC < 8 && !hasOuter) hardConflicts.push('No outer layer for cold weather.');
    dims.push(dim('climate_suitability', hardConflicts.length ? 0.3 : 0.85,
      ctx.tempC < 8 ? (hasOuter ? 'Cold covered' : 'Cold exposure') : 'Climate OK', 'high'));
  }

  const scored = dims.filter((d) => d.confidence !== 'low');
  const avg = scored.length ? scored.reduce((a, d) => a + d.score, 0) / scored.length : 0;
  return {
    dimensions: dims,
    strengths,
    tensions,
    hardConflicts,
    improvements,
    confidence: dims.every((d) => d.confidence !== 'low') ? (avg >= 0.7 ? 'high' : 'medium') : 'low',
    provenance: ['deterministic-visual-compatibility-v1'],
  };
}

/** "Why does this outfit not work?" — actionable diagnoses (4J). */
export function diagnoseOutfit(
  items: readonly CompatItem[],
  ctx: CompatContext = {},
): OutfitDiagnosis[] {
  const comp = scoreCompatibility(items, ctx);
  const out: OutfitDiagnosis[] = [];
  // Valid outfits return no forced criticism.
  if (!comp.hardConflicts.length && !comp.tensions.length) return out;
  for (const c of comp.hardConflicts) {
    out.push({
      issue: c.includes('casual') ? 'context mismatch' : 'climate mismatch',
      detail: c,
      actionableFix: c.includes('casual')
        ? 'Raise formality with the shoes or add a structured layer.'
        : 'Add an outer layer rated for the temperature.',
      confidence: 'high',
    });
  }
  const shoes = items.filter((i) => i.category === 'shoes');
  const anchor = items.filter((i) => i.category !== 'shoes').map((i) => i.formality01 ?? 0.5);
  if (shoes.length && anchor.length && Math.abs((shoes[0].formality01 ?? 0.5) - anchor[0]) > 0.35) {
    out.push({
      issue: 'shoe formality mismatch',
      detail: 'The shoes sit more than a step away from the outfit\u2019s formality anchor.',
      actionableFix: 'Choose shoes within one formality step of the top half.',
      confidence: 'medium',
    });
  }
  const sils = [...new Set(items.map((i) => i.silhouette || 'unknown').filter((s) => s !== 'unknown'))];
  if (sils.length >= 3) {
    out.push({
      issue: 'competing silhouettes',
      detail: 'Three or more distinct silhouettes pull in different visual directions.',
      actionableFix: 'Repeat one silhouette language across two pieces.',
      confidence: 'low',
    });
  }
  if (comp.tensions.some((t) => t.includes('strong colors'))) {
    out.push({
      issue: 'palette conflict',
      detail: 'Multiple saturated colors compete without a neutral bridge.',
      actionableFix: 'Keep the strongest color and neutralize the rest.',
      confidence: 'medium',
    });
  }
  return out.slice(0, 5);
}
