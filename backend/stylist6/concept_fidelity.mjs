/**
 * Stylist 6 — auditable concept fidelity for complete catalog looks.
 * A look may be coherent yet not resemble the concept that nominated it.
 * This is a bounded ranking signal, never a source of invented stock, image,
 * color, preference, season, or a replacement for hard feasibility gates.
 */
import { describeGarment } from '../../src/domain/garmentDescriptor.ts';
import { colorTokensOf } from '../../src/domain/outfitDiversity.ts';

export const CONCEPT_FIDELITY_VERSION = 's6-concept-fidelity-v1';
const normalize = value => String(value || '').toLowerCase().trim().replace(/[\s_]+/g, '-');
const SLOT_NAMES = ['top','outerwear','bottom','shoes'];
/** Exact hue evidence lives in source names / verified color fields. The
 * descriptor's `light`, `mid`, `cool-dark` etc. are coarse TONE families and
 * must not be compared to a concept's `white`, `grey`, `navy` palette. */
const paletteTokens = (palette) => new Set((Array.isArray(palette) ? palette : [])
  .flatMap(value => colorTokensOf({ name: String(value) })));
export function verifiedPaletteMatch(item, palette) {
  const expected = palette instanceof Set ? palette : paletteTokens(palette);
  if (!expected.size) return null;
  const observed = colorTokensOf({
    color: typeof item?.color === 'string' ? item.color : '',
    name: typeof item?.name === 'string' ? item.name : '',
  });
  if (!observed.length) return null;
  // Match confirmed exact hue; no invented match from a color family.
  return observed.some(hue => expected.has(hue));
}
const clamp = (x, lo, hi) => Math.max(lo,Math.min(hi,x));

/**
 * Score HOW faithfully the already-admitted garments express their own
 * seed brief: garment roles (75%) and known, brief-supported colors (25%).
 * Missing metadata is neutral; user-fixed pieces are outside this score.
 */
export function evaluateConceptFidelity(items, brief, { requiredIds = [] } = {}) {
  if (!Array.isArray(items) || !brief?.roles || typeof brief.roles !== 'object') {
    return { version:CONCEPT_FIDELITY_VERSION, score:null, adjustment:0,
      roleFit:null, paletteFit:null, mismatchedSlots:[] };
  }
  const anchors = new Set(requiredIds.map(String));
  const palette = paletteTokens(brief.palette);
  let roleCompared=0, roleMatched=0, colorCompared=0, colorMatched=0;
  const mismatchedSlots=[];
  for (const item of items) {
    if (!item || anchors.has(String(item.id)) || !SLOT_NAMES.includes(item.macroCategory)) continue;
    const descriptor = describeGarment(item);
    const roles = brief.roles[item.macroCategory];
    if (Array.isArray(roles) && roles.length) {
      roleCompared++;
      const allowed = new Set(roles.map(normalize));
      if (allowed.has(normalize(descriptor.subtype)) || allowed.has(normalize(item.subtype))) roleMatched++;
      else mismatchedSlots.push(item.macroCategory);
    }
    const colorMatch = verifiedPaletteMatch(item,palette);
    if (colorMatch !== null) {
      colorCompared++;
      if (colorMatch) colorMatched++;
    }
  }
  const roleFit=roleCompared?roleMatched/roleCompared:null;
  const paletteFit=colorCompared?colorMatched/colorCompared:null;
  if (roleFit === null) return { version:CONCEPT_FIDELITY_VERSION, score:null,adjustment:0,
    roleFit, paletteFit, mismatchedSlots };
  // Without verifiable colors, leave palette neutral rather than inventing a
  // match. Do not force every garment to share the same palette token.
  const fit=.75*roleFit+.25*(paletteFit??.5);
  const adjustment=clamp((fit-.70)*.10,-.05,.03);
  return { version:CONCEPT_FIDELITY_VERSION,
    score:Number(fit.toFixed(4)),adjustment:Number(adjustment.toFixed(4)),
    roleFit:Number(roleFit.toFixed(4)),
    paletteFit:paletteFit===null?null:Number(paletteFit.toFixed(4)),mismatchedSlots };
}
