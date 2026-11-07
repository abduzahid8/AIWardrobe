/**
 * styleLanguage.ts — Color, proportion, comparison, explanation,
 * transformation, missing-piece, accessories (4K, §32–37).
 *
 * Garment-to-garment reasoning only: no body judgment, no shame-based
 * language, no pseudo-scientific personal color analysis. Comparisons say
 * "better for X", never universally better.
 */
import type { OutfitTransformation } from '../../types/visualOS';
import { scoreCompatibility, type CompatItem } from './compatibility';

export const COLOR_FAMILIES: Record<string, string[]> = {
  black: ['black', 'charcoal'], white: ['white', 'cream', 'ecru'],
  blue: ['navy', 'blue', 'denim', 'light-blue'], gray: ['grey', 'gray', 'charcoal'],
  brown: ['brown', 'tan', 'beige', 'camel'], green: ['green', 'olive', 'khaki'],
  red: ['red', 'burgundy', 'rust'], warm: ['yellow', 'orange', 'coral'],
};

export function colorFamily(color?: string): string {
  const c = (color || '').toLowerCase();
  for (const [fam, members] of Object.entries(COLOR_FAMILIES)) if (members.includes(c)) return fam;
  return c || 'unknown';
}

export function colorRelationship(a?: string, b?: string): 'monochrome' | 'neutral-bridged' | 'analogous' | 'contrasting' | 'unknown' {
  const fa = colorFamily(a);
  const fb = colorFamily(b);
  if (!a || !b || fa === 'unknown' || fb === 'unknown') return 'unknown';
  if (fa === fb) return 'monochrome';
  const neutrals = new Set(['black', 'white', 'gray']);
  if (neutrals.has(fa) || neutrals.has(fb)) return 'neutral-bridged';
  const warm = new Set(['red', 'brown', 'warm']);
  if ((warm.has(fa) && ['red', 'brown', 'warm'].includes(fb)) || (fa === 'blue' && fb === 'green') || (fa === 'green' && fb === 'blue')) return 'analogous';
  return 'contrasting';
}

/** Proportion notes: garment-to-garment balance only (§37). */
export function proportionNotes(items: readonly CompatItem[]): string[] {
  const notes: string[] = [];
  const outer = items.find((i) => i.category === 'outerwear');
  const bottom = items.find((i) => i.category === 'bottom');
  if (outer?.silhouette === 'long' && bottom?.silhouette === 'wide') {
    notes.push('A long outer layer over wide trousers hides the leg line; a cropped layer would restore proportion.');
  }
  if (items.some((i) => i.silhouette === 'oversized' && i.category === 'top') && bottom?.silhouette === 'wide') {
    notes.push('Oversized top with a wide bottom removes shape contrast; narrow the bottom or define the waist.');
  }
  const shoes = items.filter((i) => i.category === 'shoes');
  if (shoes.some((s) => s.visualWeight === 'heavy') && (bottom?.silhouette === 'slim' || bottom?.silhouette === 'cropped')) {
    notes.push('Heavy shoes anchor slim leg lines well; keep the top half clean to balance.');
  }
  return notes;
}

/** "Which is better?" — category winners, never absolute (§32). */
export function compareOutfits(
  a: { id: string; items: CompatItem[] },
  b: { id: string; items: CompatItem[] },
  ctx: { occasion?: string; tempC?: number } = {},
): Array<{ category: string; winner: string; note: string }> {
  const ca = scoreCompatibility(a.items, ctx);
  const cb = scoreCompatibility(b.items, ctx);
  const cats = [...new Set([...ca.dimensions.map((d) => d.key), ...cb.dimensions.map((d) => d.key)])];
  return cats.map((cat) => {
    const sa = ca.dimensions.find((d) => d.key === cat)?.score ?? 0;
    const sb = cb.dimensions.find((d) => d.key === cat)?.score ?? 0;
    if (Math.abs(sa - sb) < 0.05) return { category: cat, winner: 'tie', note: `Both outfits handle ${cat.replace(/_/g, ' ')} equally well.` };
    const winner = sa > sb ? a.id : b.id;
    return { category: cat, winner, note: `${winner === a.id ? 'First' : 'Second'} outfit is better for ${cat.replace(/_/g, ' ')} (${Math.max(sa, sb).toFixed(2)} vs ${Math.min(sa, sb).toFixed(2)}).` };
  });
}

/** Concise teaching explanations (§33). */
export function explainChoice(item: CompatItem, outfit: readonly CompatItem[]): string[] {
  const out: string[] = [];
  if (item.category === 'outerwear' && item.silhouette === 'cropped') {
    out.push('The shorter jacket preserves leg proportion against wider trousers.');
  }
  if (item.primaryColor && ['brown', 'tan'].includes(item.primaryColor.toLowerCase())) {
    out.push('The brown shoe softens contrast and leans the outfit more casual.');
  }
  if (item.pattern && item.pattern !== 'solid') {
    out.push('One patterned piece carries the visual interest; the rest stays quiet.');
  }
  if (!out.length) out.push('This piece repeats the outfit\u2019s existing palette and formality.');
  void outfit;
  return out;
}

export type TransformIntent =
  | 'more_formal' | 'more_casual' | 'cleaner' | 'warmer'
  | 'minimal' | 'expressive' | 'date_night' | 'presentation';

const INTENT_FORMALITY: Record<TransformIntent, number> = {
  more_formal: 0.8, presentation: 0.85, date_night: 0.7, more_casual: 0.3,
  cleaner: 0.6, warmer: 0.5, minimal: 0.5, expressive: 0.5,
};

/**
 * Transformation engine (4K): minimum edits, anchors preserved,
 * wardrobe-only, budget-aware via candidate prices.
 */
export function transformOutfit(
  current: readonly (CompatItem & { price?: number })[],
  wardrobe: readonly (CompatItem & { price?: number })[],
  intent: TransformIntent,
  options: { anchors?: string[]; budget?: number; wardrobeOnly?: boolean } = {},
): OutfitTransformation {
  const anchors = new Set(options.anchors || []);
  const target = INTENT_FORMALITY[intent];
  const editable = current.filter((c) => !anchors.has(c.id));
  // Rank current pieces by distance from target: worst first, one swap.
  const ranked = [...editable].sort((a, b) =>
    Math.abs((b.formality01 ?? 0.5) - target) - Math.abs((a.formality01 ?? 0.5) - target));
  const victim = ranked[0];
  let replacement: (typeof wardrobe)[number] | null = null;
  if (victim) {
    const cands = wardrobe.filter((w) =>
      w.id !== victim.id && w.category === victim.category &&
      !current.some((c) => c.id === w.id) &&
      (options.budget === undefined || (w.price ?? 0) <= options.budget));
    cands.sort((a, b) => Math.abs((a.formality01 ?? 0.5) - target) - Math.abs((b.formality01 ?? 0.5) - target));
    replacement = cands[0] || null;
  }
  const after = current.map((c) => (victim && replacement && c.id === victim.id ? replacement.id : c.id));
  return {
    beforeItemIds: current.map((c) => c.id),
    afterItemIds: after,
    swappedOut: victim && replacement ? [victim.id] : [],
    swappedIn: replacement ? [replacement.id] : [],
    rationale: replacement
      ? [`One swap toward "${intent}": ${victim!.id} → ${replacement.id}, closest to target formality ${target}.`, ...(options.wardrobeOnly ? ['Wardrobe-only: no purchase required.'] : [])]
      : ['No better wardrobe candidate found; outfit kept as-is rather than forcing a worse swap.'],
    preservedAnchors: [...anchors].filter((a) => after.includes(a)),
    confidence: replacement ? 'medium' : 'low',
  };
}

/** Missing-piece detection (§34): "nothing needed" is valid. */
export function missingPiece(items: readonly CompatItem[], occasion?: string): { suggestion: string | null; reason: string } {
  const cats = new Set(items.map((i) => i.category));
  if (!cats.has('shoes')) return { suggestion: 'shoes', reason: 'No footwear present; shoes complete the outfit.' };
  if (occasion === 'formal' && !cats.has('outerwear') && items.some((i) => (i.formality01 ?? 0) >= 0.7)) {
    return { suggestion: 'outerwear', reason: 'A structured layer would finish the formal look.' };
  }
  if (items.every((i) => !i.pattern || i.pattern === 'solid') && items.length >= 3) {
    return { suggestion: null, reason: 'Nothing needed: clean solid palette stands on its own.' };
  }
  return { suggestion: null, reason: 'Nothing needed: the outfit reads complete.' };
}

/** Accessory intelligence (§35): wardrobe-first, never overpowering. */
export function accessorySuggestion(
  items: readonly CompatItem[],
  accessories: readonly CompatItem[],
): { accessoryId: string | null; reason: string } {
  if (!accessories.length) return { accessoryId: null, reason: 'No accessories in wardrobe; nothing required.' };
  const patterned = accessories.filter((a) => a.pattern && a.pattern !== 'solid');
  const solid = accessories.filter((a) => !a.pattern || a.pattern === 'solid');
  const busy = items.filter((i) => i.pattern && i.pattern !== 'solid').length >= 1;
  const pick = busy ? solid[0] : patterned[0] || solid[0];
  if (!pick) return { accessoryId: null, reason: 'Nothing needed: accessories would overpower the outfit.' };
  return { accessoryId: pick.id, reason: busy ? 'Quiet solid accessory keeps focus on the patterned garment.' : 'One accent accessory adds interest without overpowering.' };
}
