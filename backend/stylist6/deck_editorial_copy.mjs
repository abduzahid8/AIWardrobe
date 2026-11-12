/**
 * Honest, compact styling copy for Stylist 6 New Clothing.
 * One deck can legitimately contain three looks mapped to the same native
 * concept ("Refined Tailoring"). A repeated concept is not a repeated outfit,
 * but showing identical headlines obscures what the stylist changed.
 *
 * Only request context, already-approved selected items and measured
 * direction differences are used. No images, location, body shape, skin
 * color, stock, comfort or personal taste are inferred.
 */
import { describeGarment } from '../../src/domain/garmentDescriptor.ts';
import { lookConcept } from '../../src/domain/outfitConcepts.ts';

export const DECK_COPY_VERSION = 's6-grounded-deck-copy-v3';
const slotOrder = ['top','outerwear','bottom','shoes'];
const known = x => typeof x === 'string' && !!x &&
  !['unknown','neutral','missing'].includes(x.toLowerCase());
const titleCase = x => String(x || '').replace(/[-_]+/g,' ')
  .replace(/\b\w/g,letter => letter.toUpperCase());

const role = d => {
  if (!d) return '';
  const names = {
    'top': { 'fine-knit':'knitwear','t-shirt':'tee','shirt':'shirt','dress-shirt':'dress shirt',
      'polo':'polo','knit-polo':'knit polo','sweater':'sweater' },
    'outerwear': { 'blazer':'blazer','cardigan':'cardigan','coat':'coat','jacket':'jacket',
      'overshirt':'overshirt','bomber':'bomber' },
    'bottom': { 'chino':'chinos','jeans':'jeans','tailored-trouser':'tailored trousers',
      'trouser':'trousers','shorts':'shorts' },
    'shoes': { 'loafer':'loafers','derby':'derbies','oxford':'oxfords',
      'chelsea':'Chelsea boots','minimal-sneaker':'minimal sneakers',
      'sneaker':'sneakers','sport-sneaker':'sport sneakers','dress-boot':'dress boots' },
  };
  return names[d.slot]?.[d.subtype] || '';
};

function details(items) {
  const descriptors = (items || []).map(describeGarment);
  return Object.fromEntries(slotOrder.map(slot => [slot,
    descriptors.find(d => d.slot === slot) || null]));
}

function actualRelationship(part) {
  if (part.bottom?.structured && part.shoes &&
      ['loafer','oxford','derby','monk','dress-boot','chelsea'].includes(part.shoes.subtype) &&
      !part.shoes.casualSole)
    return 'Dress footwear follows the tailored trouser line.';
  if (part.outerwear?.subtype === 'cardigan' && part.top)
    return 'The cardigan creates an open layer over the base.';
  if (part.outerwear?.subtype === 'blazer' && part.bottom?.structured)
    return 'A blazer and structured trousers keep the tailoring consistent.';
  if (part.top && part.bottom && known(part.top.color) && known(part.bottom.color)) {
    const top = role(part.top) || 'top';
    const bottom = role(part.bottom) || 'bottom';
    return `A ${part.top.color} ${top} with ${part.bottom.color} ${bottom} defines the palette.`;
  }
  if (part.outerwear && part.top)
    return 'The outer layer and base form the upper-body combination.';
  return ''; // Unverified occasion-fit praise would add no garment evidence.
}

/**
 * Specific comparisons require knowledge of both the lead and alternative.
 * Never claim a change in shape or texture just because two SKU names differ.
 */
function focusDescription(basePart, part, direction) {
  const evidence = direction?.evidence || {};
  const structural = Array.isArray(evidence.structuralSlots) ? evidence.structuralSlots : [];
  const palette = Array.isArray(evidence.paletteSlots) ? evidence.paletteSlots : [];
  const clean = value => known(value) ? titleCase(value) : '';
  for (const slot of structural) {
    const before = basePart?.[slot], after = part?.[slot];
    if (!before && after && role(after)) return `This version adds ${role(after)} as a layer.`;
    if (before && !after && role(before)) return `This version removes the ${role(before)} layer.`;
    if (!before || !after) continue;
    const from = role(before), to = role(after);
    if (from && to && from !== to) return `The ${to} replaces the ${from}, changing the outfit structure.`;
    if (from && known(before.volume) && known(after.volume) && before.volume !== after.volume)
      return `The ${from} changes from a ${clean(before.volume)} to a ${clean(after.volume)} silhouette.`;
    if (from && known(before.texture) && known(after.texture) && before.texture !== after.texture)
      return `The ${from} changes from ${clean(before.texture)} to ${clean(after.texture)} texture.`;
    if (from && known(before.pattern) && known(after.pattern) && before.pattern !== after.pattern)
      return `The ${from} changes from ${clean(before.pattern)} to ${clean(after.pattern)} patterning.`;
  }
  for (const slot of palette) {
    const before = basePart?.[slot], after = part?.[slot];
    if (before && after && known(before.color) && known(after.color) &&
        before.color !== after.color && role(after))
      return `The ${role(after)} shifts from ${clean(before.color)} to ${clean(after.color)}.`;
  }
  return '';
}

/**
 * The native concept may describe an optional knit layer even when the
 * final three-slot outfit contains only a polo or sweater. Do not display
 * "layering" unless there is actually an outer layer in the selected SKUs.
 */
export function groundedNativeHeadline(nativeTitle, part) {
  const title = String(nativeTitle || '');
  if (part?.outerwear || !/\blayer(?:s|ing)?\b/i.test(title)) return title;
  const top = role(part?.top), bottom = role(part?.bottom);
  return top && bottom ? `${titleCase(top)} & ${titleCase(bottom)}` : 'Clean separates';
}

function headlineCandidates(part, direction) {
  const e = direction?.evidence || {};
  const focusSlots = [...new Set([
    ...(Array.isArray(e.structuralSlots) ? e.structuralSlots : []),
    ...(Array.isArray(e.paletteSlots) ? e.paletteSlots : []),
    ...slotOrder,
  ])];
  const result = [];
  for (const slot of focusSlots) {
    const d = part[slot];
    if (!role(d)) continue;
    if (known(d.color)) result.push(`${titleCase(d.color)} ${titleCase(role(d))}`);
    result.push(titleCase(role(d)));
  }
  const top = part.top, bottom = part.bottom, layer = part.outerwear;
  if (top && bottom && role(top) && role(bottom)) {
    result.push(`${titleCase(role(top))} & ${titleCase(role(bottom))}`);
  }
  if (layer && bottom && role(layer) && role(bottom)) {
    result.push(`${titleCase(role(layer))} & ${titleCase(role(bottom))}`);
  }
  return [...new Set(result)];
}

function shortCopy(basePart, part, context, direction) {
  const style = known(context?.style) ? titleCase(context.style) : 'your selected style';
  const occasion = known(context?.occasion) ? (/^date(?:[_ -]night)?$/i.test(context.occasion) ? 'date night' : titleCase(context.occasion)) : 'your occasion';
  const prefix = direction?.id === 'signature'
    ? `Closest to your ${style} direction for ${occasion}.`
    : direction?.id === 'refresh'
      ? `A different interpretation of ${style} for ${occasion}.`
      : direction?.id === 'discovery'
        ? `A new styling direction that still fits ${occasion}.`
        : `Another compatible option for ${occasion}; not a separate discovery direction.`;
  const change = direction?.qualified && direction?.id !== 'signature'
    ? focusDescription(basePart,part,direction) : '';
  return [prefix,change,actualRelationship(part)].filter(Boolean).join(' ');
}

/** Assign each actual selected look a title + explanation, preserving order
 * and keeping a unique headline even when all native concept IDs are equal. */
export function editorialDeckCopy(looks, context, directions) {
  const seen = new Set();
  return (looks || []).map((look,index) => {
    const native = lookConcept(look.items,context).concept;
    const direction = directions?.[index] || {id:index?'alternative':'signature'};
    const part = details(look.items);
    const truthfulNative = groundedNativeHeadline(native.title, part);
    const options = index === 0
      ? [truthfulNative]
      : [...headlineCandidates(part,direction),truthfulNative];
    let title = options.find(s=>s&& !seen.has(s.toLowerCase()));
    if (!title) title = `${truthfulNative} · Option ${index+1}`;
    seen.add(title.toLowerCase());
    return {
      concept: {id:native.id,title,referenceId:look.referenceId??null},
      description: shortCopy(details(looks[0]?.items || []),part,context,direction),
    };
  });
}
