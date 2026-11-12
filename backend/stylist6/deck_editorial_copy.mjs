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

export const DECK_COPY_VERSION = 's6-grounded-deck-copy-v1';
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
  if (part.top && part.bottom && known(part.top.color) && known(part.bottom.color))
    return `The ${part.top.color} top and ${part.bottom.color} trousers establish the palette.`;
  if (part.outerwear && part.top)
    return 'The outer layer and base form the upper-body combination.';
  return 'The pieces have passed the outfit compatibility checks.';
}

function focusDescription(part, direction) {
  const evidence = direction?.evidence || {};
  const structured = Array.isArray(evidence.structuralSlots) ? evidence.structuralSlots : [];
  const palette = Array.isArray(evidence.paletteSlots) ? evidence.paletteSlots : [];
  for (const slot of structured) {
    const d = part[slot];
    const itemName = role(d);
    if (itemName) return `The ${itemName} changes the outfit construction.`;
  }
  for (const slot of palette) {
    const d = part[slot];
    if (known(d?.color) && role(d)) return `The ${d.color} ${role(d)} shift the palette.`;
  }
  return '';
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

function shortCopy(part, context, direction) {
  const style = known(context?.style) ? titleCase(context.style) : 'your selected style';
  const occasion = known(context?.occasion) ? titleCase(context.occasion) : 'your occasion';
  const prefix = direction?.id === 'signature'
    ? `Closest to your ${style} direction for ${occasion}.`
    : direction?.id === 'refresh'
      ? `A different interpretation of ${style} for ${occasion}.`
      : direction?.id === 'discovery'
        ? `A new styling direction that still fits ${occasion}.`
        : `Another compatible option for ${occasion}; not a separate discovery direction.`;
  const change = direction?.qualified && direction?.id !== 'signature'
    ? focusDescription(part,direction) : '';
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
    const options = index === 0
      ? [native.title]
      : [...headlineCandidates(part,direction),native.title];
    let title = options.find(s=>s&& !seen.has(s.toLowerCase()));
    if (!title) title = `${native.title} · Option ${index+1}`;
    seen.add(title.toLowerCase());
    return {
      concept: {id:native.id,title,referenceId:look.referenceId??null},
      description: shortCopy(part,context,direction),
    };
  });
}
