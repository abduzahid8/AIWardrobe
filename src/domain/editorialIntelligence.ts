import {hasColdLayerIntent} from './outfitComposition.ts';
import {
  describeGarment,
  type GarmentDescriptor,
  type GarmentEvidence,
} from './garmentDescriptor.ts';
import { scoreColorHarmony } from './colorHarmony.ts';
import { colorFamilyOf } from './outfitDiversity.ts';
import { currentVisual } from './visualFashion.ts';
import type { StylingContext } from './stylingContext.ts';

const clamp = (n: number) => Math.max(0, Math.min(1, n));
export interface PriceEvidence {
  price?: unknown;
  currency?: unknown;
  isShopItem?: boolean;
  isWardrobe?: boolean;
  pricePeers?: {
    median: number;
    count: number;
    normalizedPrice?: number;
    normalizedMedian?: number;
    normalizedCount?: number;
    p25?: number;
    p75?: number;
  };
}
/** Approved server-normalized peers when present, otherwise native-currency peers.
 * No guessed conversion, hidden budget or claim of quality from price.
 * Production peers cover the eligible catalog, not just the chosen shortlist. */
export function valueIntelligence<T extends GarmentEvidence>(items: T[], context: StylingContext) {
  const luxury =
    /\b(luxury|designer|premium|no budget limit|money is no object)\b|люкс|премиаль|без ограничений/i.test(
      context.request
    ) && !/\b(no|not|avoid|without|don.t want)\s+(?:any\s+|expensive\s+)?(?:luxury|designer|premium)\b|(?:не|без)\s+(?:люкс|премиаль)|hashamat(?:li)?\s+(?:emas|kerak emas)/i.test(context.request);
  const groups = new Map<string, number[]>();
  const key = (item: T, subtype = true) => {
    const p = item as T & PriceEvidence,
      d = describeGarment(item);
    return [d.slot, subtype ? d.subtype : '*', String(p.currency || '').toUpperCase()].join(':');
  };
  for (const item of items) {
    const p = item as T & PriceEvidence,
      price = Number(p.price);
    if (
      !p.isShopItem ||
      !Number.isFinite(price) ||
      price <= 0 ||
      !/^[A-Z]{3}$/.test(String(p.currency || ''))
    )
      continue;
    for (const k of [key(item), key(item, false)]) {
      const a = groups.get(k) || [];
      a.push(price);
      groups.set(k, a);
    }
  }
  for (const a of groups.values()) a.sort((a, b) => a - b);
  return (item: T) => {
    const p = item as T & PriceEvidence,
      price = Number(p.price),
      supplied = p.pricePeers;
    if (!p.isShopItem || !Number.isFinite(price) || price <= 0)
      return { score: !p.isShopItem && p.isWardrobe === true ? 0.75 : 0.5, ratio: null, peers: 0, luxury };
    const prices = (groups.get(key(item)) || []).length >= 4
      ? groups.get(key(item))! : groups.get(key(item, false)) || [];
    const native = !!supplied && Number.isFinite(supplied.median) && supplied.median > 0
      && Number.isInteger(supplied.count) && supplied.count >= 4;
    const normalized = !!supplied && Number.isFinite(supplied.normalizedPrice)
      && Number(supplied.normalizedPrice) > 0 && Number.isFinite(supplied.normalizedMedian)
      && Number(supplied.normalizedMedian) > 0 && Number.isInteger(supplied.normalizedCount)
      && Number(supplied.normalizedCount) >= 4;
    const middle = Math.floor(prices.length / 2);
    const localMedian = prices.length % 2 ? prices[middle] : (prices[middle - 1] + prices[middle]) / 2;
    const median = native ? supplied!.median : localMedian;
    const count = native ? supplied!.count : prices.length;
    if (!normalized && (count < 4 || !Number.isFinite(median) || median <= 0))
      return { score: 0.5, ratio: null, peers: count, luxury };
    const ratio = normalized ? Number(supplied!.normalizedPrice) / Number(supplied!.normalizedMedian) : price / median;
    // A tenfold price leap needs intent; cheaper is not always better. Explicit
    // luxury removes the ordinary-intent penalty without forcing costly products.
    return {
      score: luxury ? 0.5 : clamp(0.75 - Math.log2(Math.max(1, ratio)) * 0.22),
      ratio,
      peers: normalized ? Number(supplied!.normalizedCount) : count,
      luxury,
    };
  };
}

/** Soft relationship graph. Missing observations stay neutral. These scores
 * rank coherent candidates; existing source/dress/budget gates retain authority. */
export function garmentRelationship(
  a: GarmentDescriptor,
  b: GarmentDescriptor,
  context: StylingContext
): number {
  let shape = 0.65,
    texture = 0.65,
    register = 0.65,
    purpose = 0.7;
  const ranks: Record<string, number> = { athletic: 0, casual: 1, smart: 2, formal: 3 };
  if (a.formality !== 'unknown' && b.formality !== 'unknown')
    register = clamp(1 - Math.abs(ranks[a.formality] - ranks[b.formality]) * 0.28);
  if (a.volume !== 'unknown' && b.volume !== 'unknown')
    shape = a.volume === b.volume ? 0.9 : [a.volume, b.volume].includes('regular') ? 0.85 : 0.45;
  if (a.bulk !== 'unknown' && b.bulk !== 'unknown') texture = a.bulk === b.bulk ? 0.9 : 0.55;
  if (a.texture !== 'unknown' && b.texture !== 'unknown')
    texture = (texture + (a.texture === b.texture ? 0.9 : 0.75)) / 2;
  if (
    (a.insulated && b.materials.includes('linen')) ||
    (b.insulated && a.materials.includes('linen'))
  )
    texture = Math.min(texture, 0.2);
  if (
    (a.insulated || b.insulated) &&
    context.dressCode === 'smart' &&
    !hasColdLayerIntent(context.weather)
  )
    purpose = 0.35;
  const upper = ['top', 'outerwear'].includes(a.slot) ? a : ['top', 'outerwear'].includes(b.slot) ? b : null;
  const lower = a.slot === 'bottom' ? a : b.slot === 'bottom' ? b : null;
  if (upper && lower && upper.volume === 'wide' && lower.volume === 'wide'
    && (upper.bulk === 'heavy' || lower.bulk === 'heavy')
    && context.style !== 'streetwear' && !/oversized|wide silhouette|объ[её]м/i.test(context.request))
    shape = Math.min(shape, 0.5);
  if (a.patternProminence === 'prominent' && b.patternProminence === 'prominent'
    && a.pattern !== 'solid' && b.pattern !== 'solid' && a.pattern !== 'unknown' && b.pattern !== 'unknown')
    texture = Math.min(texture, 0.45);
  const shoe = a.slot === 'shoes' ? a : b.slot === 'shoes' ? b : null,
    bottom = a.slot === 'bottom' ? a : b.slot === 'bottom' ? b : null;
  if (shoe && bottom) {
    if (bottom.volume === 'wide' && shoe.soleProfile === 'dress')
      shape = Math.min(shape, 0.55);
    if (bottom.volume === 'slim' && shoe.soleProfile === 'rugged') shape = Math.min(shape, 0.4);
    if (
      /\b(travel|trip|sightseeing)\b/.test(`${context.occasion} ${context.request}`) &&
      !/\b(hiking|trail|training|basketball|gym)\b/.test(context.request) &&
      !(context.weather?.temp != null && context.weather.temp < 5 && /snow|ice|снег|л[её]д/i.test(context.weather.condition || ''))
    ) {
      if (shoe.trainingFootwear || shoe.rugged || shoe.athletic) purpose = 0.25;
      else if (['minimal-sneaker', 'sneaker'].includes(shoe.subtype)) purpose = 0.85;
    }
  }
  const harmony = scoreColorHarmony(
    [a, b].map((d) => ({
      colors: d.color === 'unknown' ? [] : [d.color],
      colorFamily: colorFamilyOf({ color: d.color }),
      pattern: d.pattern,
    }))
  );
  return 0.24 * shape + 0.2 * texture + 0.22 * register + 0.2 * harmony.score + 0.14 * purpose;
}
export function lookRelationships(items: GarmentEvidence[], context: StylingContext): number {
  const d = items.map(describeGarment);
  let sum = 0, weakest = 1,
    n = 0;
  for (let a = 0; a < d.length; a++)
    for (let b = a + 1; b < d.length; b++) {
      const relationship = garmentRelationship(d[a], d[b], context);
      sum += relationship;
      weakest = Math.min(weakest, relationship);
      n++;
    }
  const vectors = items
    .map((item) => item.visualEmbedding)
    .filter(
      (v): v is readonly number[] =>
        Array.isArray(v) && v.length === 768 && v.every(n => Number.isFinite(n) && Math.abs(n) <= 1)
    );
  let similarity = 0,
    pairs = 0;
  for (let a = 0; a < vectors.length; a++)
    for (let b = a + 1; b < vectors.length; b++) {
      similarity += vectors[a].reduce((total, x, i) => total + x * vectors[b][i], 0);
      pairs++;
    }
  // Image similarity is weak supporting evidence, never proof of fashion compatibility.
  const graph = n ? 0.7 * sum / n + 0.3 * weakest : 0.5;
  return graph * 0.97 + (pairs ? clamp(similarity / pairs) : 0.5) * 0.03;
}
export function imageEvidenceStrength(item: GarmentEvidence): number {
  const p = currentVisual(item);
  if (!p) return 0;
  return Math.min(
    1,
    describeGarment(item).evidence.filter((e) => e.startsWith('image:')).length / 6
  );
}
/** Actual garment relationships, never the intended concept label or product ID. */
export function editorialIdentity(items: GarmentEvidence[]): string {
  const d = items.map(describeGarment),
    at = (s: string) => d.find((i) => i.slot === s);
  const top = at('top'),
    layer = at('outerwear'),
    bottom = at('bottom'),
    shoe = at('shoes');
  const shoeRole = shoe?.athletic
    ? 'sport'
    : shoe?.rugged
      ? 'rugged'
      : shoe?.casualSole
        ? 'sneaker'
        : shoe?.subtype === 'loafer'
          ? 'loafer'
          : 'dress';
  return [
    top && ['shirt','dress-shirt'].includes(top.subtype) ? 'woven-shirt' : top?.subtype || 'unknown',
    layer?.subtype || 'no-layer',
    bottom?.subtype || 'unknown',
    bottom && ['slim','wide'].includes(bottom.volume) ? bottom.volume : 'unspecified-line',
    shoeRole,
  ].join(':');
}
