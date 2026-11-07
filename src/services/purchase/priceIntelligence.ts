import type {
  MerchantOffer,
  PriceObservation,
  PriceAssessment,
  TruePrice,
  PriceWatch,
  PriceComponent,
} from '../../types/purchase';
import type { BudgetFxContext } from '../../../backend/supabase/functions/_shared/budgetFx';
import { convertShopAmount } from '../../utils/outfitBudget';
const DAY = 86400000;
const finitePrice = (value: number) => Number.isFinite(value) && value >= 0;
export function truePrice(offer: MerchantOffer, now = Date.now()): TruePrice {
  const components: TruePrice['components'] = {
    item: {
      amount: offer.currentPrice,
      currency: offer.currency,
      source: offer.url,
      confidence: offer.merchantConfidence,
      observedAt: offer.lastSeenAt,
    },
  };
  const unknown: string[] = [];
  let subtotal = 0;
  for (const key of ['item', 'shipping', 'tax', 'fees', 'discount', 'coupon'] as const) {
    const component = key === 'item' ? components.item : offer[key];
    if (!component) {
      if (['shipping', 'tax', 'fees'].includes(key)) unknown.push(key);
      continue;
    }
    if (
      !finitePrice(component.amount) ||
      component.currency !== offer.currency ||
      !Number.isFinite(Date.parse(component.observedAt)) ||
      Date.parse(component.observedAt) > now ||
      !component.source
    ) {
      unknown.push(key);
      continue;
    }
    if (key === 'item' && now - Date.parse(component.observedAt) > 7 * DAY) {
      unknown.push('item');
      continue;
    }
    components[key] = component;
    subtotal += (key === 'discount' || key === 'coupon' ? -1 : 1) * component.amount;
  }
  if (!offer.currency || !finitePrice(offer.currentPrice)) unknown.push('item');
  const unique = [...new Set(unknown)];
  return {
    currency: offer.currency,
    knownSubtotal: Math.max(0, subtotal),
    total: unique.length ? undefined : Math.max(0, subtotal),
    unknownComponents: unique,
    components,
    confidence: unique.length
      ? 'low'
      : Object.values(components).every((c) => c?.confidence === 'high')
        ? 'high'
        : 'medium',
  };
}
export function assessPrice(
  offer: MerchantOffer,
  observations: readonly PriceObservation[],
  options: { now?: number; windowDays?: 30 | 60 | 90; waitIfHigh?: boolean } = {}
): PriceAssessment {
  if (offer.quoteKind === 'estimate')
    return {
      state: 'INSUFFICIENT_HISTORY',
      quoteFresh: false,
      confidence: 'low',
      sampleCount: 0,
      spanDays: 0,
      evidence: ['Catalog/manual estimate; merchant freshness and price history are unverified'],
    };
  const now = options.now || Date.now(),
    window = options.windowDays || 90;
  const records = observations
    .slice(-1000)
    .filter(
      (o) =>
        o.offerId === offer.id &&
        o.currency === offer.currency &&
        finitePrice(o.price) &&
        o.price > 0 &&
        Date.parse(o.observedAt) <= now &&
        Date.parse(o.observedAt) >= now - window * DAY
    )
    .sort((a, b) => Date.parse(a.observedAt) - Date.parse(b.observedAt));
  const unique = [
    ...new Map(records.map((r) => [new Date(r.observedAt).toISOString().slice(0, 10), r])).values(),
  ];
  const span = unique.length ? (now - Date.parse(unique[0].observedAt)) / DAY : 0;
  const discountPercent =
    offer.originalPrice && offer.originalPrice > offer.currentPrice
      ? (100 * (offer.originalPrice - offer.currentPrice)) / offer.originalPrice
      : undefined;
  const quoteTime = Date.parse(offer.lastSeenAt);
  if (!Number.isFinite(quoteTime) || quoteTime > now || now - quoteTime > 7 * DAY)
    return {
      state: 'INSUFFICIENT_HISTORY',
      quoteFresh: Number.isFinite(quoteTime) && quoteTime <= now && now - quoteTime <= 7 * DAY,
      confidence: 'low',
      sampleCount: unique.length,
      spanDays: Math.floor(span),
      evidence: [
        'Current offer quote is stale or undated; verify the merchant price before deciding',
      ],
    };
  if (unique.length < 8 || span < 30 || !finitePrice(offer.currentPrice) || offer.currentPrice <= 0)
    return {
      state: 'INSUFFICIENT_HISTORY',
      quoteFresh: Number.isFinite(quoteTime) && quoteTime <= now && now - quoteTime <= 7 * DAY,
      confidence: 'low',
      sampleCount: unique.length,
      spanDays: Math.floor(span),
      discountPercent,
      evidence: ['Not enough comparable price history; no sale forecast available'],
    };
  const weighted = records
    .map((record, i) => ({
      price: record.price,
      days: Math.max(
        0.001,
        ((records[i + 1] ? Date.parse(records[i + 1].observedAt) : now) -
          Date.parse(record.observedAt)) /
          DAY
      ),
    }))
    .sort((a, b) => a.price - b.price);
  const half = weighted.reduce((n, r) => n + r.days, 0) / 2;
  let cumulative = 0,
    median = weighted[weighted.length - 1].price;
  for (const record of weighted) {
    cumulative += record.days;
    if (cumulative >= half) {
      median = record.price;
      break;
    }
  }
  const minimum = Math.min(...records.map((o) => o.price)),
    below = (100 * (median - offer.currentPrice)) / median;
  const high = offer.currentPrice > median * 1.15;
  return {
    quoteFresh: true,
    state: high
      ? options.waitIfHigh
        ? 'WAIT'
        : 'HIGH_PRICE'
      : offer.currentPrice <= minimum || below >= 10
        ? 'GOOD_PRICE'
        : 'NORMAL_PRICE',
    confidence: unique.length >= 15 && span >= 60 ? 'high' : 'medium',
    sampleCount: unique.length,
    spanDays: Math.floor(span),
    median,
    minimum,
    belowMedianPercent: below,
    discountPercent,
    evidence: [
      `${unique.length} observations across ${Math.floor(span)} days`,
      `${Math.abs(below).toFixed(1)}% ${below >= 0 ? 'below' : 'above'} the observed time-weighted median`,
      ...(high ? ['Consider waiting or comparing offers; future sale timing is unknown'] : []),
    ],
  };
}
export function sameOfferProduct(a: MerchantOffer, b: MerchantOffer): boolean {
  if (
    a.canonicalIdentifier &&
    b.canonicalIdentifier &&
    a.canonicalIdentifier.kind === b.canonicalIdentifier.kind &&
    a.canonicalIdentifier.value !== b.canonicalIdentifier.value
  )
    return false;
  if (a.variantId && a.variantId === b.variantId) return true;
  const x = a.canonicalIdentifier,
    y = b.canonicalIdentifier;
  if (!x || !y || x.kind !== y.kind || x.value !== y.value) return false;
  return x.kind === 'gtin'
    ? /^(\d{8}|\d{12,14})$/.test(x.value)
    : !!x.brand && x.brand.toLowerCase() === y.brand?.toLowerCase();
}
export function compareMerchantOffers(
  reference: MerchantOffer,
  offers: readonly MerchantOffer[],
  options: {
    currency?: string;
    size?: string;
    fx?: BudgetFxContext;
    region?: string;
    now?: number;
  } = {}
) {
  const currency = options.currency || reference.currency;
  const result = offers
    .filter(
      (offer) =>
        sameOfferProduct(reference, offer) &&
        (!options.region || !offer.region || offer.region === options.region)
    )
    .slice(0, 100)
    .map((offer) => {
      const price = truePrice(offer, options.now),
        stock = options.size ? offer.sizes?.[options.size] : offer.inStock;
      const comparableTotal =
        price.total === undefined
          ? undefined
          : convertShopAmount(price.total, price.currency, currency, options.fx);
      return {
        offer,
        truePrice: price,
        stock,
        comparableTotal:
          comparableTotal !== undefined && Number.isFinite(comparableTotal)
            ? comparableTotal
            : undefined,
        notes: [
          ...(stock === undefined ? ['Selected-size stock unknown'] : []),
          ...(price.total === undefined ? [`Unknown ${price.unknownComponents.join(', ')}`] : []),
          ...(!offer.region && options.region ? ['Delivery region unknown'] : []),
        ],
      };
    });
  return result.sort(
    (a, b) =>
      (b.stock === true ? 2 : b.stock === false ? 0 : 1) -
        (a.stock === true ? 2 : a.stock === false ? 0 : 1) ||
      Number(b.comparableTotal !== undefined) - Number(a.comparableTotal !== undefined) ||
      (a.comparableTotal ?? Infinity) - (b.comparableTotal ?? Infinity) ||
      Number(b.offer.merchantConfidence === 'high') - Number(a.offer.merchantConfidence === 'high')
  );
}
/** Session cache only. Full historical series never enter AsyncStorage. */
export class LocalPriceRepository {
  private offers = new Map<string, MerchantOffer>();
  private history = new Map<string, PriceObservation[]>();
  observe(offer: MerchantOffer): void {
    if (offer.quoteKind === 'estimate') throw new Error('An estimate is not a market observation');
    if (
      !offer.id ||
      !finitePrice(offer.currentPrice) ||
      !offer.currency ||
      !Number.isFinite(Date.parse(offer.lastSeenAt))
    )
      throw new Error('Invalid offer observation');
    const prior = this.offers.get(offer.id);
    if (prior && Date.parse(prior.lastSeenAt) > Date.parse(offer.lastSeenAt)) return;
    if (this.offers.size >= 100 && !this.offers.has(offer.id)) {
      const key = this.offers.keys().next().value!;
      this.offers.delete(key);
      this.history.delete(key);
    }
    this.offers.set(offer.id, offer);
    if (
      !prior ||
      prior.currentPrice !== offer.currentPrice ||
      prior.currency !== offer.currency ||
      prior.originalPrice !== offer.originalPrice ||
      prior.inStock !== offer.inStock ||
      new Date(prior.lastSeenAt).toISOString().slice(0, 10) !==
        new Date(offer.lastSeenAt).toISOString().slice(0, 10)
    ) {
      this.history.set(
        offer.id,
        [
          ...(this.history.get(offer.id) || []),
          {
            offerId: offer.id,
            observedAt: offer.lastSeenAt,
            price: offer.currentPrice,
            currency: offer.currency,
            originalPrice: offer.originalPrice,
            inStock: offer.inStock,
          },
        ].slice(-90)
      );
    }
  }
  getOffers(variantId: string) {
    return [...this.offers.values()].filter((offer) => offer.variantId === variantId);
  }
  getHistory(offerId: string) {
    return [...(this.history.get(offerId) || [])];
  }
  clear() {
    this.offers.clear();
    this.history.clear();
  }
}
export function evaluatePriceWatch(
  watch: PriceWatch,
  offers: readonly MerchantOffer[],
  now = Date.now()
) {
  if (watch.status !== 'active') return { triggered: false, reason: 'Watch is not active' };
  const matches = offers.filter(
    (offer) =>
      offer.variantId === (watch.variantId || watch.productId) &&
      (!watch.merchant || offer.merchant === watch.merchant) &&
      offer.currency === watch.currency &&
      Date.parse(offer.lastSeenAt) <= now &&
      now - Date.parse(offer.lastSeenAt) <= DAY &&
      (watch.selectedSize ? offer.sizes?.[watch.selectedSize] === true : offer.inStock === true) &&
      offer.currentPrice <= watch.threshold
  );
  return matches.length
    ? {
        triggered: true,
        offer: matches.sort((a, b) => a.currentPrice - b.currentPrice)[0],
        reason:
          'Observed in-stock item price is below your threshold; checkout extras are separate',
      }
    : { triggered: false, reason: 'No fresh matching in-stock price below the threshold' };
}
export function knownCharge(
  amount: number,
  currency: string,
  source: string,
  observedAt: string
): PriceComponent {
  return { amount, currency, source, observedAt, confidence: 'high' };
}
