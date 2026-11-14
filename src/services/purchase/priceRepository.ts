import type { MerchantOffer, PriceObservation, PriceComponent } from '../../types/purchase';
import { LocalPriceRepository } from './priceIntelligence';
import { withRequestDeadline } from '../../lib/requestDeadline';
import { supabase } from '../../../lib/supabase';
/** Read adapter is dormant until explicitly enabled after schema activation.
 * No network writes, automatic import, scrape, scheduler or historical backfill. */
export function createPriceReadRepository(local: LocalPriceRepository, remoteEnabled = false) {
  const pending = new Map<
    string,
    Promise<{ offers: MerchantOffer[]; history: PriceObservation[]; source: 'local' | 'remote' }>
  >();
  const cache = new Map<
    string,
    {
      at: number;
      value: { offers: MerchantOffer[]; history: PriceObservation[]; source: 'local' | 'remote' };
    }
  >();
  return {
    async read(variantId: string) {
      const fallback = () => {
        const offers = local.getOffers(variantId);
        return {
          offers,
          history: offers.flatMap((offer) => local.getHistory(offer.id)),
          source: 'local' as const,
        };
      };
      if (!remoteEnabled) return fallback();
      const prior = cache.get(variantId);
      if (prior && Date.now() - prior.at < 300000) return prior.value;
      const current = pending.get(variantId);
      if (current) return current;
      const request = withRequestDeadline(async (signal) => {
        const { data, error } = await supabase
          .from('shop_offers')
          .select('*,shop_merchants(name,confidence,return_policy)')
          .eq('variant_id', variantId)
          .eq('is_active', true)
          .limit(20)
          .abortSignal(signal);
        if (error) throw error;
        const offers: MerchantOffer[] = (data || []).map((raw: Record<string, any>) => {
          const component = (name: string, amount: unknown): PriceComponent | undefined => {
            const provenance = raw.component_provenance?.[name];
            if (
              amount === null ||
              amount === undefined ||
              !Number.isFinite(Number(amount)) ||
              !provenance?.source ||
              !provenance?.observedAt
            )
              return undefined;
            return {
              amount: Number(amount),
              currency: raw.currency,
              source: provenance.source,
              observedAt: provenance.observedAt,
              confidence: ['low', 'medium', 'high'].includes(provenance.confidence)
                ? provenance.confidence
                : 'low',
            };
          };
          return {
            id: raw.id,
            variantId: raw.variant_id,
            merchant: raw.shop_merchants?.name || raw.merchant_id,
            merchantProductId: raw.merchant_product_id,
            url: raw.url,
            region: raw.region,
            currency: raw.currency,
            currentPrice: Number(raw.current_price),
            originalPrice: raw.original_price == null ? undefined : Number(raw.original_price),
            shipping: component('shipping', raw.shipping_price),
            tax: component('tax', raw.tax_price),
            fees: component('fees', raw.fees),
            discount: component('discount', raw.discount),
            coupon: component('coupon', raw.coupon),
            canonicalIdentifier: raw.canonical_identifier || undefined,
            inStock: typeof raw.in_stock === 'boolean' ? raw.in_stock : undefined,
            sizes: raw.size_stock || undefined,
            lastSeenAt: raw.last_seen_at,
            merchantConfidence: raw.shop_merchants?.confidence || 'low',
            returnPolicy: raw.shop_merchants?.return_policy,
          };
        });
        const histories = await Promise.all(
          offers.slice(0, 8).map(async (offer) => {
            const response = await supabase
              .from('shop_price_history')
              .select('offer_id,observed_at,price,currency,original_price,in_stock')
              .eq('offer_id', offer.id)
              .order('observed_at', { ascending: false })
              .limit(90)
              .abortSignal(signal);
            if (response.error) throw response.error;
            return (response.data || []).map((row: Record<string, any>): PriceObservation => ({
              offerId: row.offer_id,
              observedAt: row.observed_at,
              price: Number(row.price),
              currency: row.currency,
              originalPrice: row.original_price == null ? undefined : Number(row.original_price),
              inStock: row.in_stock,
            }));
          })
        );
        return { offers, history: histories.flat(), source: 'remote' as const };
      })
        .then((value) => {
          if (cache.size >= 12) cache.delete(cache.keys().next().value!);
          cache.set(variantId, { value, at: Date.now() });
          return value;
        })
        .catch(() => fallback())
        .finally(() => pending.delete(variantId));
      pending.set(variantId, request);
      return request;
    },
  };
}
