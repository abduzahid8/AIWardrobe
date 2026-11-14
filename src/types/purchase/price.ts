import type { EvidenceConfidence } from './fit';
export interface PriceComponent {
  amount: number;
  currency: string;
  source: string;
  confidence: EvidenceConfidence;
  observedAt: string;
}
export interface MerchantOffer {
  quoteKind?: 'merchant' | 'estimate';
  id: string;
  variantId: string;
  productId?: string;
  canonicalIdentifier?: { kind: 'gtin' | 'mpn'; value: string; brand?: string };
  merchant: string;
  merchantProductId?: string;
  url: string;
  region?: string;
  currency: string;
  currentPrice: number;
  originalPrice?: number;
  shipping?: PriceComponent;
  tax?: PriceComponent;
  fees?: PriceComponent;
  discount?: PriceComponent;
  coupon?: PriceComponent;
  inStock?: boolean;
  sizes?: Record<string, boolean>;
  lastSeenAt: string;
  merchantConfidence: EvidenceConfidence;
  returnPolicy?: { text: string; source: string; observedAt: string };
}
export interface PriceObservation {
  offerId: string;
  observedAt: string;
  price: number;
  currency: string;
  originalPrice?: number;
  inStock?: boolean;
}
export type PriceState =
  'GOOD_PRICE' | 'NORMAL_PRICE' | 'HIGH_PRICE' | 'WAIT' | 'INSUFFICIENT_HISTORY';
export interface PriceAssessment {
  state: PriceState;
  quoteFresh?: boolean;
  confidence: EvidenceConfidence;
  sampleCount: number;
  spanDays: number;
  median?: number;
  minimum?: number;
  belowMedianPercent?: number;
  discountPercent?: number;
  evidence: string[];
}
export interface TruePrice {
  currency: string;
  knownSubtotal: number;
  total?: number;
  unknownComponents: string[];
  components: Partial<
    Record<'item' | 'shipping' | 'tax' | 'fees' | 'discount' | 'coupon', PriceComponent>
  >;
  confidence: EvidenceConfidence;
}
export interface PriceWatch {
  id: string;
  userId: string;
  productId: string;
  variantId?: string;
  selectedSize?: string;
  merchant?: string;
  currency: string;
  threshold: number;
  status: 'active' | 'paused' | 'triggered';
  createdAt: string;
  lastCheckedAt?: string;
  lastNotifiedOfferId?: string;
}
