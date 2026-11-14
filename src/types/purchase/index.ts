import type { ShopCatalogItem } from '../shop';
import type { EvidenceConfidence, SizeRecommendation, FitProfile } from './fit';
import type { PriceAssessment, MerchantOffer, TruePrice } from './price';
import type { ShoppingAdviceContext } from '../../services/shoppingAdvisor';
export * from './fit';
export * from './price';
export interface DuplicateAssessment {
  level: 'low' | 'medium' | 'high';
  substitutes: Array<{ item: ShopCatalogItem; similarity: number; evidence: string[] }>;
  explanation: string;
  confidence: EvidenceConfidence;
}
export interface UsefulUnlocksV2 {
  total: number;
  looks: Array<{ items: ShopCatalogItem[]; context: string }>;
  contextBreakdown: Record<string, number>;
  evaluations: number;
  truncated: boolean;
  confidence: 'sampled' | 'insufficient';
  evidence: string[];
}
export interface PurchaseContext extends ShoppingAdviceContext {
  fitProfile?: FitProfile;
  selectedSize?: string;
  allowMultiples?: boolean;
  contexts?: Array<{
    id: string;
    occasion: string;
    style?: string;
    weather?: { temp: number; condition?: string };
  }>;
}
export interface BuyCheckResult {
  verdict: 'BUY' | 'CONSIDER' | 'SKIP' | 'INSUFFICIENT_EVIDENCE';
  overallConfidence: EvidenceConfidence;
  candidate: ShopCatalogItem;
  selectedVariantId: string;
  fit: SizeRecommendation;
  wardrobe: {
    usefulOutfitUnlocks: UsefulUnlocksV2;
    duplicateRisk: DuplicateAssessment;
    wardrobeValue: 'Low' | 'Medium' | 'High';
    missingPieceValue: 'low' | 'medium' | 'high';
  };
  purchase: {
    withinBudget?: boolean;
    budgetBasis?:'item'|'checkout';
    priceAssessment?: PriceAssessment;
    offers?: Array<{ offer: MerchantOffer; truePrice: TruePrice }>;
  };
  explanation: string[];
  evidence: string[];
}
export type FindBetterMode =
  | 'overall'
  | 'cheaper'
  | 'closest-style'
  | 'fit'
  | 'wardrobe'
  | 'lowest-duplicate'
  | 'missing-piece';
export interface PurchaseAlternative {
  item: ShopCatalogItem;
  mode: FindBetterMode;
  analysis: BuyCheckResult;
  why: string[];
  tradeoffs: string[];
}
