/**
 * styleOS.ts — Personal Style Operating System canonical types (Phase 3).
 *
 * Covers: StyleEvidence (3A), StyleDNA (3B), evolution (3C), contextual
 * profiles (3D), outfit feedback (3E), OutfitMemory (3F), wardrobe health
 * (3G), lifecycle (3H), cost-per-wear (3I), underused (3J), redundancy (3K),
 * gaps (3L), capsules (3M), travel (3N), events (3O), locking (3P), seasonal
 * (3Q), goals (3R), budgets (3S), pipeline (3T), post-mortem (3U), shopping
 * plan (3V), no-buy (3W), agentic tasks (3X), proactive (3Y), fatigue (3Z),
 * outcomes, graph, queries, novelty.
 *
 * All types are plain data; deterministic logic lives in
 * src/services/style/*. Never infer sensitive identity attributes.
 */

export type StylePolarity = 'positive' | 'negative' | 'neutral';
export type StyleSourceType =
  | 'declared_preference'
  | 'outfit_worn'
  | 'outfit_saved'
  | 'outfit_liked'
  | 'outfit_disliked'
  | 'product_saved'
  | 'product_skipped'
  | 'product_bought'
  | 'product_returned'
  | 'manual_feedback'
  | 'wardrobe_item'
  | 'reference_outfit'
  | 'moodboard'
  | 'social_import';
export type StyleEntityType =
  | 'garment'
  | 'outfit'
  | 'product'
  | 'visual_family'
  | 'brand'
  | 'color'
  | 'silhouette'
  | 'material'
  | 'pattern'
  | 'style_tag';

export interface StyleEvidence {
  id: string;
  userId: string;
  sourceType: StyleSourceType;
  entityType: StyleEntityType;
  /** Stable key: e.g. `color:navy`, `brand:uniqlo`, `silhouette:relaxed`. */
  entityKey: string;
  polarity: StylePolarity;
  /** 0..1 observation strength. */
  strength: number;
  /** Free-form situational context, e.g. `{ contextId: 'office' }`. */
  context?: Record<string, string>;
  confidence: number;
  observedAt: string;
  /** Optional ISO timestamp after which weight decays to zero. */
  expiresAt?: string | null;
  provenance: string;
}

/** One aggregated StyleDNA dimension value. */
export interface StyleDimension {
  value: string;
  confidence: 'low' | 'medium' | 'high' | 'emerging';
  evidenceCount: number;
  provenance: string[];
  /** ISO timestamp of the most recent contributing evidence. */
  recency: string | null;
  /** 0..1 — contradictory evidence share. */
  contradiction: number;
}

export interface StyleDNA {
  version: 1;
  userId: string;
  computedAt: string;
  evidenceCount: number;
  dimensions: Record<string, StyleDimension>;
  /** Dimensions with too little evidence to trust yet. */
  emerging: string[];
}

export type StyleLayer = 'baseline' | 'recent' | 'override' | 'contextual';

export interface StyleEvolutionSignal {
  dimension: string;
  previousValue: string | null;
  emergingValue: string;
  confidence: 'low' | 'medium' | 'high';
  evidenceWindow: { from: string; to: string; count: number };
  changeStrength: number;
}

/** Built-in + user-configured context ids. */
export type StyleContextId =
  | 'university'
  | 'office'
  | 'casual_weekend'
  | 'dinner_date'
  | 'formal_event'
  | 'travel'
  | 'warm_weather'
  | 'rainy_day'
  | 'gym'
  | string;

export interface ContextualStyleProfile {
  contextId: StyleContextId;
  label: string;
  /** Context-scoped dimension overrides; falls back to global StyleDNA. */
  dimensions: Record<string, StyleDimension>;
  evidenceCount: number;
  confidence: 'low' | 'medium' | 'high';
}

export type OutfitRating = 'loved' | 'good' | 'okay' | 'didnt_work';
export type OutfitFeedbackReason =
  | 'uncomfortable'
  | 'too_hot'
  | 'too_cold'
  | 'too_formal'
  | 'too_casual'
  | 'color_combination'
  | 'fit_issue'
  | 'shoes'
  | 'confidence'
  | 'practicality';

export interface OutfitFeedback {
  id: string;
  userId: string;
  outfitId: string;
  rating: OutfitRating;
  reasons?: OutfitFeedbackReason[];
  contextId?: StyleContextId | string;
  createdAt: string;
}

export interface OutfitMemoryEntry {
  outfitId: string;
  userId: string;
  itemIds: string[];
  wornDate: string;
  plannedDate?: string | null;
  contextId?: string | null;
  rating?: OutfitRating | null;
  weatherTemp?: number | null;
  weatherCondition?: string | null;
  repeatCount: number;
  lastWorn: string | null;
  combinationKey: string;
  planned: boolean;
  confirmed: boolean;
  note?: string | null;
}

export type HealthGrade = 'strong' | 'healthy' | 'needs_attention' | 'gap' | 'unknown';

export interface WardrobeHealthDimension {
  key: string;
  grade: HealthGrade;
  score: number;
  evidence: string[];
  confidence: 'low' | 'medium' | 'high';
}

export interface WardrobeHealth {
  userId: string;
  computedAt: string;
  itemCount: number;
  dimensions: WardrobeHealthDimension[];
  summary: string[];
}

export type LifecycleStatus =
  | 'active'
  | 'seasonal'
  | 'storage'
  | 'repair'
  | 'sell'
  | 'donate'
  | 'archived';

export interface ItemLifecycle {
  itemId: string;
  userId: string;
  acquiredAt?: string | null;
  acquisitionPrice?: number | null;
  currency?: string | null;
  /** 'retail' | 'secondhand' | 'gift' | ... — never fabricated. */
  source?: string | null;
  condition?: 'excellent' | 'good' | 'fair' | 'worn' | 'damaged' | null;
  repairNeeded?: boolean;
  lastWornAt?: string | null;
  wearCount: number;
  status: LifecycleStatus;
  satisfaction?: 1 | 2 | 3 | 4 | 5 | null;
  fitStatus?: 'good' | 'tight' | 'loose' | 'needs_alteration' | 'unknown' | null;
  replacementCandidate?: boolean;
}

export interface CostPerWear {
  itemId: string;
  wearCount: number;
  acquisitionPrice: number | null;
  currency: string | null;
  costPerWear: number | null;
  estimated: boolean;
  explanation: string;
}

export interface UnderusedItem {
  itemId: string;
  daysOwned: number;
  wearCount: number;
  reasons: Array<
    | 'hard_to_combine'
    | 'duplicate_role'
    | 'fit_issue'
    | 'style_mismatch'
    | 'condition_issue'
    | 'missing_complement'
    | 'unknown'
  >;
  evidence: string[];
  suggestedActions: Array<'style_it' | 'repair' | 'sell' | 'donate' | 'find_complement' | 'keep'>;
}

export interface RedundantRole {
  roleKey: string;
  label: string;
  itemIds: string[];
  wearDistribution: Array<{ itemId: string; wears: number; share: number }>;
  favoriteItemId: string | null;
  underusedItemIds: string[];
  duplicateRisk: 'low' | 'medium' | 'high';
  note: string;
}

export interface WardrobeGap {
  id: string;
  type: string;
  label: string;
  blockedContexts: string[];
  blockedOutfitCount: number;
  expectedUnlocks: number;
  urgency: 'low' | 'medium' | 'high';
  confidence: 'low' | 'medium' | 'high';
  candidateCategories: string[];
  candidateAttributes: Record<string, string>;
}

export interface CapsuleRequest {
  kind: 'university_15' | 'smart_casual' | 'summer_budget' | 'travel_7day' | 'custom';
  userId: string;
  itemPoolIds?: string[];
  budget?: { amount: number; currency: string } | null;
  climate?: { tempC?: number; condition?: string } | null;
  laundryCycleDays?: number;
  repetitionTolerance?: 'low' | 'medium' | 'high';
  maxItems?: number;
}

export interface CapsuleResult {
  itemIds: string[];
  reusedOwnedIds: string[];
  shopGaps: string[];
  outfitCount: number;
  withinBudget: boolean | null;
  evidence: string[];
}

export interface TripProfile {
  destination?: string | null;
  tempC?: number | null;
  condition?: string | null;
  days: number;
  activities: string[];
  laundryAvailable: boolean;
  luggageLimit?: number | null;
  dressRequirements?: string[];
}

export interface PackingResult {
  packingList: string[];
  outfitSchedule: Array<{ day: number; itemIds: string[]; note?: string }>;
  repeatStrategy: string;
  backupItemIds: string[];
  footwearPlan: string[];
  missingItems: string[];
  buySuggestions: string[];
  evidence: string[];
}

export interface WardrobeEvent {
  id: string;
  userId: string;
  title: string;
  startsAt: string;
  contextId: string;
  dressCode?: string | null;
  climateHint?: string | null;
  source: 'manual' | 'calendar' | 'import';
  selectedOutfitId?: string | null;
  locked?: boolean;
  /** 'draft' | 'planned' | 'ready' | 'needs_attention' */
  preparationStatus: string;
  missingItemIds: string[];
}

export interface LockedOutfit {
  eventId: string;
  outfitId: string;
  itemIds: string[];
  readiness: 'ready' | 'needs_attention' | 'blocked';
  availability: Record<string, 'available' | 'laundry' | 'repair' | 'missing' | 'unknown'>;
  weatherCompatible: boolean | null;
  backupOutfitId?: string | null;
  evidence: string[];
}

export interface SeasonalRotation {
  outOfSeasonIds: string[];
  resurfaceIds: string[];
  activeIds: string[];
  seasonalGaps: string[];
  evidence: string[];
}

export interface WardrobeGoal {
  id: string;
  userId: string;
  type:
    | 'spend_less'
    | 'dress_professionally'
    | 'university_wardrobe'
    | 'reduce_size'
    | 'wear_more'
    | 'improve_formalwear'
    | 'stop_duplicates'
    | 'travel_capsule'
    | 'style_direction'
    | string;
  target?: string | null;
  timeHorizonDays?: number | null;
  budget?: { amount: number; currency: string } | null;
  priority: 'low' | 'medium' | 'high';
  status: 'active' | 'paused' | 'done' | 'abandoned';
  progress: Record<string, number>;
  constraints: string[];
}

export interface ShoppingBudget {
  id: string;
  userId: string;
  period: string;
  amount: number;
  spent: number;
  committed: number;
  available: number;
  currency: string;
  categoryAllocations: Record<string, number>;
  flexibility: 'strict' | 'flexible';
}

export type PipelineStage =
  | 'discovered'
  | 'considering'
  | 'compared'
  | 'waiting_for_price'
  | 'planned'
  | 'bought'
  | 'skipped'
  | 'returned'
  | 'archived';

export interface PipelineEntry {
  id: string;
  userId: string;
  productId: string;
  stage: PipelineStage;
  decisionReason?: string | null;
  targetPrice?: number | null;
  currency?: string | null;
  evidence: string[];
  updatedAt: string;
}

export interface PurchasePostMortem {
  productId: string;
  userId: string;
  worthIt?: boolean | null;
  fitsWell?: boolean | null;
  wearCount: number;
  wouldBuyAgain?: boolean | null;
  askedAt: string;
  answeredAt?: string | null;
  calibration: string[];
}

export interface PlannedPurchase {
  productId: string;
  label: string;
  priority: number;
  expectedUnlocks: number;
  confidence: 'low' | 'medium' | 'high';
  evidence: string[];
  tradeoffs: string[];
}

export interface ShoppingPlan {
  userId: string;
  computedAt: string;
  items: PlannedPurchase[];
  buyNothing: boolean;
  rationale: string[];
}

export type NoBuyRecommendation =
  | 'BUY_NOW'
  | 'BUY_IF_NEEDED'
  | 'WAIT_FOR_PRICE'
  | 'WAIT_FOR_BETTER_FIT_EVIDENCE'
  | 'SKIP_DUPLICATE'
  | 'SKIP_LOW_VALUE'
  | 'NO_PURCHASE_NEEDED';

export interface NoBuyDecision {
  recommendation: NoBuyRecommendation;
  productId?: string | null;
  confidence: 'low' | 'medium' | 'high';
  evidence: string[];
  explanation: string;
}

export interface AgentTask {
  id: string;
  kind:
    | 'plan_week'
    | 'build_capsule'
    | 'find_shoe'
    | 'review_underused'
    | 'check_watches'
    | 'prepare_event'
    | 'optimize_budget'
    | string;
  title: string;
  status: 'pending' | 'ready' | 'needs_input' | 'done' | 'dismissed';
  payload: Record<string, string>;
  resultSummary?: string | null;
  evidence: string[];
}

export interface ProactiveSuggestion {
  id: string;
  kind: string;
  title: string;
  detail: string;
  urgency: 'low' | 'medium' | 'high';
  confidence: 'low' | 'medium' | 'high';
  expectedValue: string;
  evidence: string[];
  tradeoffs: string[];
  createdAt: string;
}

export interface FatigueRecord {
  topicKey: string;
  lastSurfacedAt: string | null;
  surfacedCount7d: number;
  dismissedCount7d: number;
  acceptedCount7d: number;
  snoozedUntil?: string | null;
  promptType: string;
}

export interface RecommendationOutcome {
  suggestionId: string;
  outcome: 'accepted' | 'ignored' | 'dismissed' | 'purchased' | 'returned' | 'worn_repeatedly' | 'rated_poorly';
  at: string;
}

export type NoveltyMode = 'safe' | 'balanced' | 'experimental';

/** Explainability contract (§29) attached to major suggestions. */
export interface SuggestionExplanation {
  whyNow: string;
  evidence: string[];
  changeIfFollowed: string;
  tradeoffs: string[];
  confidence: 'low' | 'medium' | 'high';
}

export interface WardrobeGraphEdge {
  from: string;
  to: string;
  kind:
    | 'works_with'
    | 'worn_with'
    | 'similar_to'
    | 'substitutes'
    | 'conflicts_with'
    | 'unlocks'
    | 'preferred_for'
    | 'duplicate_role'
    | 'reference_fit';
  weight: number;
  evidence: string;
}

export interface SemanticTextDoc {
  id: string;
  text: string;
}
