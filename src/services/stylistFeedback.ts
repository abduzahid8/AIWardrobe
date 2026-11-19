import { supabase } from '../../lib/supabase';
import {
  outfitStylingFeatures,
  outfitVisualSignature,
  type OutfitStylingFeatures,
} from '../domain/outfitVisualSignature';

export interface LookFeedbackContext {
  id: string;
  concept?: { id: string };
  generationRequestId?: string;
  engineVersion?: string;
  /** Server-issued items of the displayed look (id-bearing). */
  items?: ReadonlyArray<{ id?: unknown }>;
  /** Catalog-grounded direction label the look was displayed under. */
  direction?: { id?: unknown } | null;
}
export type EligibleLookFeedback = LookFeedbackContext & {
  concept: { id: string };
  generationRequestId: string;
  engineVersion: string;
};
/** Matches the stylist_look_feedback check constraints, so the UI never
 * offers a control whose write the database would reject. */
export function canRecordLookFeedback(look: Partial<LookFeedbackContext> | null | undefined): look is EligibleLookFeedback {
  if (!look) return false;
  const lookId = String(look.id || '');
  const conceptId = String(look.concept?.id || '');
  const engineVersion = String(look.engineVersion || '');
  return /^outfit_[1-3]$/.test(lookId)
    && /^[a-z][a-z0-9-]{0,63}$/.test(conceptId)
    && Boolean(look.generationRequestId)
    && engineVersion.length >= 1 && engineVersion.length <= 80;
}

export const LOOK_FEEDBACK_SIGNATURE_VERSION = 1 as const;
export interface LookFeedbackSignature {
  version: typeof LOOK_FEEDBACK_SIGNATURE_VERSION;
  direction: string | null;
  signature: string | null;
  features: OutfitStylingFeatures;
}
const MAX_FEEDBACK_ITEM_IDS = 8;
const MAX_ITEM_ID_LENGTH = 128;
const MAX_SIGNATURE_CHARS = 1000;

/** Bounded, exact item ids of the rated look: the linking key the Stylist 6
 * ranker matches against future candidate items. Trims, bounds length and
 * de-dupes; non-string or empty entries are dropped, never invented. */
export function lookFeedbackItemIds(look: LookFeedbackContext): string[] {
  const ids: string[] = [];
  for (const item of Array.isArray(look.items) ? look.items.slice(0, 12) : []) {
    const id = typeof (item as { id?: unknown } | null)?.id === 'string'
      ? (item as { id: string }).id.trim().slice(0, MAX_ITEM_ID_LENGTH)
      : '';
    if (id && !ids.includes(id)) ids.push(id);
    if (ids.length >= MAX_FEEDBACK_ITEM_IDS) break;
  }
  return ids;
}

/**
 * Versioned styling-feature signature of the rated look, computed from the
 * shared src/domain/outfitVisualSignature.ts exports: the per-slot styling
 * features (outfitStylingFeatures) and the sq1 semantic identity string
 * (outfitVisualSignature), plus the direction label the look was displayed
 * under. Unknown pieces degrade to null — the write never fabricates styling
 * evidence, and a malformed look can never make the upsert throw.
 */
export function buildLookFeedbackSignature(look: LookFeedbackContext): LookFeedbackSignature {
  const directionId = typeof look.direction?.id === 'string'
    ? look.direction.id.trim().slice(0, 64) : null;
  try {
    const items = Array.isArray(look.items) && look.items.length
      ? (look.items as never[]) : null;
    if (!items) {
      return { version: LOOK_FEEDBACK_SIGNATURE_VERSION, direction: directionId, signature: null, features: [] };
    }
    const signature = outfitVisualSignature(items);
    return {
      version: LOOK_FEEDBACK_SIGNATURE_VERSION,
      direction: directionId,
      signature: signature.length > 0 && signature.length <= MAX_SIGNATURE_CHARS ? signature : null,
      features: outfitStylingFeatures(items),
    };
  } catch {
    return { version: LOOK_FEEDBACK_SIGNATURE_VERSION, direction: directionId, signature: null, features: [] };
  }
}

export type FeedbackTargetSlot = 'top' | 'outerwear' | 'bottom' | 'shoes' | 'accessory' | 'dress';
export type LookFeedbackReason = 'coherence' | 'occasion' | 'color' | 'fit' | 'footwear' | 'like';

export interface LookFeedbackOptions {
  reason?: LookFeedbackReason;
  targetSlot?: FeedbackTargetSlot | null;
  feedbackTags?: readonly string[];
  swappedFromSku?: string | null;
}

export interface SubmitStylistLookFeedbackInput extends LookFeedbackOptions {
  look: LookFeedbackContext;
  rating: -1 | 1;
}

const VALID_FEEDBACK_SLOTS = new Set<FeedbackTargetSlot>([
  'top',
  'outerwear',
  'bottom',
  'shoes',
  'accessory',
  'dress',
]);

const VALID_FEEDBACK_REASONS = new Set<LookFeedbackReason>([
  'coherence',
  'occasion',
  'color',
  'fit',
  'footwear',
  'like',
]);

const MAX_FEEDBACK_TAGS = 8;

function sanitizeTargetSlot(slot: unknown): FeedbackTargetSlot | null {
  if (typeof slot !== 'string') return null;
  const normalized = slot.trim().toLowerCase() as FeedbackTargetSlot;
  return VALID_FEEDBACK_SLOTS.has(normalized) ? normalized : null;
}

function sanitizeFeedbackTags(tags: readonly unknown[] | undefined): string[] {
  if (!Array.isArray(tags)) return [];
  const out: string[] = [];
  for (const raw of tags.slice(0, 16)) {
    if (typeof raw !== 'string') continue;
    const normalized = raw.trim().toLowerCase().replace(/[\s-]+/g, '_');
    if (/^[a-z][a-z0-9_]{1,39}$/.test(normalized) && !out.includes(normalized)) {
      out.push(normalized);
      if (out.length >= MAX_FEEDBACK_TAGS) break;
    }
  }
  return out;
}

function sanitizeSwappedFromSku(sku: unknown): string | null {
  if (typeof sku !== 'string') return null;
  const trimmed = sku.trim().slice(0, MAX_ITEM_ID_LENGTH);
  return trimmed.length > 0 ? trimmed : null;
}

/**
 * Resolves a Postgres check-constraint-compliant `reason` value from rating,
 * explicit reason, structured tags, and target slot so thumbs-down feedback
 * is no longer hardcoded to `'coherence'` when structured reasons exist.
 */
export function resolveFeedbackReason(
  rating: -1 | 1,
  options?: LookFeedbackOptions,
): LookFeedbackReason {
  if (options?.reason && VALID_FEEDBACK_REASONS.has(options.reason)) {
    return options.reason;
  }
  if (rating === 1) return 'like';
  const tags = sanitizeFeedbackTags(options?.feedbackTags);
  if (tags.includes('wrong_shoes')) return 'footwear';
  if (tags.includes('wrong_color')) return 'color';
  if (tags.includes('too_formal') || tags.includes('too_casual')) return 'occasion';
  if (tags.includes('not_my_fit')) return 'fit';
  const slot = sanitizeTargetSlot(options?.targetSlot);
  if (slot === 'shoes') return 'footwear';
  return 'coherence';
}

function isMissingAdditiveColumnError(error: unknown): boolean {
  if (!error || typeof error !== 'object') return false;
  const err = error as { code?: unknown; message?: unknown };
  const code = typeof err.code === 'string' ? err.code : '';
  const msg = typeof err.message === 'string' ? err.message : '';
  return (code === '42703' || code === 'PGRST204')
    && /(?:target_slot|feedback_tags|swapped_from_sku)/i.test(msg);
}

/** One authenticated upsert; no prompt, product/image list or free text.
 * The additive item_ids/look_signature columns (migration 20261113120000) and
 * target_slot/feedback_tags/swapped_from_sku columns (migration 20261119091000)
 * make the rating linkable to future Stylist 6 slot-level taste learning. */
export async function recordLookFeedback(
  look: LookFeedbackContext,
  rating: -1 | 1,
  options?: LookFeedbackOptions,
): Promise<void> {
  if (!canRecordLookFeedback(look)) return;
  const itemIds = lookFeedbackItemIds(look);
  const feedbackTags = sanitizeFeedbackTags(options?.feedbackTags);
  const explicitSlot = sanitizeTargetSlot(options?.targetSlot);
  const targetSlot = explicitSlot
    ?? (options?.targetSlot === undefined && feedbackTags.includes('wrong_shoes') ? 'shoes' : null);
  const swappedFromSku = sanitizeSwappedFromSku(options?.swappedFromSku);
  const reason = resolveFeedbackReason(rating, {
    reason: options?.reason,
    targetSlot,
    feedbackTags,
  });

  const basePayload: Record<string, unknown> = {
    request_id: look.generationRequestId,
    look_id: look.id,
    concept_id: look.concept.id,
    engine_version: look.engineVersion,
    rating,
    reason,
    ...(itemIds.length ? { item_ids: itemIds } : {}),
    look_signature: buildLookFeedbackSignature(look),
  };

  const hasAdditiveFields = Boolean(targetSlot || feedbackTags.length > 0 || swappedFromSku);
  const fullPayload: Record<string, unknown> = {
    ...basePayload,
    ...(targetSlot ? { target_slot: targetSlot } : {}),
    ...(feedbackTags.length > 0 ? { feedback_tags: feedbackTags } : {}),
    ...(swappedFromSku ? { swapped_from_sku: swappedFromSku } : {}),
  };

  const { error } = await supabase.from('stylist_look_feedback').upsert(
    fullPayload,
    { onConflict: 'user_id,request_id,look_id' },
  );
  if (error) {
    if (hasAdditiveFields && isMissingAdditiveColumnError(error)) {
      const retry = await supabase.from('stylist_look_feedback').upsert(
        basePayload,
        { onConflict: 'user_id,request_id,look_id' },
      );
      if (!retry.error) return;
    }
    throw Error('feedback_unavailable');
  }
}

export async function submitStylistLookFeedback(input: SubmitStylistLookFeedbackInput): Promise<void>;
export async function submitStylistLookFeedback(
  look: LookFeedbackContext,
  rating: -1 | 1,
  options?: LookFeedbackOptions,
): Promise<void>;
export async function submitStylistLookFeedback(
  lookOrInput: LookFeedbackContext | SubmitStylistLookFeedbackInput,
  rating?: -1 | 1,
  options?: LookFeedbackOptions,
): Promise<void> {
  if ('look' in lookOrInput && 'rating' in lookOrInput && rating === undefined) {
    const { look, rating: inputRating, ...rest } = lookOrInput;
    return recordLookFeedback(look, inputRating, rest);
  }
  return recordLookFeedback(lookOrInput as LookFeedbackContext, rating as -1 | 1, options);
}
