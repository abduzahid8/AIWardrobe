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

/** One authenticated upsert; no prompt, product/image list or free text.
 * The additive item_ids/look_signature columns (migration
 * 20261113120000) make the rating linkable to future Stylist 6 rankings. */
export async function recordLookFeedback(look: LookFeedbackContext, rating: -1 | 1): Promise<void> {
  if (!canRecordLookFeedback(look)) return;
  const itemIds = lookFeedbackItemIds(look);
  const { error } = await supabase.from('stylist_look_feedback').upsert(
    {
      request_id: look.generationRequestId,
      look_id: look.id,
      concept_id: look.concept.id,
      engine_version: look.engineVersion,
      rating,
      reason: rating === 1 ? 'like' : 'coherence',
      ...(itemIds.length ? { item_ids: itemIds } : {}),
      look_signature: buildLookFeedbackSignature(look),
    },
    { onConflict: 'user_id,request_id,look_id' }
  );
  if (error) throw Error('feedback_unavailable');
}
