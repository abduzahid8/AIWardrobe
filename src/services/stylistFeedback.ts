import { supabase } from '../../lib/supabase';
export interface LookFeedbackContext {
  id: string;
  concept?: { id: string };
  generationRequestId?: string;
  engineVersion?: string;
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
/** One authenticated upsert; no prompt, product/image list or free text. */
export async function recordLookFeedback(look: LookFeedbackContext, rating: -1 | 1): Promise<void> {
  if (!canRecordLookFeedback(look)) return;
  const { error } = await supabase.from('stylist_look_feedback').upsert(
    {
      request_id: look.generationRequestId,
      look_id: look.id,
      concept_id: look.concept.id,
      engine_version: look.engineVersion,
      rating,
      reason: rating === 1 ? 'like' : 'coherence',
    },
    { onConflict: 'user_id,request_id,look_id' }
  );
  if (error) throw Error('feedback_unavailable');
}
