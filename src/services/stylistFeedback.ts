import { supabase } from '../../lib/supabase';
export interface LookFeedbackContext {
  id: string;
  concept?: { id: string };
  generationRequestId?: string;
  engineVersion?: string;
}
/** One authenticated upsert; no prompt, product/image list or free text. */
export async function recordLookFeedback(look: LookFeedbackContext, rating: -1 | 1): Promise<void> {
  if (!look.concept?.id || !look.generationRequestId || !look.engineVersion) return;
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
