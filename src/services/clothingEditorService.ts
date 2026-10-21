import { supabase } from '../../lib/supabase';
import type { ClothingItem } from '../types/domain';
import { authEvents } from '../../store/authEvents';

/** The editor persists directly to the same owner-protected table as wardrobe uploads. */
export async function saveClothingEdits(id: string, updates: Partial<ClothingItem>): Promise<{ id: string }> {
  const revision = authEvents.getRevision();
  const { data: sessionData, error: sessionError } = await supabase.auth.getSession();
  const owner = sessionData.session?.user.id;
  if (sessionError || !owner || revision !== authEvents.getRevision()) throw new Error('Signed-in account required');
  const fields: Record<string, unknown> = { updated_at: new Date().toISOString() };
  if (updates.category !== undefined) fields.category = updates.category;
  if (updates.subCategory !== undefined) { fields.sub_category = updates.subCategory; fields.type = updates.subCategory; }
  if (updates.primaryColor !== undefined) fields.primary_color = updates.primaryColor;
  if (updates.colorHex !== undefined) fields.color_hex = updates.colorHex;
  if (updates.seasons !== undefined) fields.seasons = updates.seasons;
  if (updates.occasions !== undefined) fields.occasions = updates.occasions;
  const { data, error } = await supabase.from('clothing_items').update(fields)
    .eq('id', id).eq('user_id', owner).select('id').single();
  if (error) throw error;
  if (revision !== authEvents.getRevision()) throw new Error('Account changed during save');
  if (data?.id !== id) throw new Error('Item save was not confirmed');
  return { id: data.id };
}


/** Idempotent owner-scoped removal: an already absent owner row is complete. */
export async function deleteClothingItem(id: string): Promise<void> {
  const revision = authEvents.getRevision();
  const { data, error: sessionError } = await supabase.auth.getSession();
  const owner = data.session?.user.id;
  if (sessionError || !owner || revision !== authEvents.getRevision()) throw new Error('Signed-in account required');
  const { error } = await supabase.from('clothing_items').delete().eq('id', id).eq('user_id', owner);
  if (error) throw error;
  if (revision !== authEvents.getRevision()) throw new Error('Account changed during removal');
}
