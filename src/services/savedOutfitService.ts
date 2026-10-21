import { supabase } from '../../lib/supabase';
import { authEvents } from '../../store/authEvents';
import { parseCalendarDate } from '../utils/calendarDate';
import { prepareWardrobeItemMedia } from './wardrobeItemMedia';

export interface SavedOutfitInput {
  id: string;
  items: Array<{ id: string; type?: string; image?: string; imageUrl?: string; name?: string }>;
  date: string;
  occasion: string;
  caption?: string;
}

export async function saveCloudOutfit(input: SavedOutfitInput): Promise<SavedOutfitInput['items']> {
  if (!input.items.length || !parseCalendarDate(input.date)) throw new Error('Invalid saved outfit');
  const snapshot = { ...input, items: input.items.map(item => ({ ...item })) };
  const revision = authEvents.getRevision();
  const { data, error: sessionError } = await supabase.auth.getSession();
  const owner = data.session?.user.id;
  if (sessionError || !owner || revision !== authEvents.getRevision()) throw new Error('Signed-in account required');
  const items: SavedOutfitInput['items'] = [];
  for (const item of snapshot.items) {
    const source = item.image || item.imageUrl;
    if (!source) { items.push(item); continue; }
    const media = await prepareWardrobeItemMedia(owner, {
      id: item.id, imageUrl: source,
      thumbnailUrl: item.image && item.imageUrl ? item.imageUrl : undefined,
    });
    if (revision !== authEvents.getRevision()) throw new Error('Account changed during save');
    items.push({ ...item,
      ...(item.image ? { image: media.imageUrl } : {}),
      ...(item.imageUrl ? { imageUrl: item.image ? media.thumbnailUrl : media.imageUrl } : {}),
    });
  }
  if (revision !== authEvents.getRevision()) throw new Error('Account changed during save');
  const { data: saved, error } = await supabase.from('saved_outfits').upsert({
    id: snapshot.id, user_id: owner, items, date: snapshot.date,
    occasion: snapshot.occasion, caption: snapshot.caption || '', visibility: 'Everyone', is_ootd: false,
  }).select('id').single();
  if (error) throw error;
  if (revision !== authEvents.getRevision()) throw new Error('Account changed during save');
  if (saved?.id !== snapshot.id) throw new Error('Outfit save was not confirmed');
  return items;
}

export async function removeCloudOutfit(id: string): Promise<void> {
  const revision = authEvents.getRevision();
  const { data, error: sessionError } = await supabase.auth.getSession();
  const owner = data.session?.user.id;
  if (sessionError || !owner || revision !== authEvents.getRevision()) throw new Error('Signed-in account required');
  const { error } = await supabase.from('saved_outfits').delete().eq('id', id).eq('user_id', owner);
  if (error) throw error;
  if (revision !== authEvents.getRevision()) throw new Error('Account changed during removal');
}

export interface CloudSavedOutfit {
  id: string;
  date: string | null;
  occasion: string | null;
  items: unknown;
  created_at: string;
}

/** Load the entire owner's collection without the API's implicit row limit. */
export async function listCloudOutfits(): Promise<CloudSavedOutfit[]> {
  const revision = authEvents.getRevision();
  const { data, error: sessionError } = await supabase.auth.getSession();
  const owner = data.session?.user.id;
  if (sessionError || !owner || revision !== authEvents.getRevision()) throw new Error('Signed-in account required');
  const outfits: CloudSavedOutfit[] = [];
  const pageSize = 250;
  let cursor: string | undefined;
  for (;;) {
    let query = supabase.from('saved_outfits').select('id, date, occasion, items, created_at')
      .eq('user_id', owner).order('id', { ascending: false });
    if (cursor) query = query.lt('id', cursor);
    const { data: rows, error } = await query.limit(pageSize);
    if (error) throw error;
    if (revision !== authEvents.getRevision()) throw new Error('Account changed during collection read');
    if (!rows) throw new Error('Collection read was not confirmed');
    outfits.push(...rows);
    if (rows.length < pageSize) return outfits.sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? '') || b.id.localeCompare(a.id));
    const next = rows[rows.length - 1].id;
    if (typeof next !== 'string' || (cursor && next >= cursor)) throw new Error('Collection cursor did not advance');
    cursor = next;
  }
}
