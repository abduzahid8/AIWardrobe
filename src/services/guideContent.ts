import { supabase } from '../../lib/supabase';
import { queryClient } from '../lib/queryClient';

export interface GuideContent {
  title: string;
  subtitle: string;
  cta_text: string;
  cta_url: string | null;
  hero_image_url: string | null;
  background_color: string;
}
const key = ['guide-page'] as const;
export const getCachedGuideContent = () => queryClient.getQueryData<GuideContent>(key) ?? null;

/** Shared by idle preload, Inspo and the full guide. Concurrent opens use one request. */
export function loadGuideContent(): Promise<GuideContent> {
  return queryClient.fetchQuery({
    queryKey: key,
    staleTime: 5 * 60 * 1000,
    queryFn: async ({ signal }) => {
      const { data, error } = await supabase.from('guide_page').select('*')
        .eq('is_active', true).abortSignal(signal).single();
      if (error) throw error;
      if (!data) throw new Error('Guide content unavailable');
      return data as GuideContent;
    },
  });
}
