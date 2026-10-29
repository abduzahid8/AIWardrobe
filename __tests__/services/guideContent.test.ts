import { getCachedGuideContent, loadGuideContent } from '../../src/services/guideContent';
import { queryClient } from '../../src/lib/queryClient';
import { supabase } from '../../lib/supabase';

jest.mock('../../lib/supabase', () => ({ supabase: { from: jest.fn() } }));
beforeEach(() => { queryClient.clear(); jest.clearAllMocks(); });
afterEach(() => queryClient.clear());

test('guide preload and simultaneous screen opens share a request and warm cache', async () => {
  let finish!: (value: unknown) => void;
  const response = new Promise(resolve => { finish = resolve; });
  const query = { select: jest.fn().mockReturnThis(), eq: jest.fn().mockReturnThis(),
    abortSignal: jest.fn().mockReturnThis(), single: jest.fn(() => response) };
  (supabase.from as jest.Mock).mockReturnValue(query);
  const preload = loadGuideContent();
  const screen = loadGuideContent();
  expect(supabase.from).toHaveBeenCalledTimes(1);
  const content = { title: 'Shared guide', subtitle: '', cta_text: 'Open', cta_url: null, hero_image_url: null, background_color: '#fff' };
  finish({ data: content, error: null });
  expect(await preload).toEqual(content);
  expect(await screen).toEqual(content);
  expect(getCachedGuideContent()).toEqual(content);
  expect(await loadGuideContent()).toEqual(content);
  expect(supabase.from).toHaveBeenCalledTimes(1);
  queryClient.clear();
  expect(getCachedGuideContent()).toBeNull();
});
