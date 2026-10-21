import { createStore } from 'zustand/vanilla';
import { createOutfitsSlice } from '../../store/slices/wardrobeOutfitsSlice';
import { removeCloudOutfit } from '../../src/services/savedOutfitService';
import { authEvents } from '../../store/authEvents';
import type { WardrobeState } from '../../store/wardrobeStore';
jest.mock('../../src/services/savedOutfitService', () => ({ removeCloudOutfit: jest.fn() }));
const fixture = () => createStore<WardrobeState>()((set, get, api) => ({ ...createOutfitsSlice(set, get, api), outfits: [{ id: 'outfit', saved: true }] } as unknown as WardrobeState));
beforeEach(() => { jest.clearAllMocks(); authEvents.emitLogin('owner'); });

test('a failed cloud unsave preserves the saved outfit locally', async () => {
  (removeCloudOutfit as jest.Mock).mockRejectedValueOnce(new Error('offline'));
  const state = fixture();
  await expect(state.getState().unsaveOutfit('outfit')).rejects.toThrow('offline');
  expect(state.getState().outfits[0].saved).toBe(true);
});

test('confirmed unsave updates local state, while old-account completion is discarded', async () => {
  (removeCloudOutfit as jest.Mock).mockResolvedValueOnce(undefined);
  const state = fixture(); await state.getState().unsaveOutfit('outfit');
  expect(state.getState().outfits[0].saved).toBe(false);
  let finish!: () => void;
  (removeCloudOutfit as jest.Mock).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
  const pending = state.getState().unsaveOutfit('outfit');
  authEvents.emitLogout(); authEvents.emitLogin('other');
  state.setState({ outfits: [{ id: 'outfit', saved: true } as never] });
  finish(); await pending;
  expect(state.getState().outfits[0].saved).toBe(true);
});
