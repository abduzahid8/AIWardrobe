import { act, renderHook } from '@testing-library/react-native';
import { useWardrobeLoader } from '../../features/outfit-generator/hooks/useWardrobeLoader';
import useWardrobeStore from '../../store/wardrobeStore';
import { authEvents } from '../../store/authEvents';

jest.mock('../../store/wardrobeStore', () => ({ __esModule: true,
  default: jest.requireActual('zustand').create(() => ({ items: [], rehydrateFromCloud: jest.fn(() => new Promise(() => {})) })) }));
jest.mock('../../store/auth', () => ({ __esModule: true,
  default: jest.requireActual('zustand').create(() => ({ session: { user: { id: 'owner' } } })) }));
jest.mock('../../hooks/useShopCatalog', () => ({ useShopCatalog: () => ({ items: [], loading: false }) }));

beforeEach(() => {
  authEvents.emitLogin('owner');
  useWardrobeStore.setState({ items: [
    { id: 'mine', userId: 'owner', imageUrl: 'file://mine', category: 'top', subCategory: 'shirt' },
    { id: 'foreign', userId: 'someone-else', imageUrl: 'file://foreign', category: 'top' },
  ] as never });
  (useWardrobeStore.getState().rehydrateFromCloud as jest.Mock).mockClear();
});

test('cached picker choices are usable without waiting for cloud sync', () => {
  const { result } = renderHook(() => useWardrobeLoader('wardrobe'));
  expect(result.current.wardrobeItems.map(item => item.id)).toEqual(['mine']);
  expect(result.current.loading).toBe(false);
  expect(useWardrobeStore.getState().rehydrateFromCloud).toHaveBeenCalledTimes(1);
  act(() => useWardrobeStore.setState({ items: [...useWardrobeStore.getState().items] }));
  expect(useWardrobeStore.getState().rehydrateFromCloud).toHaveBeenCalledTimes(1);
});

test('logout immediately hides cached personal choices', () => {
  const { result } = renderHook(() => useWardrobeLoader('wardrobe'));
  act(() => authEvents.emitLogout());
  expect(result.current.wardrobeItems).toEqual([]);
});
