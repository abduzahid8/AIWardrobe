/**
 * recentOutfitIds.test.ts — rolling exclusion window for outfit rotation.
 *
 * Covers: id filtering, FIFO cap, persistence, hydration order, and the
 * record-after-hydrate merge (a first record must never clobber stored ids).
 */
import AsyncStorage from '@react-native-async-storage/async-storage';

const STORAGE_KEY = 'aiExcludeShownItemIds';

type Module = typeof import('../../src/services/recentOutfitIds');

function loadModule(): Module {
    let mod!: Module;
    jest.isolateModules(() => {
        mod = require('../../src/services/recentOutfitIds');
    });
    return mod;
}

const outfit = (...ids: string[]) => ({ items: ids.map((id) => ({ id })) });

beforeEach(() => {
    jest.clearAllMocks();
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
    (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
});

test('records real pool ids and skips synthesized/placeholder ids', async () => {
    const mod = loadModule();
    mod.recordShownOutfits([
        outfit('aaa', 'placeholder_shoes', 'local_12345', 'item_0', ''),
        outfit('bbb'),
    ]);
    const ids = await mod.getRecentShownIds();
    expect(ids).toEqual(['aaa', 'bbb']);
});

test('dedupes repeated ids across decks', async () => {
    const mod = loadModule();
    mod.recordShownOutfits([outfit('aaa', 'bbb')]);
    mod.recordShownOutfits([outfit('bbb', 'ccc')]);
    const ids = await mod.getRecentShownIds();
    expect(ids).toEqual(['aaa', 'bbb', 'ccc']);
});

test('FIFO-caps the window at 150 ids', async () => {
    const mod = loadModule();
    const decks = Array.from({ length: 6 }, (_, d) =>
        outfit(...Array.from({ length: 30 }, (_, i) => `id_${d * 30 + i}`)),
    );
    decks.forEach((deck) => mod.recordShownOutfits([deck]));
    const ids = await mod.getRecentShownIds();
    expect(ids.length).toBe(150);
    expect(ids[0]).toBe('id_30'); // oldest 30 dropped
    expect(ids[ids.length - 1]).toBe('id_179');
});

test('hydrates stored window from AsyncStorage on first read', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(
        JSON.stringify(['old1', 'placeholder_shoes', 'old2']),
    );
    const mod = loadModule();
    const ids = await mod.getRecentShownIds();
    expect(ids).toEqual(['old1', 'old2']);
});

test('first record after hydration merges with stored ids and persists both', async () => {
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(JSON.stringify(['old1', 'old2']));
    const mod = loadModule();
    mod.recordShownOutfits([outfit('new1')]);
    const ids = await mod.getRecentShownIds(); // awaits write chain
    expect(ids).toEqual(['old1', 'old2', 'new1']);
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(
        STORAGE_KEY,
        JSON.stringify(['old1', 'old2', 'new1']),
    );
});

test('does not persist when nothing new was added', async () => {
    const mod = loadModule();
    mod.recordShownOutfits([outfit('aaa')]);
    await mod.getRecentShownIds();
    (AsyncStorage.setItem as jest.Mock).mockClear();
    mod.recordShownOutfits([outfit('aaa', 'placeholder_shoes')]);
    await mod.getRecentShownIds();
    expect(AsyncStorage.setItem).not.toHaveBeenCalled();
});

test('keeps exclusions and pending writes isolated between accounts', async () => {
    const mod = loadModule();
    mod.recordShownOutfits([outfit('account-a-item')], 'account-a');
    mod.recordShownOutfits([outfit('account-b-item')], 'account-b');
    expect(await mod.getRecentShownIds('account-a')).toEqual(['account-a-item']);
    expect(await mod.getRecentShownIds('account-b')).toEqual(['account-b-item']);
    expect(AsyncStorage.setItem).toHaveBeenCalledWith(`${STORAGE_KEY}::account-a`, JSON.stringify(['account-a-item']));
    expect(await mod.getRecentShownIds('signed-out')).toEqual([]);
});

test('records physical identity and rehydrates normalized names and images', async () => {
    const mod=loadModule();
    mod.recordShownOutfits([{items:[{id:'shoe-original',name:'Brown Leather Loafers',imageUrl:'https://cdn.test/loafers.jpg?width=200'}]}],'identity-user');
    const keys=await mod.getRecentShownProductKeys('identity-user');
    expect(keys).toEqual(expect.arrayContaining(['id:shoe-original','n:brown leather loafers','i:cdn.test/loafers.jpg']));
    const saved=(AsyncStorage.setItem as jest.Mock).mock.calls.find(([key])=>key.endsWith('identity-user::products'))![1];
    (AsyncStorage.getItem as jest.Mock).mockImplementation(async (key:string)=>key.endsWith('identity-user::products')?saved:null);
    expect(await loadModule().getRecentShownProductKeys('identity-user')).toEqual(keys);
    expect(await loadModule().getRecentShownProductKeys('another-user')).toEqual([]);
});
