import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../../lib/supabase';
import useWardrobeStore from '../../store/wardrobeStore';
import { generateOutfitsLocally, type LocalClothingItem } from '../../src/services/outfitGenerationService';
import { getRotationHistory, makeRotationScope } from '../../src/services/recentOutfitIds';
import { productIdentityKeys } from '../../backend/supabase/functions/_shared/outfitDiversity';

const makeItem = (slot: 'top' | 'bottom' | 'shoes', index: number): LocalClothingItem => ({
    id: `${slot}-${index}`,
    userId: 'rotation-user',
    imageUrl: `https://example.com/${slot}-${index}.jpg`,
    category: slot as any,
    subCategory: slot === 'top' ? 'shirt' : slot === 'bottom' ? 'trousers' : 'loafers',
    primaryColor: ['navy', 'white', 'brown'][index % 3],
    colorHex: '#000000',
    pattern: 'solid',
    material: 'cotton',
    seasons: [],
    occasions: [],
    wearCount: 0,
    lastWornAt: null,
    isFavorite: false,
    createdAt: '2026-01-01T00:00:00Z',
    updatedAt: '2026-01-01T00:00:00Z',
    macroCategory: slot,
    name: `${slot} ${index}`,
});

const keysInDeck = (outfits: Array<{ items: Array<{ id: string; name?: string; imageUrl?: string }> }>) =>
    new Set(outfits.flatMap((outfit) => outfit.items.flatMap((item) => productIdentityKeys(item))));

beforeEach(async () => {
    await AsyncStorage.clear();
    useWardrobeStore.setState({
        items: [
            ...Array.from({ length: 9 }, (_, i) => makeItem('top', i)),
            ...Array.from({ length: 9 }, (_, i) => makeItem('bottom', i)),
            ...Array.from({ length: 9 }, (_, i) => makeItem('shoes', i)),
        ],
    });
    (supabase.auth.getSession as jest.Mock).mockResolvedValue({
        data: { session: { user: { id: 'rotation-user' }, access_token: 'test-token' } },
        error: null,
    });
});

test('rotates three consecutive local generations across physical products', async () => {
    const scope = makeRotationScope('rotation-user', 'wardrobe', 'casual', 'Everyday');
    const decks: Array<Array<{ items: Array<{ id: string; name?: string; imageUrl?: string }> }>> = [];

    for (let generation = 0; generation < 3; generation += 1) {
        const result = await generateOutfitsLocally({
            stylePreferences: 'casual',
            occasion: 'Everyday',
            limit: 3,
        });
        expect(result.success).toBe(true);
        expect(result.outfits).toHaveLength(3);
        decks.push(result.outfits);
    }

    const deckKeys = decks.map(keysInDeck);
    expect([...deckKeys[0]].some((key) => deckKeys[1].has(key))).toBe(false);
    expect([...deckKeys[1]].some((key) => deckKeys[2].has(key))).toBe(false);
    expect([...deckKeys[0]].some((key) => deckKeys[2].has(key))).toBe(false);

    const history = await getRotationHistory(scope);
    expect(history.previousDeckProductKeys.length).toBeGreaterThan(0);
    expect(history.olderRecentProductKeys.length).toBeGreaterThan(0);
});

test('restores only the minimum previous-deck shoe when five physical shoes are available', async () => {
    useWardrobeStore.setState({
        items: [
            ...Array.from({ length: 6 }, (_, i) => makeItem('top', i)),
            ...Array.from({ length: 6 }, (_, i) => makeItem('bottom', i)),
            ...Array.from({ length: 5 }, (_, i) => makeItem('shoes', i)),
        ],
    });

    const first = await generateOutfitsLocally({ stylePreferences: 'casual', occasion: 'Everyday', limit: 3 });
    const second = await generateOutfitsLocally({ stylePreferences: 'casual', occasion: 'Everyday', limit: 3 });
    expect(first.success).toBe(true);
    expect(second.success).toBe(true);

    const shoes = (deck: typeof first.outfits) =>
        new Set(deck.flatMap((outfit) => outfit.items.filter((item) => item.macroCategory === 'shoes').map((item) => item.id)));
    const firstShoes = shoes(first.outfits);
    const secondShoes = shoes(second.outfits);
    expect(secondShoes.size).toBe(3);
    expect([...secondShoes].filter((id) => firstShoes.has(id))).toHaveLength(1);
});
