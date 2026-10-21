import { authEvents } from '../../store/authEvents';
/**
 * Focused regression tests for the outfit calendar hook.
 */

jest.mock('@react-navigation/native', () => ({
    useFocusEffect: (_effect: () => void | (() => void)) => undefined,
}));

jest.mock('../../lib/supabase', () => ({
    supabase: {
        from: jest.fn(() => ({
            select: jest.fn().mockReturnThis(),
            order: jest.fn().mockResolvedValue({ data: [], error: null }),
        })),
    },
}));

jest.mock('expo-asset', () => ({
    Asset: {
        fromModule: jest.fn(() => ({
            downloadAsync: jest.fn().mockResolvedValue(undefined),
            localUri: 'file://mock-asset.png',
            uri: 'file://mock-asset.png',
        })),
    },
}));

import { Alert } from 'react-native';
import * as Haptics from 'expo-haptics';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { renderHook, act } from '@testing-library/react-native';
import { useOutfitCalendar } from '../../features/calendar/hooks/useOutfitCalendar';

describe('useOutfitCalendar', () => {
    beforeEach(async () => {
        jest.clearAllMocks();
        await AsyncStorage.clear();
        authEvents.emitLogin('calendar-owner');
    });

    it('saves selected items to the explicitly selected date', async () => {
        const { result } = renderHook(() => useOutfitCalendar());

        act(() => {
            result.current.setSelectedDate('2026-04-16');
            result.current.toggleItemSelection({
                id: 'top-1',
                type: 'ribbed knit top',
                image: 'https://example.com/top.png',
                name: 'Ribbed Knit Top',
            });
        });

        await act(async () => {
            await result.current.saveOutfit();
        });

        const lastSetCall = (AsyncStorage.setItem as jest.Mock).mock.calls.at(-1);
        const storedLogs = JSON.parse(lastSetCall[1]);

        expect(storedLogs['2026-04-16']).toMatchObject({
            date: '2026-04-16',
            occasion: 'casual',
            items: [
                {
                    id: 'top-1',
                    type: 'ribbed knit top',
                    image: 'https://example.com/top.png',
                    name: 'Ribbed Knit Top',
                },
            ],
        });
    });

    it('converts local asset picks into persistable URIs before saving', async () => {
        const { result } = renderHook(() => useOutfitCalendar());

        act(() => {
            result.current.setSelectedDate('2026-04-17');
            result.current.toggleItemSelection({
                id: 'shop-top-1',
                type: 'top',
                image: '',
                imageUrl: '',
                localImage: 'https://example.com/test-image.jpg',
                name: 'Ribbed Knit Top',
            });
        });

        await act(async () => {
            await result.current.saveOutfit();
        });

        const lastSetCall = (AsyncStorage.setItem as jest.Mock).mock.calls.at(-1);
        const storedLogs = JSON.parse(lastSetCall[1]);

        expect(storedLogs['2026-04-17'].items[0].image).toBe('file://mock-asset.png');
    });

    it('supports same-tick replacement updates without keeping both items selected', () => {
        const { result } = renderHook(() => useOutfitCalendar());

        act(() => {
            result.current.toggleItemSelection({
                id: 'old-top',
                type: 'top',
                image: 'https://example.com/old-top.png',
                name: 'Old Top',
            });
        });

        act(() => {
            result.current.toggleItemSelection({
                id: 'old-top',
                type: 'top',
                image: 'https://example.com/old-top.png',
                name: 'Old Top',
            });
            result.current.toggleItemSelection({
                id: 'new-top',
                type: 'top',
                image: 'https://example.com/new-top.png',
                name: 'New Top',
            });
        });

        expect(result.current.selectedItems.map(item => item.id)).toEqual(['new-top']);
    });
});


it('keeps the draft open and reports a storage failure without success feedback', async () => {
    const alert = jest.spyOn(Alert, 'alert').mockImplementation(() => undefined);
    const error = jest.spyOn(console, 'error').mockImplementation(() => undefined);
    const { result } = renderHook(() => useOutfitCalendar());
    act(() => {
        result.current.openLogModalForDate('2026-10-16');
        result.current.toggleItemSelection({ id: 'top', type: 'top', image: 'file://top.png' });
    });
    (AsyncStorage.setItem as jest.Mock).mockRejectedValueOnce(new Error('disk full'));
    (Haptics.notificationAsync as jest.Mock).mockClear();
    await act(async () => { await result.current.saveOutfit(); });
    expect(result.current.showLogModal).toBe(true);
    expect(result.current.selectedItems).toHaveLength(1);
    expect(result.current.outfitLogs).toEqual({});
    expect(alert).toHaveBeenCalledWith('common.error', 'calendar.persistenceError');
    expect(Haptics.notificationAsync).not.toHaveBeenCalled();
    alert.mockRestore();
    error.mockRestore();
});
