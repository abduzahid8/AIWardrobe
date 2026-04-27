/**
 * useOutfitLogs — manages outfit log CRUD with AsyncStorage persistence
 */

import { useState, useEffect, useCallback } from 'react';
import { Alert } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useTranslation } from 'react-i18next';
import type { OutfitLog } from './useCalendarState';

const STORAGE_KEY = '@outfit_logs';

export function useOutfitLogs() {
    const [outfitLogs, setOutfitLogs] = useState<Record<string, OutfitLog>>({});

    // Load logs from storage
    useEffect(() => {
        (async () => {
            try {
                const stored = await AsyncStorage.getItem(STORAGE_KEY);
                if (stored) setOutfitLogs(JSON.parse(stored));
            } catch (err) {
                console.error('Failed to load outfit logs:', err);
            }
        })();
    }, []);

    const saveLogs = useCallback(async (logs: Record<string, OutfitLog>) => {
        try {
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(logs));
            setOutfitLogs(logs);
        } catch (err) {
            console.error('Failed to save outfit logs:', err);
        }
    }, []);

    const saveOutfit = useCallback(
        async (
            dateStr: string,
            selectedItems: Array<{ id: string; type: string; image: string; color?: string }>,
            occasion: string,
            note?: string,
            rating?: number
        ) => {
            if (selectedItems.length === 0) {
                const { t } = useTranslation();
                Alert.alert(t('outfitLogs.noItems'), t('outfitLogs.selectAtLeastOne'));
                return false;
            }
            const newLog: OutfitLog = {
                date: dateStr,
                items: selectedItems,
                occasion,
                note,
                rating,
            };
            const updated = { ...outfitLogs, [dateStr]: newLog };
            await saveLogs(updated);
            return true;
        },
        [outfitLogs, saveLogs]
    );

    const deleteOutfitLog = useCallback(
        async (dateStr: string) => {
            const updated = { ...outfitLogs };
            delete updated[dateStr];
            await saveLogs(updated);
        },
        [outfitLogs, saveLogs]
    );

    const getMonthlyStats = useCallback(
        (year: number, month: number) => {
            const entries = Object.entries(outfitLogs).filter(([date]) => {
                const d = new Date(date);
                return d.getFullYear() === year && d.getMonth() === month;
            });
            return {
                totalOutfits: entries.length,
                uniqueItems: new Set(entries.flatMap(([, log]) => log.items.map((i) => i.id))).size,
            };
        },
        [outfitLogs]
    );

    return {
        outfitLogs,
        saveOutfit,
        deleteOutfitLog,
        getMonthlyStats,
    };
}
