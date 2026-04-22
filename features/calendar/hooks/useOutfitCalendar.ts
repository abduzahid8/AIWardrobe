/**
 * useOutfitCalendar — All state and business logic for the Outfit Calendar feature.
 *
 * Extracted from the monolithic OutfitCalendarScreen (1,127 lines → ~150 line screen).
 */

import { useState, useCallback } from 'react';
import { Alert } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Asset } from 'expo-asset';

import {
    type OutfitLog,
    type OutfitItem,
    type OccasionId,
    type WardrobeItem,
    OCCASIONS,
    MONTHS,
    WEEKDAYS,
    formatDate,
    getDaysInMonth,
    getFirstDayOfMonth,
    getOccasionColor,
    createOutfitLog,
    wardrobeToOutfitItem,
} from '../types';
import { supabase } from '../../../lib/supabase';

// Re-export everything consumers need
export {
    type OutfitLog,
    type OutfitItem,
    type OccasionId,
    type WardrobeItem,
    OCCASIONS,
    MONTHS,
    WEEKDAYS,
    formatDate,
    getDaysInMonth,
    getFirstDayOfMonth,
    getOccasionColor,
};

// ── Hook ──

export function useOutfitCalendar() {
    const today = new Date();
    const todayStr = formatDate(today.getFullYear(), today.getMonth(), today.getDate());

    const [currentMonth, setCurrentMonth] = useState(today.getMonth());
    const [currentYear, setCurrentYear] = useState(today.getFullYear());
    const [selectedDate, setSelectedDate] = useState<string | null>(null);
    const [showDayModal, setShowDayModal] = useState(false);
    const [showLogModal, setShowLogModal] = useState(false);
    const [showAddPopover, setShowAddPopover] = useState(false);
    const [outfitLogs, setOutfitLogs] = useState<Record<string, OutfitLog>>({});
    const [todaysOutfit, setTodaysOutfit] = useState<OutfitLog | null>(null);
    const [streak, setStreak] = useState(0);
    const [wardrobeItems, setWardrobeItems] = useState<WardrobeItem[]>([]);
    const [selectedItems, setSelectedItems] = useState<WardrobeItem[]>([]);
    const [selectedOccasion, setSelectedOccasion] = useState<OccasionId>('casual');

    const calculateStreakFromLogs = useCallback((logs: Record<string, OutfitLog>) => {
        let streakCount = 0;
        const [year, month, day] = todayStr.split('-').map(Number);
        const checkDate = new Date(year, month - 1, day);

        while (logs[formatDate(checkDate.getFullYear(), checkDate.getMonth(), checkDate.getDate())]) {
            streakCount++;
            checkDate.setDate(checkDate.getDate() - 1);
        }

        return streakCount;
    }, [todayStr]);

    // Load data
    const loadOutfitLogs = useCallback(async () => {
        try {
            const data = await AsyncStorage.getItem('outfitLogs');
            if (data) {
                const logs = JSON.parse(data);
                setOutfitLogs(logs);
                setTodaysOutfit(logs[todayStr] ?? null);
                setStreak(calculateStreakFromLogs(logs));
            } else {
                setOutfitLogs({});
                setTodaysOutfit(null);
                setStreak(0);
            }
        } catch (error) {
            console.error('Error loading outfit logs:', error);
        }
    }, [calculateStreakFromLogs, todayStr]);

    const loadWardrobeItems = useCallback(async () => {
        try {
            // Primary: load directly from Supabase clothing_items
            const { data, error } = await supabase
                .from('clothing_items')
                .select('id, type, category, image_url, color, name, primary_color')
                .order('created_at', { ascending: false });

            if (!error && data && data.length > 0) {
                const mapped: WardrobeItem[] = data.map((row: any) => ({
                    id: row.id,
                    type: row.type || row.category || '',
                    image: row.image_url || '',
                    imageUrl: row.image_url || '',
                    color: Array.isArray(row.color) ? row.color[0] : (row.color || row.primary_color || ''),
                    name: row.name || '',
                    category: row.category || '',
                }));
                setWardrobeItems(mapped);
                return;
            }
        } catch {
            // ignore, fall through to AsyncStorage
        }

        // Fallback: read from local AsyncStorage
        try {
            const stored = await AsyncStorage.getItem('myWardrobeItems');
            if (stored) setWardrobeItems(JSON.parse(stored));
        } catch (error) {
            console.error('Error loading wardrobe:', error);
        }
    }, []);

    const resetLogDraft = useCallback(() => {
        setSelectedItems([]);
        setSelectedOccasion('casual');
    }, []);

    const closeLogModal = useCallback(() => {
        setShowLogModal(false);
        resetLogDraft();
    }, [resetLogDraft]);

    const openLogModalForDate = useCallback((dateStr: string) => {
        setSelectedDate(dateStr);
        setShowAddPopover(false);
        resetLogDraft();
        setShowLogModal(true);
    }, [resetLogDraft]);

    const resolveWardrobeImage = useCallback(async (item: WardrobeItem) => {
        const directImage = item.image || item.imageUrl || '';
        if (directImage) return directImage;

        if (item.localImage != null) {
            const asset = Asset.fromModule(item.localImage);
            await asset.downloadAsync();
            return asset.localUri || asset.uri || '';
        }

        return '';
    }, []);

    useFocusEffect(useCallback(() => {
        loadOutfitLogs();
        loadWardrobeItems();
    }, [loadOutfitLogs, loadWardrobeItems]));

    // Navigation
    const goToPrevMonth = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        if (currentMonth === 0) {
            setCurrentMonth(11);
            setCurrentYear(currentYear - 1);
        } else {
            setCurrentMonth(currentMonth - 1);
        }
    };

    const goToNextMonth = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        if (currentMonth === 11) {
            setCurrentMonth(0);
            setCurrentYear(currentYear + 1);
        } else {
            setCurrentMonth(currentMonth + 1);
        }
    };

    // Day interaction
    const handleDayPress = (day: number) => {
        const dateStr = formatDate(currentYear, currentMonth, day);
        setSelectedDate(dateStr);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

        if (outfitLogs[dateStr]) {
            setShowDayModal(true);
        } else if (dateStr === todayStr || new Date(dateStr) > today) {
            setShowAddPopover(true);
        }
    };

    // Item selection
    const toggleItemSelection = (item: WardrobeItem) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setSelectedItems(prevSelectedItems => {
            if (prevSelectedItems.find(i => i.id === item.id)) {
                return prevSelectedItems.filter(i => i.id !== item.id);
            }
            if (prevSelectedItems.length >= 6) {
                return prevSelectedItems;
            }
            return [...prevSelectedItems, item];
        });
    };

    // Save
    const saveOutfit = async () => {
        if (selectedItems.length === 0) return;
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

        const dateToSave = selectedDate || todayStr;
        const outfitItems = await Promise.all(
            selectedItems.map(async item => ({
                ...wardrobeToOutfitItem(item),
                image: await resolveWardrobeImage(item),
            }))
        );
        const newLog = createOutfitLog(dateToSave, outfitItems, selectedOccasion);

        const updatedLogs = { ...outfitLogs, [dateToSave]: newLog };
        try {
            await AsyncStorage.setItem('outfitLogs', JSON.stringify(updatedLogs));
            setOutfitLogs(updatedLogs);
            setTodaysOutfit(updatedLogs[todayStr] ?? null);
            setStreak(calculateStreakFromLogs(updatedLogs));
            closeLogModal();
        } catch (error) {
            console.error('Error saving outfit:', error);
        }
    };

    // Delete
    const deleteOutfitLog = async (dateStr: string) => {
        const updatedLogs = { ...outfitLogs };
        delete updatedLogs[dateStr];
        try {
            await AsyncStorage.setItem('outfitLogs', JSON.stringify(updatedLogs));
            setOutfitLogs(updatedLogs);
            setTodaysOutfit(updatedLogs[todayStr] ?? null);
            setStreak(calculateStreakFromLogs(updatedLogs));
            setShowDayModal(false);
        } catch (error) {
            console.error('Error deleting outfit:', error);
        }
    };

    const confirmDelete = (dateStr: string) => {
        Alert.alert('Delete Outfit', 'Are you sure?', [
            { text: 'Cancel', style: 'cancel' },
            { text: 'Delete', style: 'destructive', onPress: () => deleteOutfitLog(dateStr) },
        ]);
    };

    // Stats
    const getMonthlyStats = () => {
        const monthKey = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}`;
        const monthLogs = Object.entries(outfitLogs)
            .filter(([date]) => date.startsWith(monthKey));
        return {
            logged: monthLogs.length,
            total: getDaysInMonth(currentYear, currentMonth),
        };
    };

    return {
        // State
        today,
        todayStr,
        currentMonth,
        currentYear,
        selectedDate,
        showDayModal,
        showLogModal,
        showAddPopover,
        outfitLogs,
        todaysOutfit,
        streak,
        wardrobeItems,
        selectedItems,
        selectedOccasion,

        // Actions
        setSelectedDate,
        setShowDayModal,
        setShowLogModal,
        setShowAddPopover,
        setSelectedOccasion,
        openLogModalForDate,
        closeLogModal,
        goToPrevMonth,
        goToNextMonth,
        handleDayPress,
        toggleItemSelection,
        saveOutfit,
        confirmDelete,
        getMonthlyStats,
        getOccasionColor,
    };
}
