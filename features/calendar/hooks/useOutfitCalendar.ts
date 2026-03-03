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

// ── Constants ──

export const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
export const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

export const OCCASIONS = [
    { id: 'work', label: 'Work', icon: '💼', color: '#3B82F6' },
    { id: 'casual', label: 'Casual', icon: '☕', color: '#22C55E' },
    { id: 'date', label: 'Date', icon: '💕', color: '#EC4899' },
    { id: 'party', label: 'Party', icon: '🎉', color: '#F59E0B' },
    { id: 'sport', label: 'Sport', icon: '🏃', color: '#8B5CF6' },
    { id: 'formal', label: 'Formal', icon: '🎩', color: '#1A1A1A' },
] as const;

// ── Types ──

export interface OutfitLog {
    date: string;
    items: Array<{ id: string; type: string; image: string; color?: string }>;
    occasion: string;
    note?: string;
    rating?: number;
}

// ── Helpers ──

export const getDaysInMonth = (year: number, month: number) =>
    new Date(year, month + 1, 0).getDate();

export const getFirstDayOfMonth = (year: number, month: number) =>
    new Date(year, month, 1).getDay();

export const formatDate = (year: number, month: number, day: number) =>
    `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

export const getOccasionColor = (occasionId: string) =>
    OCCASIONS.find(o => o.id === occasionId)?.color || '#6B7280';

// ── Hook ──

export function useOutfitCalendar() {
    const today = new Date();
    const todayStr = formatDate(today.getFullYear(), today.getMonth(), today.getDate());

    const [currentMonth, setCurrentMonth] = useState(today.getMonth());
    const [currentYear, setCurrentYear] = useState(today.getFullYear());
    const [selectedDate, setSelectedDate] = useState<string | null>(null);
    const [showDayModal, setShowDayModal] = useState(false);
    const [showLogModal, setShowLogModal] = useState(false);
    const [outfitLogs, setOutfitLogs] = useState<Record<string, OutfitLog>>({});
    const [todaysOutfit, setTodaysOutfit] = useState<OutfitLog | null>(null);
    const [streak, setStreak] = useState(0);
    const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);
    const [selectedItems, setSelectedItems] = useState<any[]>([]);
    const [selectedOccasion, setSelectedOccasion] = useState<string>('casual');

    // Load data
    const loadOutfitLogs = useCallback(async () => {
        try {
            const data = await AsyncStorage.getItem('outfitLogs');
            if (data) {
                const logs = JSON.parse(data);
                setOutfitLogs(logs);
                if (logs[todayStr]) setTodaysOutfit(logs[todayStr]);
                let streakCount = 0;
                const checkDate = new Date(today);
                while (logs[formatDate(checkDate.getFullYear(), checkDate.getMonth(), checkDate.getDate())]) {
                    streakCount++;
                    checkDate.setDate(checkDate.getDate() - 1);
                }
                setStreak(streakCount);
            }
        } catch (error) {
            console.error('Error loading outfit logs:', error);
        }
    }, [todayStr]);

    const loadWardrobeItems = useCallback(async () => {
        try {
            const data = await AsyncStorage.getItem('myWardrobeItems');
            if (data) setWardrobeItems(JSON.parse(data));
        } catch (error) {
            console.error('Error loading wardrobe:', error);
        }
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
            setShowLogModal(true);
        }
    };

    // Item selection
    const toggleItemSelection = (item: any) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        if (selectedItems.find(i => i.id === item.id)) {
            setSelectedItems(selectedItems.filter(i => i.id !== item.id));
        } else if (selectedItems.length < 6) {
            setSelectedItems([...selectedItems, item]);
        }
    };

    // Save
    const saveOutfit = async () => {
        if (selectedItems.length === 0) return;
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

        const dateToSave = selectedDate || todayStr;
        const newLog: OutfitLog = {
            date: dateToSave,
            items: selectedItems.map(item => ({
                id: item.id,
                type: item.type || item.category,
                image: item.image || item.imageUrl,
                color: item.color,
            })),
            occasion: selectedOccasion,
        };

        const updatedLogs = { ...outfitLogs, [dateToSave]: newLog };
        try {
            await AsyncStorage.setItem('outfitLogs', JSON.stringify(updatedLogs));
            setOutfitLogs(updatedLogs);
            if (dateToSave === todayStr) {
                setTodaysOutfit(newLog);
                setStreak(prev => prev + 1);
            }
            setShowLogModal(false);
            setSelectedItems([]);
            setSelectedOccasion('casual');
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
            if (dateStr === todayStr) setTodaysOutfit(null);
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
        setSelectedOccasion,
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
