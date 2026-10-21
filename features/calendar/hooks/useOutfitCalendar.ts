/**
 * useOutfitCalendar — All state and business logic for the Outfit Calendar feature.
 *
 * Extracted from the monolithic OutfitCalendarScreen (1,127 lines → ~150 line screen).
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { Alert } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import { Asset } from 'expo-asset';
import { useTranslation } from 'react-i18next';

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
import { fetchItemsFromServer } from '../../../store/wardrobeSyncService';
import { authEvents } from '../../../store/authEvents';
import { readCalendarLogs, updateCalendarLogs } from '../../../src/services/calendarStorage';
import { getMacroCategory } from '../../../src/utils/categoryMapper';

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
    const { t } = useTranslation();
    const today = new Date();
    const todayStr = formatDate(today.getFullYear(), today.getMonth(), today.getDate());

    const [currentMonth, setCurrentMonth] = useState(today.getMonth());
    const [currentYear, setCurrentYear] = useState(today.getFullYear());
    const [selectedDate, setSelectedDate] = useState<string | null>(null);
    const [showDayModal, setShowDayModal] = useState(false);
    const [showLogModal, setShowLogModal] = useState(false);
    const [showAddPopover, setShowAddPopover] = useState(false);
    const [logsLoadFailed, setLogsLoadFailed] = useState(false);
    const logsRequest = useRef(0);
    const mounted = useRef(true);
    useEffect(() => { mounted.current = true; return () => { mounted.current = false; logsRequest.current++; }; }, []);
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
        const revision = authEvents.getRevision(), owner = authEvents.getUserId(), request = ++logsRequest.current;
        const current = () => mounted.current && request === logsRequest.current &&
            revision === authEvents.getRevision() && owner === authEvents.getUserId();
        try {
            const logs = await readCalendarLogs();
            if (!current()) return;
            setLogsLoadFailed(false);
            if (Object.keys(logs).length) {
                setOutfitLogs(logs);
                setTodaysOutfit(logs[todayStr] ?? null);
                setStreak(calculateStreakFromLogs(logs));
            } else {
                setOutfitLogs({});
                setTodaysOutfit(null);
                setStreak(0);
            }
        } catch (error) {
            if (current()) setLogsLoadFailed(true);
            console.error('Error loading outfit logs:', error);
        }
    }, [calculateStreakFromLogs, todayStr]);

    const loadWardrobeItems = useCallback(async () => {
        const revision = authEvents.getRevision();
        try {
            let items = await fetchItemsFromServer();
            if (revision !== authEvents.getRevision()) return;
            if (items === null) {
                // Offline selection may use the current owner's typed cache,
                // never the legacy unscoped myWardrobeItems key.
                const auth = require('../../../store/auth').default.getState();
                const owner = auth.session?.user.id;
                if (!owner || !authEvents.canApplyToUser(owner)) return;
                const cache: { items: import('../../../src/types/domain').ClothingItem[] } =
                    require('../../../store/wardrobeStore').default.getState();
                items = cache.items.filter(item => item.userId === owner);
            }
            setWardrobeItems(items.map(item => ({
                id: item.id,
                type: item.subCategory || item.category || '',
                image: item.imageUrl || '',
                imageUrl: item.imageUrl || '',
                color: item.primaryColor || '',
                name: item.name || '',
                category: item.category || '',
            })));
        } catch (error) {
            console.error('Error loading wardrobe:', error);
        }
    }, []);

    useEffect(() => {
        let revision = authEvents.getRevision();
        const clearPersonalState = () => {
            const next = authEvents.getRevision();
            if (next === revision) return;
            revision = next;
            logsRequest.current++;
            setLogsLoadFailed(false);
            setWardrobeItems([]);
            setOutfitLogs({});
            setTodaysOutfit(null);
            setStreak(0);
            setSelectedItems([]);
            setSelectedDate(null);
            setSelectedOccasion('casual');
            setShowDayModal(false);
            setShowLogModal(false);
            setShowAddPopover(false);
        };
        const removeLogin = authEvents.onLogin(clearPersonalState);
        const removeLogout = authEvents.onLogout(clearPersonalState);
        return () => { removeLogin(); removeLogout(); };
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
        return () => { logsRequest.current++; };
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
        } else if (dateStr >= todayStr) {
            setShowAddPopover(true);
        }
    };

    // Item selection — enforces one item per macro-category
    const toggleItemSelection = (item: WardrobeItem) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setSelectedItems(prevSelectedItems => {
            if (prevSelectedItems.find(i => i.id === item.id)) {
                return prevSelectedItems.filter(i => i.id !== item.id);
            }
            if (prevSelectedItems.length >= 6) {
                return prevSelectedItems;
            }
            const macro = getMacroCategory(item.type || item.category || 'other');
            const hasExistingInCategory = prevSelectedItems.some(
                existing => getMacroCategory(existing.type || existing.category || 'other') === macro
            );
            if (hasExistingInCategory) {
                return prevSelectedItems.map(existing =>
                    getMacroCategory(existing.type || existing.category || 'other') === macro
                        ? item
                        : existing
                );
            }
            return [...prevSelectedItems, item];
        });
    };

    // Save
    const saveOutfit = async () => {
        if (selectedItems.length === 0) return;
        const dateToSave = selectedDate || todayStr;
        const revision = authEvents.getRevision();
        const current = () => mounted.current && revision === authEvents.getRevision();
        try {
            const outfitItems = await Promise.all(
                selectedItems.map(async item => ({
                    ...wardrobeToOutfitItem(item),
                    image: await resolveWardrobeImage(item),
                }))
            );
            const newLog = createOutfitLog(dateToSave, outfitItems, selectedOccasion);

            const updatedLogs = await updateCalendarLogs(logs => { logs[dateToSave] = newLog; }, revision);
            if (!current()) return;
            logsRequest.current++;
            setLogsLoadFailed(false);
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            setOutfitLogs(updatedLogs);
            setTodaysOutfit(updatedLogs[todayStr] ?? null);
            setStreak(calculateStreakFromLogs(updatedLogs));
            closeLogModal();
        } catch (error) {
            console.error('Error saving outfit:', error);
            if (!current()) return;
            Alert.alert(t('common.error'), t('calendar.persistenceError'));
        }
    };

    // Delete
    const deleteOutfitLog = async (dateStr: string) => {
        const revision = authEvents.getRevision();
        const current = () => mounted.current && revision === authEvents.getRevision();
        try {
            const updatedLogs = await updateCalendarLogs(logs => { delete logs[dateStr]; }, revision);
            if (!current()) return;
            logsRequest.current++;
            setLogsLoadFailed(false);
            setOutfitLogs(updatedLogs);
            setTodaysOutfit(updatedLogs[todayStr] ?? null);
            setStreak(calculateStreakFromLogs(updatedLogs));
            setShowDayModal(false);
        } catch (error) {
            console.error('Error deleting outfit:', error);
            if (!current()) return;
            Alert.alert(t('common.error'), t('calendar.persistenceError'));
        }
    };

    const confirmDelete = (dateStr: string) => {
        Alert.alert(t('calendar.deleteOutfit'), t('calendar.deleteOutfitConfirm'), [
            { text: t('common.cancel'), style: 'cancel' },
            { text: t('common.delete'), style: 'destructive', onPress: () => deleteOutfitLog(dateStr) },
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
        setCurrentMonth,
        setCurrentYear,
        selectedDate,
        showDayModal,
        showLogModal,
        showAddPopover,
        outfitLogs,
        logsLoadFailed,
        reloadOutfitLogs: loadOutfitLogs,
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
