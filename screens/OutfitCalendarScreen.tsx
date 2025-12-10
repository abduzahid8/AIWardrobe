import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Image,
    Modal,
    Dimensions,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
    FadeIn,
    FadeInDown,
    FadeInUp,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { LinearGradient } from 'expo-linear-gradient';

import { colors, spacing, shadows, borderRadius } from '../src/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const DAY_SIZE = (SCREEN_WIDTH - spacing.l * 2 - spacing.xs * 12) / 7;

// Get days in month
const getDaysInMonth = (year: number, month: number) => {
    return new Date(year, month + 1, 0).getDate();
};

// Get first day of month (0 = Sunday)
const getFirstDayOfMonth = (year: number, month: number) => {
    return new Date(year, month, 1).getDay();
};

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WEEKDAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

// Occasions for outfit logging
const OCCASIONS = [
    { id: 'work', label: 'Work', icon: '💼', color: '#3B82F6' },
    { id: 'casual', label: 'Casual', icon: '☕', color: '#22C55E' },
    { id: 'date', label: 'Date', icon: '💕', color: '#EC4899' },
    { id: 'party', label: 'Party', icon: '🎉', color: '#F59E0B' },
    { id: 'sport', label: 'Sport', icon: '🏃', color: '#8B5CF6' },
    { id: 'formal', label: 'Formal', icon: '🎩', color: '#1A1A1A' },
];

interface OutfitLog {
    date: string;
    items: Array<{ id: string; type: string; image: string; color?: string }>;
    occasion: string;
    note?: string;
    rating?: number;
}

const OutfitCalendarScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();

    const today = new Date();
    const [currentMonth, setCurrentMonth] = useState(today.getMonth());
    const [currentYear, setCurrentYear] = useState(today.getFullYear());
    const [selectedDate, setSelectedDate] = useState<string | null>(null);
    const [showDayModal, setShowDayModal] = useState(false);
    const [outfitLogs, setOutfitLogs] = useState<Record<string, OutfitLog>>({});
    const [todaysOutfit, setTodaysOutfit] = useState<OutfitLog | null>(null);
    const [streak, setStreak] = useState(0);
    const [wardrobeItems, setWardrobeItems] = useState<any[]>([]);
    const [showLogModal, setShowLogModal] = useState(false);
    const [selectedItems, setSelectedItems] = useState<any[]>([]);
    const [selectedOccasion, setSelectedOccasion] = useState<string>('casual');
    const [viewMode, setViewMode] = useState<'month' | 'week'>('month');

    // Format date as YYYY-MM-DD
    const formatDate = (year: number, month: number, day: number) => {
        return `${year}-${String(month + 1).padStart(2, '0')}-${String(day).padStart(2, '0')}`;
    };

    const todayStr = formatDate(today.getFullYear(), today.getMonth(), today.getDate());

    // Load outfit logs
    const loadOutfitLogs = useCallback(async () => {
        try {
            const data = await AsyncStorage.getItem('outfitLogs');
            if (data) {
                const logs = JSON.parse(data);
                setOutfitLogs(logs);
                if (logs[todayStr]) {
                    setTodaysOutfit(logs[todayStr]);
                }
                // Calculate streak
                let streakCount = 0;
                let checkDate = new Date(today);
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

    // Load wardrobe items
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

    // Navigate months
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

    // Handle day press
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

    // Toggle item selection
    const toggleItemSelection = (item: any) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        if (selectedItems.find(i => i.id === item.id)) {
            setSelectedItems(selectedItems.filter(i => i.id !== item.id));
        } else if (selectedItems.length < 6) {
            setSelectedItems([...selectedItems, item]);
        }
    };

    // Save outfit
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

    // Delete outfit log
    const deleteOutfitLog = async (dateStr: string) => {
        const updatedLogs = { ...outfitLogs };
        delete updatedLogs[dateStr];
        try {
            await AsyncStorage.setItem('outfitLogs', JSON.stringify(updatedLogs));
            setOutfitLogs(updatedLogs);
            if (dateStr === todayStr) {
                setTodaysOutfit(null);
            }
            setShowDayModal(false);
        } catch (error) {
            console.error('Error deleting outfit:', error);
        }
    };

    // Get occasion color
    const getOccasionColor = (occasionId: string) => {
        return OCCASIONS.find(o => o.id === occasionId)?.color || '#6B7280';
    };

    // Calculate monthly stats
    const getMonthlyStats = () => {
        const monthKey = `${currentYear}-${String(currentMonth + 1).padStart(2, '0')}`;
        const monthLogs = Object.entries(outfitLogs)
            .filter(([date]) => date.startsWith(monthKey));
        return {
            logged: monthLogs.length,
            total: getDaysInMonth(currentYear, currentMonth),
        };
    };

    // Render calendar days
    const renderCalendarDays = () => {
        const daysInMonth = getDaysInMonth(currentYear, currentMonth);
        const firstDay = getFirstDayOfMonth(currentYear, currentMonth);
        const days = [];

        // Empty cells
        for (let i = 0; i < firstDay; i++) {
            days.push(<View key={`empty-${i}`} style={styles.dayCell} />);
        }

        // Days
        for (let day = 1; day <= daysInMonth; day++) {
            const dateStr = formatDate(currentYear, currentMonth, day);
            const log = outfitLogs[dateStr];
            const isToday = dateStr === todayStr;
            const isFuture = new Date(dateStr) > today;

            days.push(
                <TouchableOpacity
                    key={day}
                    style={[
                        styles.dayCell,
                        isToday && styles.todayCell,
                    ]}
                    onPress={() => handleDayPress(day)}
                    activeOpacity={0.7}
                >
                    <Text style={[
                        styles.dayText,
                        isToday && styles.todayText,
                        isFuture && styles.futureDayText,
                    ]}>
                        {day}
                    </Text>
                    {log && (
                        <View style={[styles.outfitDot, { backgroundColor: getOccasionColor(log.occasion) }]} />
                    )}
                </TouchableOpacity>
            );
        }

        return days;
    };

    const monthlyStats = getMonthlyStats();

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                <ScrollView showsVerticalScrollIndicator={false}>
                    {/* Header */}
                    <Animated.View entering={FadeIn} style={styles.header}>
                        <View>
                            <Text style={styles.title}>Outfit Planner</Text>
                            <Text style={styles.subtitle}>Track & plan your daily looks</Text>
                        </View>
                        {streak > 0 && (
                            <View style={styles.streakBadge}>
                                <Text style={styles.streakEmoji}>🔥</Text>
                                <Text style={styles.streakNumber}>{streak}</Text>
                            </View>
                        )}
                    </Animated.View>

                    {/* Today's Section */}
                    <Animated.View entering={FadeInDown.delay(100)}>
                        <TouchableOpacity
                            style={styles.todayCard}
                            onPress={() => {
                                if (!todaysOutfit) {
                                    setSelectedDate(todayStr);
                                    setShowLogModal(true);
                                }
                            }}
                            activeOpacity={todaysOutfit ? 1 : 0.8}
                        >
                            <LinearGradient
                                colors={todaysOutfit ? ['#22C55E', '#16A34A'] : ['#1A1A1A', '#2D2D2D']}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 1 }}
                                style={styles.todayGradient}
                            >
                                <View style={styles.todayContent}>
                                    <View>
                                        <Text style={styles.todayLabel}>TODAY</Text>
                                        <Text style={styles.todayTitle}>
                                            {todaysOutfit ? 'Outfit Logged ✓' : 'Log Your Outfit'}
                                        </Text>
                                        <Text style={styles.todayDate}>
                                            {today.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
                                        </Text>
                                    </View>
                                    {todaysOutfit ? (
                                        <View style={styles.todayItems}>
                                            {todaysOutfit.items.slice(0, 3).map((item, idx) => (
                                                <View key={idx} style={styles.todayItemThumb}>
                                                    <Image source={{ uri: item.image }} style={styles.todayItemImage} />
                                                </View>
                                            ))}
                                            {todaysOutfit.items.length > 3 && (
                                                <View style={styles.todayItemMore}>
                                                    <Text style={styles.todayItemMoreText}>+{todaysOutfit.items.length - 3}</Text>
                                                </View>
                                            )}
                                        </View>
                                    ) : (
                                        <View style={styles.todayAddIcon}>
                                            <Ionicons name="add" size={28} color="#FFF" />
                                        </View>
                                    )}
                                </View>
                            </LinearGradient>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* Quick Actions */}
                    <Animated.View entering={FadeInDown.delay(150)} style={styles.quickActions}>
                        <TouchableOpacity
                            style={styles.quickAction}
                            onPress={() => (navigation as any).navigate('WardrobeVideo')}
                        >
                            <View style={[styles.quickActionIcon, { backgroundColor: '#EFF6FF' }]}>
                                <Ionicons name="videocam" size={20} color="#3B82F6" />
                            </View>
                            <Text style={styles.quickActionText}>Scan Wardrobe</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.quickAction}
                            onPress={() => (navigation as any).navigate('AIChat')}
                        >
                            <View style={[styles.quickActionIcon, { backgroundColor: '#FFF1F2' }]}>
                                <Ionicons name="sparkles" size={20} color="#EC4899" />
                            </View>
                            <Text style={styles.quickActionText}>Get AI Outfit</Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={styles.quickAction}
                            onPress={() => (navigation as any).navigate('Stats')}
                        >
                            <View style={[styles.quickActionIcon, { backgroundColor: '#F0FDF4' }]}>
                                <Ionicons name="stats-chart" size={20} color="#22C55E" />
                            </View>
                            <Text style={styles.quickActionText}>Wardrobe Stats</Text>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* Calendar Card */}
                    <Animated.View entering={FadeInUp.delay(200)} style={styles.calendarCard}>
                        {/* Month Header */}
                        <View style={styles.monthHeader}>
                            <TouchableOpacity onPress={goToPrevMonth} style={styles.navButton}>
                                <Ionicons name="chevron-back" size={22} color={colors.text.primary} />
                            </TouchableOpacity>
                            <View style={styles.monthInfo}>
                                <Text style={styles.monthTitle}>
                                    {MONTHS[currentMonth]} {currentYear}
                                </Text>
                                <Text style={styles.monthStats}>
                                    {monthlyStats.logged}/{monthlyStats.total} days logged
                                </Text>
                            </View>
                            <TouchableOpacity onPress={goToNextMonth} style={styles.navButton}>
                                <Ionicons name="chevron-forward" size={22} color={colors.text.primary} />
                            </TouchableOpacity>
                        </View>

                        {/* Weekday Headers */}
                        <View style={styles.weekdayRow}>
                            {WEEKDAYS.map((day, idx) => (
                                <View key={idx} style={styles.weekdayCell}>
                                    <Text style={styles.weekdayText}>{day}</Text>
                                </View>
                            ))}
                        </View>

                        {/* Calendar Grid */}
                        <View style={styles.calendarGrid}>
                            {renderCalendarDays()}
                        </View>

                        {/* Occasion Legend */}
                        <View style={styles.occasionLegend}>
                            {OCCASIONS.slice(0, 4).map(occ => (
                                <View key={occ.id} style={styles.legendItem}>
                                    <View style={[styles.legendDot, { backgroundColor: occ.color }]} />
                                    <Text style={styles.legendText}>{occ.label}</Text>
                                </View>
                            ))}
                        </View>
                    </Animated.View>

                    {/* Recent Outfits */}
                    <Animated.View entering={FadeInUp.delay(300)} style={styles.recentSection}>
                        <Text style={styles.sectionTitle}>Recent Outfits</Text>
                        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                            {Object.entries(outfitLogs)
                                .sort((a, b) => b[0].localeCompare(a[0]))
                                .slice(0, 5)
                                .map(([date, log]) => (
                                    <TouchableOpacity
                                        key={date}
                                        style={styles.recentCard}
                                        onPress={() => {
                                            setSelectedDate(date);
                                            setShowDayModal(true);
                                        }}
                                    >
                                        <View style={styles.recentImages}>
                                            {log.items.slice(0, 2).map((item, idx) => (
                                                <Image
                                                    key={idx}
                                                    source={{ uri: item.image }}
                                                    style={[styles.recentImage, idx === 1 && styles.recentImageOverlap]}
                                                />
                                            ))}
                                        </View>
                                        <Text style={styles.recentDate}>
                                            {new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                                        </Text>
                                        <View style={[styles.recentOccasion, { backgroundColor: getOccasionColor(log.occasion) }]}>
                                            <Text style={styles.recentOccasionText}>
                                                {OCCASIONS.find(o => o.id === log.occasion)?.icon}
                                            </Text>
                                        </View>
                                    </TouchableOpacity>
                                ))}
                            {Object.keys(outfitLogs).length === 0 && (
                                <View style={styles.emptyRecent}>
                                    <Text style={styles.emptyText}>No outfits logged yet</Text>
                                </View>
                            )}
                        </ScrollView>
                    </Animated.View>

                    <View style={{ height: spacing.xxl }} />
                </ScrollView>

                {/* Log Outfit Modal */}
                <Modal
                    visible={showLogModal}
                    animationType="slide"
                    transparent={true}
                    onRequestClose={() => setShowLogModal(false)}
                >
                    <View style={styles.modalOverlay}>
                        <View style={styles.logModal}>
                            <View style={styles.modalHandle} />

                            <View style={styles.modalHeader}>
                                <Text style={styles.modalTitle}>Log Outfit</Text>
                                <TouchableOpacity onPress={() => setShowLogModal(false)}>
                                    <Ionicons name="close" size={24} color={colors.text.primary} />
                                </TouchableOpacity>
                            </View>

                            {/* Occasion Selection */}
                            <Text style={styles.modalLabel}>Occasion</Text>
                            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.occasionScroll}>
                                {OCCASIONS.map(occ => (
                                    <TouchableOpacity
                                        key={occ.id}
                                        style={[
                                            styles.occasionChip,
                                            selectedOccasion === occ.id && { backgroundColor: occ.color },
                                        ]}
                                        onPress={() => {
                                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                            setSelectedOccasion(occ.id);
                                        }}
                                    >
                                        <Text style={styles.occasionEmoji}>{occ.icon}</Text>
                                        <Text style={[
                                            styles.occasionLabel,
                                            selectedOccasion === occ.id && { color: '#FFF' },
                                        ]}>
                                            {occ.label}
                                        </Text>
                                    </TouchableOpacity>
                                ))}
                            </ScrollView>

                            {/* Selected Items */}
                            {selectedItems.length > 0 && (
                                <View style={styles.selectedRow}>
                                    {selectedItems.map((item, idx) => (
                                        <View key={idx} style={styles.selectedItem}>
                                            <Image source={{ uri: item.image || item.imageUrl }} style={styles.selectedItemImage} />
                                            <TouchableOpacity style={styles.removeBtn} onPress={() => toggleItemSelection(item)}>
                                                <Ionicons name="close-circle" size={18} color="#EF4444" />
                                            </TouchableOpacity>
                                        </View>
                                    ))}
                                </View>
                            )}

                            {/* Wardrobe Items */}
                            <Text style={styles.modalLabel}>Select Items ({selectedItems.length}/6)</Text>
                            <ScrollView style={styles.wardrobeScroll}>
                                <View style={styles.wardrobeGrid}>
                                    {wardrobeItems.map((item, idx) => {
                                        const isSelected = selectedItems.find(i => i.id === item.id);
                                        return (
                                            <TouchableOpacity
                                                key={item.id || idx}
                                                style={[styles.wardrobeItem, isSelected && styles.wardrobeItemSelected]}
                                                onPress={() => toggleItemSelection(item)}
                                            >
                                                <Image
                                                    source={{ uri: item.image || item.imageUrl }}
                                                    style={styles.wardrobeItemImage}
                                                />
                                                {isSelected && (
                                                    <View style={styles.selectedCheck}>
                                                        <Ionicons name="checkmark" size={16} color="#FFF" />
                                                    </View>
                                                )}
                                            </TouchableOpacity>
                                        );
                                    })}
                                </View>
                            </ScrollView>

                            {/* Save Button */}
                            <TouchableOpacity
                                style={[styles.saveButton, selectedItems.length === 0 && styles.saveButtonDisabled]}
                                onPress={saveOutfit}
                                disabled={selectedItems.length === 0}
                            >
                                <Text style={styles.saveButtonText}>
                                    Save Outfit ({selectedItems.length} items)
                                </Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </Modal>

                {/* Day Detail Modal */}
                <Modal
                    visible={showDayModal}
                    animationType="fade"
                    transparent={true}
                    onRequestClose={() => setShowDayModal(false)}
                >
                    <TouchableOpacity
                        style={styles.modalOverlay}
                        activeOpacity={1}
                        onPress={() => setShowDayModal(false)}
                    >
                        <View style={styles.dayModal}>
                            {selectedDate && outfitLogs[selectedDate] && (
                                <>
                                    <View style={styles.dayModalHeader}>
                                        <Text style={styles.dayModalTitle}>
                                            {new Date(selectedDate).toLocaleDateString('en-US', {
                                                weekday: 'long',
                                                month: 'long',
                                                day: 'numeric',
                                            })}
                                        </Text>
                                        <View style={[styles.occasionTag, { backgroundColor: getOccasionColor(outfitLogs[selectedDate].occasion) }]}>
                                            <Text style={styles.occasionTagText}>
                                                {OCCASIONS.find(o => o.id === outfitLogs[selectedDate].occasion)?.icon}{' '}
                                                {OCCASIONS.find(o => o.id === outfitLogs[selectedDate].occasion)?.label}
                                            </Text>
                                        </View>
                                    </View>

                                    <View style={styles.dayModalItems}>
                                        {outfitLogs[selectedDate].items.map((item, idx) => (
                                            <View key={idx} style={styles.dayModalItem}>
                                                <Image source={{ uri: item.image }} style={styles.dayModalItemImage} />
                                                <Text style={styles.dayModalItemType}>{item.type}</Text>
                                            </View>
                                        ))}
                                    </View>

                                    <TouchableOpacity
                                        style={styles.deleteButton}
                                        onPress={() => {
                                            Alert.alert('Delete Outfit', 'Are you sure?', [
                                                { text: 'Cancel', style: 'cancel' },
                                                { text: 'Delete', style: 'destructive', onPress: () => deleteOutfitLog(selectedDate) },
                                            ]);
                                        }}
                                    >
                                        <Ionicons name="trash-outline" size={18} color="#EF4444" />
                                        <Text style={styles.deleteButtonText}>Delete</Text>
                                    </TouchableOpacity>
                                </>
                            )}
                        </View>
                    </TouchableOpacity>
                </Modal>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background,
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: spacing.l,
        paddingVertical: spacing.m,
    },
    title: {
        fontSize: 28,
        fontWeight: '800',
        color: colors.text.primary,
    },
    subtitle: {
        fontSize: 14,
        color: colors.text.secondary,
        marginTop: 2,
    },
    streakBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FFF7ED',
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
        borderRadius: borderRadius.full,
        gap: 4,
    },
    streakEmoji: {
        fontSize: 18,
    },
    streakNumber: {
        fontSize: 18,
        fontWeight: '800',
        color: '#EA580C',
    },
    // Today Card
    todayCard: {
        marginHorizontal: spacing.l,
        marginBottom: spacing.m,
        borderRadius: borderRadius.xl,
        overflow: 'hidden',
    },
    todayGradient: {
        padding: spacing.m,
    },
    todayContent: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    todayLabel: {
        fontSize: 11,
        fontWeight: '600',
        color: 'rgba(255,255,255,0.7)',
        letterSpacing: 1,
        marginBottom: 4,
    },
    todayTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: '#FFF',
        marginBottom: 4,
    },
    todayDate: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.8)',
    },
    todayItems: {
        flexDirection: 'row',
    },
    todayItemThumb: {
        width: 40,
        height: 40,
        borderRadius: 8,
        overflow: 'hidden',
        marginLeft: -8,
        borderWidth: 2,
        borderColor: '#22C55E',
    },
    todayItemImage: {
        width: '100%',
        height: '100%',
    },
    todayItemMore: {
        width: 40,
        height: 40,
        borderRadius: 8,
        backgroundColor: 'rgba(255,255,255,0.3)',
        justifyContent: 'center',
        alignItems: 'center',
        marginLeft: -8,
    },
    todayItemMoreText: {
        fontSize: 12,
        fontWeight: '600',
        color: '#FFF',
    },
    todayAddIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: 'rgba(255,255,255,0.2)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    // Quick Actions
    quickActions: {
        flexDirection: 'row',
        paddingHorizontal: spacing.l,
        gap: spacing.s,
        marginBottom: spacing.m,
    },
    quickAction: {
        flex: 1,
        backgroundColor: colors.surface,
        borderRadius: borderRadius.l,
        padding: spacing.s,
        alignItems: 'center',
        ...shadows.soft,
    },
    quickActionIcon: {
        width: 40,
        height: 40,
        borderRadius: 20,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: spacing.xs,
    },
    quickActionText: {
        fontSize: 11,
        fontWeight: '600',
        color: colors.text.primary,
        textAlign: 'center',
    },
    // Calendar
    calendarCard: {
        marginHorizontal: spacing.l,
        backgroundColor: colors.surface,
        borderRadius: borderRadius.xl,
        padding: spacing.m,
        ...shadows.soft,
    },
    monthHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: spacing.m,
    },
    navButton: {
        padding: spacing.xs,
    },
    monthInfo: {
        alignItems: 'center',
    },
    monthTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: colors.text.primary,
    },
    monthStats: {
        fontSize: 12,
        color: colors.text.secondary,
        marginTop: 2,
    },
    weekdayRow: {
        flexDirection: 'row',
        marginBottom: spacing.xs,
    },
    weekdayCell: {
        width: DAY_SIZE,
        alignItems: 'center',
        marginHorizontal: spacing.xs,
    },
    weekdayText: {
        fontSize: 12,
        fontWeight: '600',
        color: colors.text.secondary,
    },
    calendarGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
    },
    dayCell: {
        width: DAY_SIZE,
        height: DAY_SIZE + 4,
        alignItems: 'center',
        justifyContent: 'center',
        marginHorizontal: spacing.xs,
        marginBottom: spacing.xs,
    },
    todayCell: {
        backgroundColor: colors.button.primary,
        borderRadius: DAY_SIZE / 2,
    },
    dayText: {
        fontSize: 14,
        fontWeight: '500',
        color: colors.text.primary,
    },
    todayText: {
        fontWeight: '700',
        color: '#FFF',
    },
    futureDayText: {
        color: colors.text.muted,
    },
    outfitDot: {
        width: 5,
        height: 5,
        borderRadius: 2.5,
        marginTop: 2,
    },
    occasionLegend: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: spacing.m,
        marginTop: spacing.m,
        paddingTop: spacing.m,
        borderTopWidth: 1,
        borderTopColor: colors.border,
    },
    legendItem: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    legendDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
    },
    legendText: {
        fontSize: 11,
        color: colors.text.secondary,
    },
    // Recent
    recentSection: {
        marginTop: spacing.m,
        paddingLeft: spacing.l,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: colors.text.primary,
        marginBottom: spacing.m,
    },
    recentCard: {
        width: 100,
        marginRight: spacing.m,
        alignItems: 'center',
    },
    recentImages: {
        flexDirection: 'row',
        marginBottom: spacing.xs,
    },
    recentImage: {
        width: 44,
        height: 44,
        borderRadius: 8,
        backgroundColor: colors.surfaceHighlight,
    },
    recentImageOverlap: {
        marginLeft: -12,
    },
    recentDate: {
        fontSize: 12,
        fontWeight: '600',
        color: colors.text.primary,
    },
    recentOccasion: {
        position: 'absolute',
        top: 0,
        right: 20,
        width: 20,
        height: 20,
        borderRadius: 10,
        justifyContent: 'center',
        alignItems: 'center',
    },
    recentOccasionText: {
        fontSize: 10,
    },
    emptyRecent: {
        paddingVertical: spacing.xl,
        paddingHorizontal: spacing.l,
    },
    emptyText: {
        fontSize: 14,
        color: colors.text.secondary,
    },
    // Modal styles
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'flex-end',
    },
    logModal: {
        backgroundColor: colors.background,
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        padding: spacing.l,
        maxHeight: '85%',
    },
    modalHandle: {
        width: 40,
        height: 4,
        backgroundColor: colors.border,
        borderRadius: 2,
        alignSelf: 'center',
        marginBottom: spacing.m,
    },
    modalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: spacing.m,
    },
    modalTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
    },
    modalLabel: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.secondary,
        marginBottom: spacing.s,
        marginTop: spacing.m,
    },
    occasionScroll: {
        marginBottom: spacing.m,
    },
    occasionChip: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.surfaceHighlight,
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
        borderRadius: borderRadius.full,
        marginRight: spacing.s,
        gap: 6,
    },
    occasionEmoji: {
        fontSize: 16,
    },
    occasionLabel: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.primary,
    },
    selectedRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.s,
        marginBottom: spacing.m,
    },
    selectedItem: {
        position: 'relative',
    },
    selectedItemImage: {
        width: 56,
        height: 56,
        borderRadius: 12,
        backgroundColor: colors.surfaceHighlight,
    },
    removeBtn: {
        position: 'absolute',
        top: -6,
        right: -6,
    },
    wardrobeScroll: {
        maxHeight: 240,
    },
    wardrobeGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.s,
    },
    wardrobeItem: {
        width: (SCREEN_WIDTH - spacing.l * 2 - spacing.s * 3) / 4,
        aspectRatio: 1,
        backgroundColor: colors.surfaceHighlight,
        borderRadius: 12,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    wardrobeItemSelected: {
        borderColor: colors.text.accent,
    },
    wardrobeItemImage: {
        width: '100%',
        height: '100%',
    },
    selectedCheck: {
        position: 'absolute',
        top: 4,
        right: 4,
        width: 22,
        height: 22,
        borderRadius: 11,
        backgroundColor: colors.text.accent,
        justifyContent: 'center',
        alignItems: 'center',
    },
    saveButton: {
        backgroundColor: colors.button.primary,
        paddingVertical: spacing.m,
        borderRadius: borderRadius.m,
        alignItems: 'center',
        marginTop: spacing.m,
    },
    saveButtonDisabled: {
        backgroundColor: colors.border,
    },
    saveButtonText: {
        color: '#FFF',
        fontSize: 16,
        fontWeight: '700',
    },
    // Day Modal
    dayModal: {
        backgroundColor: colors.surface,
        marginHorizontal: spacing.l,
        borderRadius: borderRadius.xl,
        padding: spacing.l,
    },
    dayModalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: spacing.m,
    },
    dayModalTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: colors.text.primary,
    },
    occasionTag: {
        paddingHorizontal: spacing.s,
        paddingVertical: 4,
        borderRadius: 12,
    },
    occasionTagText: {
        fontSize: 12,
        fontWeight: '600',
        color: '#FFF',
    },
    dayModalItems: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: spacing.s,
        marginBottom: spacing.m,
    },
    dayModalItem: {
        alignItems: 'center',
    },
    dayModalItemImage: {
        width: 64,
        height: 64,
        borderRadius: 12,
        backgroundColor: colors.surfaceHighlight,
        marginBottom: 4,
    },
    dayModalItemType: {
        fontSize: 11,
        color: colors.text.secondary,
    },
    deleteButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 4,
        paddingVertical: spacing.s,
    },
    deleteButtonText: {
        fontSize: 14,
        color: '#EF4444',
        fontWeight: '600',
    },
});

export default OutfitCalendarScreen;
