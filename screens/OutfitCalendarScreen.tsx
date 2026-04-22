import React, { useEffect, useMemo, useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Image,
    Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

import { useAppNavigation } from '../hooks/useAppNavigation';
import useWardrobeStore from '../store/wardrobeStore';
import {
    useOutfitCalendar,
    formatDate,
    getDaysInMonth,
    OCCASIONS,
} from '../features/calendar/hooks/useOutfitCalendar';
import { OutfitLogForm } from '../features/calendar/components/OutfitLogForm';
import { type OutfitItem, type WardrobeItem, matchesCategory } from '../features/calendar/types';
import { shoppingService, type Product } from '../src/services/shoppingService';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const PAGE_PADDING = 20;
const DAY_GAP = 12;
const DAY_CARD_WIDTH = (SCREEN_WIDTH - PAGE_PADDING * 2 - DAY_GAP * 4) / 5;

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const SLOT_META = [
    { key: 'top', label: 'Top', shortLabel: 'T' },
    { key: 'pants', label: 'Bottom', shortLabel: 'B' },
    { key: 'shoes', label: 'Shoes', shortLabel: 'S' },
] as const;

const formatReadableDate = (date: Date) =>
    date.toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric',
    });

const formatItemName = (item: OutfitItem | null) => item?.name || item?.type || 'Not added yet';

const OutfitCalendarScreen = () => {
    const navigation = useAppNavigation();
    const cal = useOutfitCalendar();

    const storeItems = useWardrobeStore((state) => state.items);
    const fetchItems = useWardrobeStore((state) => state.fetchItems);

    const [shopItems, setShopItems] = useState<Product[]>([]);
    const [selectedDay, setSelectedDay] = useState(() => {
        const now = new Date();
        return now.getDate();
    });

    useEffect(() => {
        let isActive = true;

        shoppingService
            .searchProducts({ query: '', limit: 20 })
            .then((results) => {
                if (isActive) setShopItems(results);
            })
            .catch(() => {});

        fetchItems().catch(() => {});

        return () => {
            isActive = false;
        };
    }, [fetchItems]);

    const daysInMonth = useMemo(
        () => getDaysInMonth(cal.currentYear, cal.currentMonth),
        [cal.currentMonth, cal.currentYear]
    );
    const safeSelectedDay = Math.max(1, Math.min(selectedDay, daysInMonth));

    useEffect(() => {
        setSelectedDay((current) => {
            if (current < 1) return 1;
            if (current > daysInMonth) return daysInMonth;
            return current;
        });
    }, [daysInMonth]);

    const storeWardrobeItems = useMemo<WardrobeItem[]>(
        () =>
            storeItems.map((item) => ({
                id: item.id,
                type: item.subCategory || item.category || '',
                image: item.imageUrl || item.thumbnailUrl || '',
                imageUrl: item.imageUrl || item.thumbnailUrl || '',
                color: item.primaryColor || '',
                name: item.name || item.subCategory || '',
                category: item.category,
            })),
        [storeItems]
    );

    const resolvedWardrobeItems = storeWardrobeItems.length > 0 ? storeWardrobeItems : cal.wardrobeItems;

    const selectedDateKey = formatDate(cal.currentYear, cal.currentMonth, safeSelectedDay);
    const selectedDate = useMemo(
        () => new Date(cal.currentYear, cal.currentMonth, safeSelectedDay),
        [cal.currentMonth, cal.currentYear, safeSelectedDay]
    );
    const selectedLog = cal.outfitLogs[selectedDateKey];
    const selectedOccasion = selectedLog
        ? OCCASIONS.find((occasion) => occasion.id === selectedLog.occasion) ?? null
        : null;

    const selectedSlots = useMemo(() => {
        if (!selectedLog) {
            return {
                top: null,
                pants: null,
                shoes: null,
                extras: [] as OutfitItem[],
            };
        }

        const top = selectedLog.items.find((item) => matchesCategory(item.type, 'top')) ?? null;
        const pants = selectedLog.items.find((item) => matchesCategory(item.type, 'pants')) ?? null;
        const shoes = selectedLog.items.find((item) => matchesCategory(item.type, 'shoes')) ?? null;

        const usedIds = new Set([top?.id, pants?.id, shoes?.id].filter(Boolean));
        const extras = selectedLog.items.filter((item) => !usedIds.has(item.id));

        return { top, pants, shoes, extras };
    }, [selectedLog]);

    const visibleDays = useMemo(() => {
        const count = Math.min(5, daysInMonth);
        let start = safeSelectedDay - 2;

        if (start < 1) start = 1;

        let end = start + count - 1;
        if (end > daysInMonth) {
            end = daysInMonth;
            start = Math.max(1, end - count + 1);
        }

        return Array.from({ length: end - start + 1 }, (_, index) => {
            const day = start + index;
            const dateKey = formatDate(cal.currentYear, cal.currentMonth, day);
            const log = cal.outfitLogs[dateKey];
            const date = new Date(cal.currentYear, cal.currentMonth, day);

            return {
                day,
                dateKey,
                date,
                log,
                isSelected: day === safeSelectedDay,
                isToday: dateKey === cal.todayStr,
            };
        });
    }, [cal.currentMonth, cal.currentYear, cal.outfitLogs, cal.todayStr, daysInMonth, safeSelectedDay]);

    const monthlyStats = cal.getMonthlyStats();
    const isPastDay = selectedDateKey < cal.todayStr;

    const openSelectedDay = () => {
        cal.openLogModalForDate(selectedDateKey);
    };

    const renderMiniSlot = (item: OutfitItem | null | undefined, shortLabel: string, isSelected: boolean) => (
        <View style={[styles.miniSlot, isSelected && styles.miniSlotSelected]}>
            {item?.image ? (
                <Image source={{ uri: item.image }} style={styles.miniSlotImage} resizeMode="cover" />
            ) : (
                <Text style={[styles.miniSlotPlaceholder, isSelected && styles.miniSlotPlaceholderSelected]}>
                    {shortLabel}
                </Text>
            )}
        </View>
    );

    const renderPreviewTile = (
        label: string,
        item: OutfitItem | null,
        variant: 'large' | 'small'
    ) => (
        <View
            style={[
                styles.previewTile,
                variant === 'large' ? styles.previewTileLarge : styles.previewTileSmall,
            ]}
        >
            {item?.image ? (
                <View style={styles.previewMedia}>
                    <Image source={{ uri: item.image }} style={styles.previewImage} resizeMode="contain" />
                </View>
            ) : (
                <View style={styles.previewEmpty}>
                    <Text style={styles.previewEmptyLabel}>{label}</Text>
                    <Text style={styles.previewEmptyHint}>Not added</Text>
                </View>
            )}

            <View style={styles.previewLabelPill}>
                <Text style={styles.previewLabelText}>{label}</Text>
            </View>
        </View>
    );

    const renderDockItem = (label: string, item: OutfitItem | null) => (
        <View style={styles.dockItem}>
            <View style={styles.dockThumbWrap}>
                {item?.image ? (
                    <Image source={{ uri: item.image }} style={styles.dockThumb} resizeMode="contain" />
                ) : (
                    <View style={[styles.dockThumb, styles.dockThumbEmpty]}>
                        <Ionicons name="add" size={18} color="#94A3B8" />
                    </View>
                )}
            </View>
            <Text style={[styles.dockName, !item && styles.dockNameMuted]} numberOfLines={2}>
                {item ? formatItemName(item) : label}
            </Text>
        </View>
    );

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                <View style={styles.topBar}>
                    <TouchableOpacity
                        style={styles.iconButton}
                        onPress={() => navigation.goBack()}
                        accessibilityLabel="Close calendar"
                    >
                        <Ionicons name="close" size={20} color="#0F172A" />
                    </TouchableOpacity>

                    <View style={styles.streakPill}>
                        <Text style={styles.streakEmoji}>🔥</Text>
                        <Text style={styles.streakCount}>{cal.streak}</Text>
                    </View>

                    <View style={styles.iconButtonPlaceholder} />
                </View>

                <ScrollView
                    style={styles.scrollView}
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    <Animated.View entering={FadeIn.duration(220)} style={styles.plannerCard}>
                        <View style={styles.plannerHeader}>
                            <View style={styles.plannerCopy}>
                                <Text style={styles.plannerTitle}>Plan looks</Text>
                            </View>

                            <TouchableOpacity style={styles.openDayButton} onPress={openSelectedDay}>
                                <Ionicons
                                    name={selectedLog ? 'create-outline' : 'add-circle-outline'}
                                    size={18}
                                    color="#0F172A"
                                />
                            </TouchableOpacity>
                        </View>

                        <View style={styles.monthBar}>
                            <TouchableOpacity style={styles.monthArrow} onPress={cal.goToPrevMonth}>
                                <Ionicons name="chevron-back" size={20} color="#0F172A" />
                            </TouchableOpacity>

                            <Text style={styles.monthTitle}>
                                {MONTHS[cal.currentMonth]} {cal.currentYear}
                            </Text>

                            <TouchableOpacity style={styles.monthArrow} onPress={cal.goToNextMonth}>
                                <Ionicons name="chevron-forward" size={20} color="#0F172A" />
                            </TouchableOpacity>
                        </View>

                        <View style={styles.daysRow}>
                            {visibleDays.map((day) => (
                                <TouchableOpacity
                                    key={day.dateKey}
                                    style={[styles.dayCard, day.isSelected && styles.dayCardSelected]}
                                    activeOpacity={0.85}
                                    onPress={() => setSelectedDay(day.day)}
                                >
                                    <Text
                                        style={[
                                            styles.dayWeekday,
                                            day.isSelected && styles.dayTextOnSelected,
                                            day.isToday && !day.isSelected && styles.dayWeekdayToday,
                                        ]}
                                    >
                                        {WEEKDAYS[day.date.getDay()]}
                                    </Text>

                                    <Text
                                        style={[
                                            styles.dayNumber,
                                            day.isSelected && styles.dayTextOnSelected,
                                        ]}
                                    >
                                        {day.day}
                                    </Text>

                                    <View style={styles.miniSlotStack}>
                                        {SLOT_META.map((slot) =>
                                            renderMiniSlot(
                                                day.log?.items.find((item) => matchesCategory(item.type, slot.key)) ?? null,
                                                slot.shortLabel,
                                                day.isSelected
                                            )
                                        )}
                                    </View>

                                    <View
                                        style={[
                                            styles.dayStatusPill,
                                            day.log ? styles.dayStatusLogged : styles.dayStatusOpen,
                                            day.isSelected && styles.dayStatusSelected,
                                        ]}
                                    >
                                        <Ionicons
                                            name={day.log ? 'checkmark' : 'add'}
                                            size={14}
                                            color={day.isSelected ? '#FFFFFF' : day.log ? '#10B981' : '#64748B'}
                                        />
                                    </View>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </Animated.View>

                    <Animated.View entering={FadeInDown.delay(70).duration(260)} style={styles.summaryCard}>
                        <View style={styles.summaryItem}>
                            <Ionicons name="calendar-outline" size={15} color="#94A3B8" />
                            <Text style={styles.summaryValue}>{safeSelectedDay}</Text>
                        </View>
                        <View style={styles.summaryDivider} />
                        <View style={styles.summaryItem}>
                            <Ionicons name="checkmark-done-outline" size={15} color="#94A3B8" />
                            <Text style={styles.summaryValue}>{monthlyStats.logged}</Text>
                        </View>
                        <View style={styles.summaryDivider} />
                        <View style={styles.summaryItem}>
                            <Ionicons name="flame-outline" size={15} color="#94A3B8" />
                            <Text style={styles.summaryValue}>{cal.streak}</Text>
                        </View>
                        <View style={styles.summaryDivider} />
                        <View style={styles.summaryItem}>
                            <Ionicons name="today-outline" size={15} color="#94A3B8" />
                            <Text style={styles.summaryValue}>{MONTHS[cal.currentMonth].toUpperCase()}</Text>
                        </View>
                    </Animated.View>

                    <Animated.View entering={FadeInDown.delay(120).duration(280)} style={styles.outfitCardWrap}>
                        <View style={styles.outfitCard}>
                            <View style={styles.outfitHeader}>
                                <View style={styles.outfitHeaderCopy}>
                                    <Text style={styles.outfitTitle}>{formatReadableDate(selectedDate)}</Text>
                                </View>

                                <View
                                    style={[
                                        styles.statusBadge,
                                        selectedLog
                                            ? { backgroundColor: selectedOccasion?.color ?? '#0F172A' }
                                            : styles.statusBadgeEmpty,
                                    ]}
                                >
                                    <Text
                                        style={[
                                            styles.statusBadgeText,
                                            !selectedLog && styles.statusBadgeTextEmpty,
                                        ]}
                                    >
                                        {selectedLog
                                            ? `${selectedOccasion?.icon ?? '✓'} ${selectedOccasion?.label ?? 'Logged'}`
                                            : isPastDay
                                            ? 'Empty day'
                                            : 'Open day'}
                                    </Text>
                                </View>
                            </View>

                            <View style={styles.boardShell}>
                                <View style={styles.previewRow}>
                                    <View style={styles.previewPrimaryColumn}>
                                        {renderPreviewTile('Top', selectedSlots.top, 'large')}
                                    </View>

                                    <View style={styles.previewSecondaryColumn}>
                                        {renderPreviewTile('Bottom', selectedSlots.pants, 'small')}
                                        {renderPreviewTile('Shoes', selectedSlots.shoes, 'small')}
                                    </View>
                                </View>
                            </View>

                            <View style={styles.pieceDock}>
                                {renderDockItem('Top', selectedSlots.top)}
                                <View style={styles.pieceDockDivider} />
                                {renderDockItem('Bottom', selectedSlots.pants)}
                                <View style={styles.pieceDockDivider} />
                                {renderDockItem('Shoes', selectedSlots.shoes)}
                            </View>

                            {selectedSlots.extras.length > 0 && (
                                <View style={styles.extraSection}>
                                    <Text style={styles.extraTitle}>Extras</Text>
                                    <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                                        {selectedSlots.extras.map((item) => (
                                            <View key={item.id} style={styles.extraCard}>
                                                <Image
                                                    source={{ uri: item.image }}
                                                    style={styles.extraImage}
                                                    resizeMode="cover"
                                                />
                                                <Text style={styles.extraName} numberOfLines={2}>
                                                    {formatItemName(item)}
                                                </Text>
                                            </View>
                                        ))}
                                    </ScrollView>
                                </View>
                            )}

                            <View style={styles.actionRow}>
                                <TouchableOpacity style={styles.primaryAction} onPress={openSelectedDay}>
                                    <Ionicons
                                        name={selectedLog ? 'create-outline' : 'add-circle-outline'}
                                        size={18}
                                        color="#FFFFFF"
                                    />
                                    <Text style={styles.primaryActionText}>
                                        {selectedLog
                                            ? 'Replace outfit'
                                            : isPastDay
                                            ? 'Log outfit'
                                            : 'Plan outfit'}
                                    </Text>
                                </TouchableOpacity>

                                {selectedLog ? (
                                    <TouchableOpacity
                                        style={styles.secondaryAction}
                                        onPress={() => cal.confirmDelete(selectedDateKey)}
                                    >
                                        <Ionicons name="trash-outline" size={18} color="#EF4444" />
                                        <Text style={styles.secondaryActionDangerText}>Delete</Text>
                                    </TouchableOpacity>
                                ) : (
                                    <TouchableOpacity
                                        style={styles.secondaryAction}
                                        onPress={() =>
                                            navigation.navigate('AIOutfit', {
                                                calendarDate: selectedDateKey,
                                            })
                                        }
                                    >
                                        <Ionicons name="sparkles-outline" size={18} color="#0F172A" />
                                        <Text style={styles.secondaryActionText}>Create with AI</Text>
                                    </TouchableOpacity>
                                )}
                            </View>
                        </View>
                    </Animated.View>
                </ScrollView>

                <OutfitLogForm
                    visible={cal.showLogModal}
                    wardrobeItems={resolvedWardrobeItems}
                    shopItems={shopItems}
                    selectedItems={cal.selectedItems}
                    selectedOccasion={cal.selectedOccasion}
                    onClose={cal.closeLogModal}
                    onToggleItem={cal.toggleItemSelection}
                    onSelectOccasion={cal.setSelectedOccasion}
                    onSave={cal.saveOutfit}
                />
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#EEF4FF',
    },
    safeArea: {
        flex: 1,
    },
    topBar: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: PAGE_PADDING,
        paddingTop: 4,
        paddingBottom: 8,
    },
    iconButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderWidth: 1,
        borderColor: 'rgba(15,23,42,0.06)',
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#0F172A',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.08,
        shadowRadius: 16,
        elevation: 3,
    },
    iconButtonPlaceholder: {
        width: 44,
        height: 44,
    },
    streakPill: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 28,
        backgroundColor: 'rgba(255,255,255,0.92)',
        borderWidth: 1,
        borderColor: 'rgba(15,23,42,0.06)',
    },
    streakEmoji: {
        fontSize: 18,
    },
    streakCount: {
        fontSize: 22,
        fontWeight: '900',
        color: '#0F172A',
    },
    streakLabel: {
        fontSize: 12,
        fontWeight: '700',
        color: '#94A3B8',
        textTransform: 'uppercase',
        letterSpacing: 0.6,
    },
    scrollView: {
        flex: 1,
    },
    scrollContent: {
        paddingHorizontal: PAGE_PADDING,
        paddingBottom: 36,
        gap: 16,
    },
    plannerCard: {
        backgroundColor: 'rgba(255,255,255,0.92)',
        borderRadius: 32,
        padding: 18,
        borderWidth: 1,
        borderColor: 'rgba(15,23,42,0.06)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.08,
        shadowRadius: 24,
        elevation: 4,
    },
    plannerHeader: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        gap: 12,
        marginBottom: 16,
    },
    plannerCopy: {
        flex: 1,
        paddingRight: 8,
    },
    plannerTitle: {
        fontSize: 22,
        fontWeight: '900',
        color: '#0F172A',
        lineHeight: 26,
    },
    eyebrow: {
        fontSize: 11,
        fontWeight: '800',
        color: '#94A3B8',
        letterSpacing: 1.4,
        marginBottom: 8,
    },
    title: {
        fontSize: 24,
        fontWeight: '900',
        color: '#0F172A',
        lineHeight: 30,
        marginBottom: 8,
    },
    subtitle: {
        fontSize: 16,
        lineHeight: 22,
        color: '#64748B',
    },
    openDayButton: {
        alignItems: 'center',
        justifyContent: 'center',
        width: 42,
        height: 42,
        borderRadius: 21,
        backgroundColor: '#F8FAFF',
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    openDayText: {
        fontSize: 14,
        fontWeight: '700',
        color: '#0F172A',
    },
    monthBar: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: '#FBFDFF',
        borderWidth: 1,
        borderColor: '#E2E8F0',
        borderRadius: 24,
        paddingHorizontal: 12,
        paddingVertical: 10,
        marginBottom: 16,
    },
    monthArrow: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: '#FFFFFF',
        borderWidth: 1,
        borderColor: '#E2E8F0',
        alignItems: 'center',
        justifyContent: 'center',
    },
    monthTitle: {
        fontSize: 18,
        fontWeight: '800',
        color: '#0F172A',
    },
    daysRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        gap: DAY_GAP,
    },
    dayCard: {
        width: DAY_CARD_WIDTH,
        borderRadius: 24,
        paddingHorizontal: 8,
        paddingTop: 12,
        paddingBottom: 10,
        backgroundColor: '#FFFFFF',
        borderWidth: 1,
        borderColor: '#E6ECF5',
        alignItems: 'center',
    },
    dayCardSelected: {
        backgroundColor: '#254F86',
        borderColor: '#254F86',
    },
    dayWeekday: {
        fontSize: 10,
        fontWeight: '800',
        color: '#94A3B8',
        textTransform: 'uppercase',
        letterSpacing: 0.8,
        marginBottom: 6,
    },
    dayWeekdayToday: {
        color: '#254F86',
    },
    dayNumber: {
        fontSize: 18,
        fontWeight: '900',
        color: '#0F172A',
        marginBottom: 10,
    },
    dayTextOnSelected: {
        color: '#FFFFFF',
    },
    miniSlotStack: {
        width: '100%',
        gap: 6,
        marginBottom: 10,
    },
    miniSlot: {
        height: 26,
        borderRadius: 10,
        backgroundColor: '#F8FAFF',
        borderWidth: 1,
        borderColor: '#E2E8F0',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
    },
    miniSlotSelected: {
        backgroundColor: 'rgba(255,255,255,0.1)',
        borderColor: 'rgba(255,255,255,0.18)',
    },
    miniSlotImage: {
        width: '100%',
        height: '100%',
    },
    miniSlotPlaceholder: {
        fontSize: 10,
        fontWeight: '800',
        color: '#94A3B8',
    },
    miniSlotPlaceholderSelected: {
        color: 'rgba(255,255,255,0.75)',
    },
    dayStatusPill: {
        width: 28,
        height: 28,
        borderRadius: 14,
        alignItems: 'center',
        justifyContent: 'center',
    },
    dayStatusLogged: {
        backgroundColor: '#E9FBF4',
    },
    dayStatusOpen: {
        backgroundColor: '#F8FAFF',
    },
    dayStatusSelected: {
        backgroundColor: 'rgba(255,255,255,0.14)',
    },
    dayStatusText: {
        fontSize: 11,
        fontWeight: '800',
        color: '#10B981',
    },
    dayStatusTextSelected: {
        color: '#FFFFFF',
    },
    summaryCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.92)',
        borderRadius: 28,
        paddingVertical: 18,
        paddingHorizontal: 8,
        borderWidth: 1,
        borderColor: 'rgba(15,23,42,0.06)',
    },
    summaryItem: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
    },
    summaryDivider: {
        width: 1,
        height: 30,
        backgroundColor: '#E2E8F0',
    },
    summaryValue: {
        fontSize: 18,
        fontWeight: '900',
        color: '#0F172A',
    },
    summaryLabel: {
        fontSize: 10,
        fontWeight: '800',
        color: '#94A3B8',
        letterSpacing: 1,
    },
    outfitCardWrap: {
        marginBottom: 8,
    },
    outfitCard: {
        backgroundColor: 'rgba(255,255,255,0.94)',
        borderRadius: 32,
        padding: 18,
        borderWidth: 1,
        borderColor: 'rgba(15,23,42,0.06)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.08,
        shadowRadius: 24,
        elevation: 4,
    },
    outfitHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        marginBottom: 10,
    },
    outfitHeaderCopy: {
        flex: 1,
        paddingRight: 8,
    },
    outfitTitle: {
        fontSize: 20,
        fontWeight: '900',
        color: '#0F172A',
    },
    outfitSubtitle: {
        fontSize: 14,
        lineHeight: 20,
        color: '#64748B',
        marginBottom: 12,
    },
    statusBadge: {
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 18,
        maxWidth: 140,
    },
    statusBadgeEmpty: {
        backgroundColor: '#E2E8F0',
    },
    statusBadgeText: {
        fontSize: 12,
        fontWeight: '800',
        color: '#FFFFFF',
        textAlign: 'center',
    },
    statusBadgeTextEmpty: {
        color: '#475569',
    },
    boardShell: {
        backgroundColor: '#F8FAFF',
        borderRadius: 28,
        borderWidth: 1,
        borderColor: '#EEF2F7',
        padding: 12,
        marginBottom: 14,
    },
    previewRow: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        gap: 12,
    },
    previewPrimaryColumn: {
        flex: 1.08,
    },
    previewSecondaryColumn: {
        flex: 0.92,
        gap: 12,
    },
    previewTile: {
        borderRadius: 24,
        overflow: 'hidden',
        backgroundColor: '#FFFFFF',
        borderWidth: 1,
        borderColor: '#EDF2F7',
        position: 'relative',
    },
    previewTileLarge: {
        height: 248,
    },
    previewTileSmall: {
        height: 118,
    },
    previewMedia: {
        flex: 1,
        padding: 18,
        alignItems: 'center',
        justifyContent: 'center',
    },
    previewImage: {
        width: '100%',
        height: '100%',
    },
    previewEmpty: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 16,
        gap: 4,
    },
    previewEmptyLabel: {
        fontSize: 16,
        fontWeight: '800',
        color: '#475569',
    },
    previewEmptyHint: {
        fontSize: 12,
        fontWeight: '600',
        color: '#94A3B8',
    },
    previewLabelPill: {
        position: 'absolute',
        left: 12,
        top: 12,
        backgroundColor: 'rgba(255,255,255,0.94)',
        paddingHorizontal: 10,
        paddingVertical: 5,
        borderRadius: 14,
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    previewLabelText: {
        fontSize: 11,
        fontWeight: '800',
        color: '#334155',
    },
    pieceDock: {
        flexDirection: 'row',
        alignItems: 'stretch',
        backgroundColor: '#FBFDFF',
        borderRadius: 24,
        borderWidth: 1,
        borderColor: '#E2E8F0',
        paddingHorizontal: 12,
        paddingVertical: 12,
        marginBottom: 12,
    },
    pieceDockDivider: {
        width: 1,
        backgroundColor: '#E2E8F0',
        marginHorizontal: 8,
    },
    dockItem: {
        flex: 1,
        alignItems: 'center',
    },
    dockThumbWrap: {
        width: '100%',
        alignItems: 'center',
        marginBottom: 10,
    },
    dockThumb: {
        width: 64,
        height: 64,
        borderRadius: 18,
        backgroundColor: '#F1F5F9',
    },
    dockThumbEmpty: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    dockLabel: {
        fontSize: 11,
        fontWeight: '800',
        color: '#94A3B8',
        letterSpacing: 1,
        marginBottom: 6,
        textTransform: 'uppercase',
    },
    dockName: {
        fontSize: 11,
        fontWeight: '700',
        color: '#0F172A',
        lineHeight: 15,
        textAlign: 'center',
    },
    dockNameMuted: {
        color: '#94A3B8',
    },
    extraSection: {
        marginBottom: 12,
    },
    extraTitle: {
        fontSize: 14,
        fontWeight: '800',
        color: '#0F172A',
        marginBottom: 10,
    },
    extraCard: {
        width: 108,
        marginRight: 10,
        backgroundColor: '#FBFDFF',
        borderRadius: 20,
        padding: 8,
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    extraImage: {
        width: '100%',
        height: 92,
        borderRadius: 14,
        backgroundColor: '#E2E8F0',
        marginBottom: 8,
    },
    extraName: {
        fontSize: 12,
        fontWeight: '700',
        color: '#0F172A',
        lineHeight: 16,
    },
    actionRow: {
        flexDirection: 'row',
        gap: 10,
        marginTop: 8,
    },
    primaryAction: {
        flex: 1,
        minHeight: 52,
        borderRadius: 18,
        backgroundColor: '#0F172A',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: 8,
        paddingHorizontal: 16,
    },
    primaryActionText: {
        fontSize: 15,
        fontWeight: '800',
        color: '#FFFFFF',
    },
    secondaryAction: {
        minHeight: 52,
        borderRadius: 18,
        backgroundColor: '#FFFFFF',
        borderWidth: 1,
        borderColor: '#E2E8F0',
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: 8,
        paddingHorizontal: 16,
    },
    secondaryActionText: {
        fontSize: 15,
        fontWeight: '800',
        color: '#0F172A',
    },
    secondaryActionDangerText: {
        fontSize: 15,
        fontWeight: '800',
        color: '#EF4444',
    },
});

export default OutfitCalendarScreen;
