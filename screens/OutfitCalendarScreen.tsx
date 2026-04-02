/**
 * OutfitCalendarScreen — 2050 Liquid Glass Design
 */

import React, { useState, useMemo, useEffect } from 'react';
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
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInDown } from 'react-native-reanimated';

import { useOutfitCalendar, formatDate, getDaysInMonth } from '../features/calendar/hooks/useOutfitCalendar';
import { OutfitLogForm } from '../features/calendar/components/OutfitLogForm';
import { type WardrobeItem, matchesCategory } from '../features/calendar/types';
import { shoppingService, type Product } from '../src/services/shoppingService';
import useWardrobeStore from '../store/wardrobeStore';
import { INSPO_SHOP_ITEMS } from '../data/inspoShopItems';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Mock wardrobe data for demo
const MOCK_WARDROBE: WardrobeItem[] = [
    { id: '1', type: 'jacket', name: 'Adaptive Exo-Skin Ensemble', image: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=400' },
    { id: '2', type: 't-shirt', name: 'Bio-Pulse Tee', image: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400' },
    { id: '3', type: 'pants', name: 'Kinetic-Warp Cargo Trousers', image: 'https://images.unsplash.com/photo-1552374196-1ab2a1c593e8?w=400' },
    { id: '4', type: 'shoes', name: 'Plasma-Step Runners', image: 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400' },
];

const weekdays = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

const OutfitCalendarScreen = () => {
    const navigation = useNavigation();
    const cal = useOutfitCalendar();
    const today = new Date();
    const [shopItems, setShopItems] = useState<Product[]>([]);
    const storeItems = useWardrobeStore(s => s.items);
    const fetchItems = useWardrobeStore(s => s.fetchItems);

    // Map wardrobe store items → WardrobeItem format for the picker
    const storeWardrobeItems: WardrobeItem[] = storeItems.map(item => ({
        id: item.id,
        type: item.subCategory || item.category || '',
        image: item.imageUrl || item.thumbnailUrl || '',
        imageUrl: item.imageUrl || '',
        color: item.primaryColor || '',
        name: item.name || item.subCategory || '',
        category: item.category,
    }));

    // Best available wardrobe items: store → calendar hook → mock
    const resolvedWardrobeItems: WardrobeItem[] =
        storeWardrobeItems.length > 0 ? storeWardrobeItems
        : cal.wardrobeItems.length > 0 ? cal.wardrobeItems
        : MOCK_WARDROBE;

    useEffect(() => {
        shoppingService.searchProducts({ query: '', limit: 20 }).then(setShopItems).catch(() => {});
        fetchItems().catch(() => {});
    }, []);
    const [selectedDay, setSelectedDay] = useState(
        today.getMonth() === cal.currentMonth && today.getFullYear() === cal.currentYear
            ? today.getDate()
            : 1
    );

    const currentMonth = cal.currentMonth;
    const currentYear = cal.currentYear;

    // Generate days for horizontal strip
    const daysStrip = useMemo(() => {
        const days = [];
        const daysInMonth = getDaysInMonth(currentYear, currentMonth);
        const startDay = Math.max(1, selectedDay - 3);
        for (let i = 0; i < 8; i++) {
            const day = startDay + i;
            if (day > daysInMonth) break;
            const dateStr = formatDate(currentYear, currentMonth, day);
            const log = cal.outfitLogs[dateStr];
            days.push({
                day,
                weekday: weekdays[(new Date(currentYear, currentMonth, day).getDay()) % 7],
                hasOutfit: log && log.items.length > 0,
                outfitImage: log?.items[0]?.image,
                isSelected: day === selectedDay,
            });
        }
        return days;
    }, [selectedDay, currentMonth, currentYear, cal.outfitLogs]);

    // Get selected day outfit
    const selectedDateStr = formatDate(currentYear, currentMonth, selectedDay);
    const selectedOutfit = cal.outfitLogs[selectedDateStr];

    // Group items by type using shared matchesCategory
    const outfitItems = useMemo(() => {
        if (!selectedOutfit) return { top: null, pants: null, shoes: null };
        return {
            top: selectedOutfit.items.find(i => matchesCategory(i.type, 'top')) ?? null,
            pants: selectedOutfit.items.find(i => matchesCategory(i.type, 'pants')) ?? null,
            shoes: selectedOutfit.items.find(i => matchesCategory(i.type, 'shoes')) ?? null,
        };
    }, [selectedOutfit]);

    const handleDayPress = (day: number) => {
        setSelectedDay(day);
        cal.handleDayPress(day);
    };

    const handlePrevMonth = () => {
        cal.goToPrevMonth();
        const prevMonth = currentMonth === 0 ? 11 : currentMonth - 1;
        const prevYear = currentMonth === 0 ? currentYear - 1 : currentYear;
        const isCurrentMonth = today.getMonth() === prevMonth && today.getFullYear() === prevYear;
        setSelectedDay(isCurrentMonth ? today.getDate() : 1);
    };

    const handleNextMonth = () => {
        cal.goToNextMonth();
        const nextMonth = currentMonth === 11 ? 0 : currentMonth + 1;
        const nextYear = currentMonth === 11 ? currentYear + 1 : currentYear;
        const isCurrentMonth = today.getMonth() === nextMonth && today.getFullYear() === nextYear;
        setSelectedDay(isCurrentMonth ? today.getDate() : 1);
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Top Bar */}
                <View style={styles.topBar}>
                    <TouchableOpacity style={styles.glassButton}>
                        <Ionicons name="share-outline" size={20} color="#1E293B" />
                    </TouchableOpacity>

                    <View style={styles.streakPill}>
                        <Text style={styles.streakFire}>🔥</Text>
                        <Text style={styles.streakCount}>{cal.streak}</Text>
                        <View style={styles.streakSep} />
                        <View>
                            <Text style={styles.streakTitle}>STREAK</Text>
                            <Text style={styles.streakDays}>{cal.streak} DAYS</Text>
                        </View>
                    </View>

                    <TouchableOpacity style={styles.glassButton} onPress={() => navigation.goBack()}>
                        <Ionicons name="close" size={20} color="#1E293B" />
                    </TouchableOpacity>
                </View>

                {/* Month Navigation */}
                <View style={styles.monthPill}>
                    <TouchableOpacity onPress={handlePrevMonth} style={styles.monthArrow}>
                        <Ionicons name="chevron-back" size={22} color="#1E293B" />
                    </TouchableOpacity>
                    <Text style={styles.monthLabel}>{monthNames[currentMonth]} {currentYear}</Text>
                    <TouchableOpacity onPress={handleNextMonth} style={styles.monthArrow}>
                        <Ionicons name="chevron-forward" size={22} color="#1E293B" />
                    </TouchableOpacity>
                </View>

                <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
                    {/* Days Strip */}
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.daysRow} style={styles.daysRowWrap}>
                        {daysStrip.map((item, idx) => (
                            <Animated.View key={idx} entering={FadeIn.delay(idx * 40)}>
                                <TouchableOpacity
                                    style={[styles.dayCard, item.isSelected && styles.dayCardActive]}
                                    onPress={() => handleDayPress(item.day)}
                                    activeOpacity={0.75}
                                >
                                    <Text style={[styles.dayWeekday, item.isSelected && styles.dayTextWhite]}>
                                        {item.weekday}
                                    </Text>
                                    <View style={[styles.dayNumBubble, item.isSelected && styles.dayNumBubbleActive]}>
                                        <Text style={[styles.dayNum, item.isSelected && styles.dayTextWhite]}>
                                            {item.day}
                                        </Text>
                                    </View>
                                    {item.hasOutfit ? (
                                        <View style={styles.miniSlots}>
                                            <View style={styles.miniSlot}>{outfitItems.top && <Image source={{ uri: outfitItems.top.image }} style={styles.miniImg} />}</View>
                                            <View style={styles.miniSlot}>{outfitItems.pants && <Image source={{ uri: outfitItems.pants.image }} style={styles.miniImg} />}</View>
                                            <View style={styles.miniSlot}>{outfitItems.shoes && <Image source={{ uri: outfitItems.shoes.image }} style={styles.miniImg} />}</View>
                                        </View>
                                    ) : (
                                        <View style={styles.miniSlots}>
                                            <View style={styles.emptyMini}><Text style={styles.emptyMiniText}>+</Text></View>
                                            <View style={styles.emptyMini}><Text style={styles.emptyMiniText}>+</Text></View>
                                            <TouchableOpacity style={styles.addMiniBtn} onPress={() => cal.setShowLogModal(true)}>
                                                <Text style={styles.addMiniBtnText}>ADD</Text>
                                            </TouchableOpacity>
                                        </View>
                                    )}
                                </TouchableOpacity>
                            </Animated.View>
                        ))}
                    </ScrollView>

                    {/* Outfit Detail */}
                    {selectedOutfit ? (
                        <Animated.View entering={FadeInDown.duration(300)} style={styles.detailWrap}>
                            {/* Main Outfit Glass Card */}
                            <View style={styles.glassCard}>
                                <View style={styles.cardTopRow}>
                                    <View style={styles.cardLabelPill}><Text style={styles.cardLabelText}>OUTFIT</Text></View>
                                    <TouchableOpacity style={styles.editPill}><Text style={styles.editPillText}>Edit</Text></TouchableOpacity>
                                </View>
                                <View style={styles.mainImgBox}>
                                    {selectedOutfit.items[0] && <Image source={{ uri: selectedOutfit.items[0].image }} style={styles.mainImg} resizeMode="cover" />}
                                </View>
                                <View style={styles.cardFooter}>
                                    <Text style={styles.outfitTitle}>Adaptive Exo-Skin Ensemble</Text>
                                    <TouchableOpacity style={styles.editCircle}>
                                        <Ionicons name="create-outline" size={16} color="#64748B" />
                                    </TouchableOpacity>
                                </View>
                            </View>

                            {/* Item Cards Row */}
                            <View style={styles.itemRow}>
                                {outfitItems.top && (
                                    <View style={styles.itemGlass}>
                                        <View style={styles.itemLabelRow}>
                                            <View style={styles.itemTag}><Text style={styles.itemTagText}>TOP</Text></View>
                                            <TouchableOpacity><Ionicons name="copy-outline" size={14} color="#94A3B8" /></TouchableOpacity>
                                        </View>
                                        <Image source={{ uri: outfitItems.top.image }} style={styles.itemImg} resizeMode="cover" />
                                        <Text style={styles.itemName} numberOfLines={2}>Bio-Pulse Tee</Text>
                                    </View>
                                )}
                                {outfitItems.shoes && (
                                    <View style={styles.itemGlass}>
                                        <View style={styles.itemLabelRow}>
                                            <View style={styles.itemTag}><Text style={styles.itemTagText}>SHOES</Text></View>
                                        </View>
                                        <Image source={{ uri: outfitItems.shoes.image }} style={styles.itemImg} resizeMode="cover" />
                                        <Text style={styles.itemName} numberOfLines={2}>Plasma-Step Runners</Text>
                                    </View>
                                )}
                            </View>

                            <View style={styles.itemRow}>
                                {outfitItems.pants && (
                                    <View style={styles.itemGlass}>
                                        <View style={styles.itemLabelRow}>
                                            <View style={styles.itemTag}><Text style={styles.itemTagText}>PANTS</Text></View>
                                        </View>
                                        <Image source={{ uri: outfitItems.pants.image }} style={styles.itemImg} resizeMode="cover" />
                                        <Text style={styles.itemName} numberOfLines={2}>Kinetic-Warp Cargo Trousers</Text>
                                        <TouchableOpacity style={styles.editCircle}>
                                            <Ionicons name="create-outline" size={14} color="#94A3B8" />
                                        </TouchableOpacity>
                                    </View>
                                )}
                                {!outfitItems.pants && <View style={[styles.itemGlass, styles.itemGlassEmpty]}><Text style={styles.emptySlotText}>No pants logged</Text></View>}
                            </View>
                        </Animated.View>
                    ) : (
                        <Animated.View entering={FadeInDown.duration(400)} style={styles.emptyWrap}>
                            <View style={styles.emptyIconCircle}>
                                <Text style={{ fontSize: 36 }}>👕</Text>
                            </View>
                            <Text style={styles.emptyTitle}>No Outfit Logged</Text>
                            <Text style={styles.emptySubtitle}>Tap below to record what you wore today</Text>
                            <TouchableOpacity style={styles.logPill} onPress={() => navigation.navigate('AIOutfit' as never)}>
                                <Ionicons name="sparkles" size={20} color="#FFF" />
                                <Text style={styles.logPillText}>Create Outfit with AI</Text>
                            </TouchableOpacity>
                        </Animated.View>
                    )}
                </ScrollView>

                {/* Modal */}
                <OutfitLogForm
                    visible={cal.showLogModal}
                    wardrobeItems={resolvedWardrobeItems}
                    shopItems={shopItems}
                    inspoShopItems={INSPO_SHOP_ITEMS}
                    selectedItems={cal.selectedItems}
                    selectedOccasion={cal.selectedOccasion}
                    onClose={() => cal.setShowLogModal(false)}
                    onToggleItem={cal.toggleItemSelection}
                    onSelectOccasion={cal.setSelectedOccasion}
                    onSave={cal.saveOutfit}
                />
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F0F2F8' },
    safeArea: { flex: 1 },

    // Top Bar
    topBar: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingTop: 10,
        paddingBottom: 14,
    },
    glassButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.95)',
        justifyContent: 'center',
        alignItems: 'center',
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.18,
        shadowRadius: 12,
        elevation: 4,
    },
    streakPill: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.88)',
        paddingHorizontal: 18,
        paddingVertical: 10,
        borderRadius: 30,
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.95)',
        gap: 8,
        shadowColor: '#F59E0B',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.15,
        shadowRadius: 12,
        elevation: 4,
    },
    streakFire: { fontSize: 22 },
    streakCount: { fontSize: 22, fontWeight: '900', color: '#0F172A' },
    streakSep: { width: 1.5, height: 26, backgroundColor: '#E2E8F0', marginHorizontal: 2 },
    streakTitle: { fontSize: 9, fontWeight: '800', color: '#94A3B8', letterSpacing: 1 },
    streakDays: { fontSize: 11, fontWeight: '700', color: '#F59E0B' },

    // Month Navigation
    monthPill: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginHorizontal: 20,
        marginBottom: 18,
        paddingVertical: 14,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderRadius: 30,
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.95)',
        gap: 24,
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.12,
        shadowRadius: 16,
        elevation: 3,
    },
    monthArrow: { padding: 4 },
    monthLabel: { fontSize: 18, fontWeight: '700', color: '#0F172A', letterSpacing: -0.3 },

    scrollContent: { paddingBottom: 50 },

    // Day Strip
    daysRowWrap: { marginBottom: 22 },
    daysRow: { paddingHorizontal: 16, gap: 10 },
    dayCard: {
        width: 72,
        backgroundColor: 'rgba(255,255,255,0.82)',
        borderRadius: 22,
        paddingTop: 10,
        paddingBottom: 8,
        paddingHorizontal: 7,
        alignItems: 'center',
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.95)',
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.1,
        shadowRadius: 10,
        elevation: 2,
    },
    dayCardActive: {
        backgroundColor: '#0F172A',
        borderColor: '#1E293B',
    },
    dayWeekday: {
        fontSize: 10,
        fontWeight: '700',
        color: '#94A3B8',
        letterSpacing: 0.5,
        marginBottom: 5,
        textTransform: 'uppercase',
    },
    dayNumBubble: {
        width: 34,
        height: 34,
        borderRadius: 17,
        backgroundColor: '#F1F5F9',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 8,
    },
    dayNumBubbleActive: { backgroundColor: 'rgba(255,255,255,0.15)' },
    dayNum: { fontSize: 16, fontWeight: '800', color: '#0F172A' },
    dayTextWhite: { color: '#FFFFFF' },
    miniSlots: { width: '100%', gap: 4 },
    miniSlot: {
        width: '100%',
        height: 26,
        borderRadius: 8,
        backgroundColor: '#F1F5F9',
        overflow: 'hidden',
    },
    miniImg: { width: '100%', height: '100%' },
    emptyMini: {
        width: '100%',
        height: 22,
        borderRadius: 8,
        borderWidth: 1,
        borderColor: '#E2E8F0',
        justifyContent: 'center',
        alignItems: 'center',
    },
    emptyMiniText: { fontSize: 10, color: '#CBD5E1', fontWeight: '700' },
    addMiniBtn: {
        width: '100%',
        height: 22,
        borderRadius: 8,
        backgroundColor: '#0F172A',
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: 2,
    },
    addMiniBtnText: { fontSize: 8, fontWeight: '800', color: '#FFF', letterSpacing: 0.5 },

    // Outfit Detail
    detailWrap: { paddingHorizontal: 16, gap: 14 },
    glassCard: {
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderRadius: 28,
        padding: 16,
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.95)',
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.14,
        shadowRadius: 20,
        elevation: 5,
    },
    cardTopRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 14,
    },
    cardLabelPill: {
        backgroundColor: '#F1F5F9',
        paddingHorizontal: 12,
        paddingVertical: 5,
        borderRadius: 20,
    },
    cardLabelText: { fontSize: 11, fontWeight: '800', color: '#64748B', letterSpacing: 1 },
    editPill: {
        backgroundColor: '#F8FAFC',
        paddingHorizontal: 14,
        paddingVertical: 5,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: '#E2E8F0',
    },
    editPillText: { fontSize: 12, fontWeight: '600', color: '#475569' },
    mainImgBox: {
        width: '100%',
        height: 210,
        borderRadius: 20,
        overflow: 'hidden',
        backgroundColor: '#F1F5F9',
        marginBottom: 14,
    },
    mainImg: { width: '100%', height: '100%' },
    cardFooter: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    outfitTitle: { fontSize: 15, fontWeight: '700', color: '#0F172A', flex: 1, marginRight: 8 },
    editCircle: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: '#F1F5F9',
        justifyContent: 'center',
        alignItems: 'center',
    },

    // Item Cards
    itemRow: { flexDirection: 'row', gap: 14 },
    itemGlass: {
        flex: 1,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderRadius: 24,
        padding: 12,
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.95)',
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 14,
        elevation: 3,
    },
    itemGlassEmpty: {
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: 140,
        opacity: 0.5,
    },
    emptySlotText: { fontSize: 12, color: '#94A3B8', fontWeight: '500' },
    itemLabelRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 10,
    },
    itemTag: {
        backgroundColor: '#F0F4FF',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 14,
    },
    itemTagText: { fontSize: 10, fontWeight: '800', color: '#6366F1', letterSpacing: 0.5 },
    itemImg: {
        width: '100%',
        height: 100,
        borderRadius: 16,
        backgroundColor: '#F1F5F9',
        marginBottom: 10,
    },
    itemName: { fontSize: 12, fontWeight: '600', color: '#0F172A', lineHeight: 16 },

    // Empty State
    emptyWrap: {
        alignItems: 'center',
        paddingVertical: 50,
        paddingHorizontal: 32,
    },
    emptyIconCircle: {
        width: 90,
        height: 90,
        borderRadius: 45,
        backgroundColor: 'rgba(255,255,255,0.9)',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 20,
        shadowColor: '#7C8DB5',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.15,
        shadowRadius: 20,
        elevation: 5,
    },
    emptyTitle: { fontSize: 20, fontWeight: '800', color: '#0F172A', marginBottom: 8 },
    emptySubtitle: { fontSize: 14, color: '#94A3B8', textAlign: 'center', marginBottom: 32, lineHeight: 20 },
    logPill: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#0F172A',
        paddingHorizontal: 28,
        paddingVertical: 16,
        borderRadius: 30,
        gap: 8,
        shadowColor: '#0F172A',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.3,
        shadowRadius: 16,
        elevation: 6,
    },
    logPillText: { color: '#FFFFFF', fontSize: 16, fontWeight: '700' },
});

export default OutfitCalendarScreen;
