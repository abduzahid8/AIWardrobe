/**
 * OutfitCalendarScreen — Thin orchestrator.
 *
 * Logic lives in useOutfitCalendar hook.
 * UI split into CalendarGrid, DayDetailModal, and OutfitLogForm.
 *
 * Reduced from 1,127 lines → ~150 lines.
 */

import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeIn, FadeInDown, FadeInUp } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';

import { colors, spacing, shadows, borderRadius } from '../src/theme';
import { useOutfitCalendar, OCCASIONS, getOccasionColor } from '../features/calendar/hooks/useOutfitCalendar';
import { CalendarGrid } from '../features/calendar/components/CalendarGrid';
import { DayDetailModal } from '../features/calendar/components/DayDetailModal';
import { OutfitLogForm } from '../features/calendar/components/OutfitLogForm';

const OutfitCalendarScreen = () => {
    const navigation = useNavigation();
    const cal = useOutfitCalendar();

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
                        {cal.streak > 0 && (
                            <View style={styles.streakBadge}>
                                <Text style={styles.streakEmoji}>🔥</Text>
                                <Text style={styles.streakNumber}>{cal.streak}</Text>
                            </View>
                        )}
                    </Animated.View>

                    {/* Today Card */}
                    <Animated.View entering={FadeInDown.delay(100)}>
                        <TouchableOpacity
                            style={styles.todayCard}
                            onPress={() => {
                                if (!cal.todaysOutfit) {
                                    cal.setSelectedDate(cal.todayStr);
                                    cal.setShowLogModal(true);
                                }
                            }}
                            activeOpacity={cal.todaysOutfit ? 1 : 0.8}
                        >
                            <LinearGradient
                                colors={cal.todaysOutfit ? ['#22C55E', '#16A34A'] : ['#1A1A1A', '#2D2D2D']}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 1 }}
                                style={styles.todayGradient}
                            >
                                <View style={styles.todayContent}>
                                    <View>
                                        <Text style={styles.todayLabel}>TODAY</Text>
                                        <Text style={styles.todayTitle}>
                                            {cal.todaysOutfit ? 'Outfit Logged ✓' : 'Log Your Outfit'}
                                        </Text>
                                        <Text style={styles.todayDate}>
                                            {cal.today.toLocaleDateString('en-US', { weekday: 'long', month: 'short', day: 'numeric' })}
                                        </Text>
                                    </View>
                                    {cal.todaysOutfit ? (
                                        <View style={styles.todayItems}>
                                            {cal.todaysOutfit.items.slice(0, 3).map((item, idx) => (
                                                <View key={idx} style={styles.todayThumb}>
                                                    <Image source={{ uri: item.image }} style={styles.todayThumbImg} />
                                                </View>
                                            ))}
                                        </View>
                                    ) : (
                                        <View style={styles.addIcon}>
                                            <Ionicons name="add" size={28} color="#FFF" />
                                        </View>
                                    )}
                                </View>
                            </LinearGradient>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* Quick Actions */}
                    <Animated.View entering={FadeInDown.delay(150)} style={styles.quickActions}>
                        {[
                            { icon: 'videocam', label: 'Scan Wardrobe', screen: 'WardrobeVideo', bg: '#EFF6FF', fg: '#3B82F6' },
                            { icon: 'sparkles', label: 'Get AI Outfit', screen: 'AIChat', bg: '#FFF1F2', fg: '#EC4899' },
                            { icon: 'stats-chart', label: 'Wardrobe Stats', screen: 'Stats', bg: '#F0FDF4', fg: '#22C55E' },
                        ].map(a => (
                            <TouchableOpacity key={a.screen} style={styles.quickAction} onPress={() => (navigation as any).navigate(a.screen)}>
                                <View style={[styles.quickActionIcon, { backgroundColor: a.bg }]}>
                                    <Ionicons name={a.icon as any} size={20} color={a.fg} />
                                </View>
                                <Text style={styles.quickActionText}>{a.label}</Text>
                            </TouchableOpacity>
                        ))}
                    </Animated.View>

                    {/* Calendar */}
                    <Animated.View entering={FadeInUp.delay(200)}>
                        <CalendarGrid
                            currentMonth={cal.currentMonth}
                            currentYear={cal.currentYear}
                            todayStr={cal.todayStr}
                            today={cal.today}
                            outfitLogs={cal.outfitLogs}
                            onDayPress={cal.handleDayPress}
                            onPrevMonth={cal.goToPrevMonth}
                            onNextMonth={cal.goToNextMonth}
                            monthlyStats={cal.getMonthlyStats()}
                        />
                    </Animated.View>

                    {/* Recent Outfits */}
                    <Animated.View entering={FadeInUp.delay(300)} style={styles.recentSection}>
                        <Text style={styles.sectionTitle}>Recent Outfits</Text>
                        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                            {Object.entries(cal.outfitLogs)
                                .sort((a, b) => b[0].localeCompare(a[0]))
                                .slice(0, 5)
                                .map(([date, log]) => (
                                    <TouchableOpacity
                                        key={date}
                                        style={styles.recentCard}
                                        onPress={() => { cal.setSelectedDate(date); cal.setShowDayModal(true); }}
                                    >
                                        <View style={styles.recentImages}>
                                            {log.items.slice(0, 2).map((item, idx) => (
                                                <Image key={idx} source={{ uri: item.image }} style={[styles.recentImage, idx === 1 && styles.recentOverlap]} />
                                            ))}
                                        </View>
                                        <Text style={styles.recentDate}>
                                            {new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}
                                        </Text>
                                    </TouchableOpacity>
                                ))}
                            {Object.keys(cal.outfitLogs).length === 0 && (
                                <Text style={styles.emptyText}>No outfits logged yet</Text>
                            )}
                        </ScrollView>
                    </Animated.View>

                    <View style={{ height: spacing.xxl }} />
                </ScrollView>

                {/* Modals */}
                <OutfitLogForm
                    visible={cal.showLogModal}
                    wardrobeItems={cal.wardrobeItems}
                    selectedItems={cal.selectedItems}
                    selectedOccasion={cal.selectedOccasion}
                    onClose={() => cal.setShowLogModal(false)}
                    onToggleItem={cal.toggleItemSelection}
                    onSelectOccasion={cal.setSelectedOccasion}
                    onSave={cal.saveOutfit}
                />
                <DayDetailModal
                    visible={cal.showDayModal}
                    selectedDate={cal.selectedDate}
                    outfitLogs={cal.outfitLogs}
                    onClose={() => cal.setShowDayModal(false)}
                    onDelete={cal.confirmDelete}
                />
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: colors.background },
    safeArea: { flex: 1 },
    header: {
        flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
        paddingHorizontal: spacing.l, paddingVertical: spacing.m,
    },
    title: { fontSize: 28, fontWeight: '800', color: colors.text.primary },
    subtitle: { fontSize: 14, color: colors.text.secondary, marginTop: 2 },
    streakBadge: {
        flexDirection: 'row', alignItems: 'center',
        backgroundColor: '#FFF7ED', paddingHorizontal: spacing.m, paddingVertical: spacing.s,
        borderRadius: borderRadius.full, gap: 4,
    },
    streakEmoji: { fontSize: 18 },
    streakNumber: { fontSize: 18, fontWeight: '800', color: '#EA580C' },
    todayCard: { marginHorizontal: spacing.l, marginBottom: spacing.m, borderRadius: borderRadius.xl, overflow: 'hidden' },
    todayGradient: { padding: spacing.m },
    todayContent: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
    todayLabel: { fontSize: 11, fontWeight: '600', color: 'rgba(255,255,255,0.7)', letterSpacing: 1, marginBottom: 4 },
    todayTitle: { fontSize: 20, fontWeight: '700', color: '#FFF', marginBottom: 4 },
    todayDate: { fontSize: 13, color: 'rgba(255,255,255,0.8)' },
    todayItems: { flexDirection: 'row' },
    todayThumb: { width: 40, height: 40, borderRadius: 8, overflow: 'hidden', marginLeft: -8, borderWidth: 2, borderColor: '#22C55E' },
    todayThumbImg: { width: '100%', height: '100%' },
    addIcon: { width: 48, height: 48, borderRadius: 24, backgroundColor: 'rgba(255,255,255,0.2)', justifyContent: 'center', alignItems: 'center' },
    quickActions: { flexDirection: 'row', paddingHorizontal: spacing.l, gap: spacing.s, marginBottom: spacing.m },
    quickAction: { flex: 1, backgroundColor: colors.surface, borderRadius: borderRadius.l, padding: spacing.s, alignItems: 'center', ...shadows.soft },
    quickActionIcon: { width: 40, height: 40, borderRadius: 20, justifyContent: 'center', alignItems: 'center', marginBottom: spacing.xs },
    quickActionText: { fontSize: 11, fontWeight: '600', color: colors.text.primary, textAlign: 'center' },
    recentSection: { marginTop: spacing.m, paddingLeft: spacing.l },
    sectionTitle: { fontSize: 18, fontWeight: '700', color: colors.text.primary, marginBottom: spacing.m },
    recentCard: { width: 100, marginRight: spacing.m, alignItems: 'center' },
    recentImages: { flexDirection: 'row', marginBottom: spacing.xs },
    recentImage: { width: 44, height: 44, borderRadius: 8, backgroundColor: colors.surfaceHighlight },
    recentOverlap: { marginLeft: -12 },
    recentDate: { fontSize: 12, fontWeight: '600', color: colors.text.primary },
    emptyText: { fontSize: 14, color: colors.text.secondary, paddingVertical: spacing.xl },
});

export default OutfitCalendarScreen;
