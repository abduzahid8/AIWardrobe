import React, { useMemo } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import { ScreenWrapper } from '../components/ui/ScreenWrapper';
import { LiquidGlassCard } from '../components/ui/LiquidGlassCard';
import { QuickStat } from '../components/ui/QuickStat';
import { StatsCard } from '../components/ui/StatsCard';
import { CachedImage } from '../components/ui/CachedImage';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeInRight } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing, typography } from '../src/theme';
import useWardrobeStore from '../store/wardrobeStore';
import {
  calculateStreak,
  getClosetUtilization,
  getUnwornItems,
  generateStyleInsights,
} from '../src/services/retentionService';
import type { ClothingItem } from '../src/types/domain';
import { useTranslation } from 'react-i18next';

const UnwornItemCard: React.FC<{ item: ClothingItem; onPress?: () => void }> = ({ item, onPress }) => (
  <TouchableOpacity style={styles.unwornCard} onPress={onPress} disabled={!onPress}>
    {item.imageUrl ? (
      <CachedImage uri={item.imageUrl} style={styles.unwornImage} contentFit="cover" fadeIn={false} />
    ) : (
      <View style={[styles.unwornImage, styles.unwornPlaceholder]}>
        <View style={[styles.unwornColorDot, { backgroundColor: item.colorHex || '#CCC' }]} />
      </View>
    )}
    <ScaledText style={styles.unwornLabel} numberOfLines={2}>
      {item.subCategory || item.category}
    </ScaledText>
  </TouchableOpacity>
);

const WeeklyInsightsScreen: React.FC<{ navigation: any }> = ({ navigation }) => {
  const { t } = useTranslation();
  const items = useWardrobeStore((state) => state.items);
  const wearLogs = useWardrobeStore((state) => state.wearLogs);

  const streak = useMemo(() => calculateStreak(wearLogs), [wearLogs]);
  const utilization = useMemo(() => getClosetUtilization(items, wearLogs, 30), [items, wearLogs]);
  const unwornItems = useMemo(() => getUnwornItems(items, wearLogs, 30), [items, wearLogs]);
  const insights = useMemo(() => generateStyleInsights(items, wearLogs), [items, wearLogs]);

  const mostWorn = useMemo(() => {
    return [...items]
      .filter((i) => i.wearCount > 0)
      .sort((a, b) => b.wearCount - a.wearCount)
      .slice(0, 5);
  }, [items]);

  const totalWears = useMemo(() => {
    return items.reduce((sum, i) => sum + i.wearCount, 0);
  }, [items]);

  const categoryBreakdown = useMemo(() => {
    const counts: Record<string, number> = {};
    items.forEach((item) => {
      counts[item.category] = (counts[item.category] || 0) + 1;
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [items]);

  return (
    <ScreenWrapper animation="fade">
      <LinearGradient
        colors={['rgba(188, 210, 245, 0.32)', 'rgba(216, 229, 252, 0.24)', '#FFFFFF']}
        style={StyleSheet.absoluteFill}
        pointerEvents="none"
      />
      <View pointerEvents="none" style={styles.backgroundOrbTop} />
      <View pointerEvents="none" style={styles.backgroundOrbBottom} />

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
            <Ionicons name="chevron-back" size={24} color={colors.text.primary} />
          </TouchableOpacity>
          <ScaledText style={styles.headerTitle}>{t('weeklyInsights.title')}</ScaledText>
          <TouchableOpacity
            onPress={() => navigation.navigate('WardrobeAnalytics')}
            style={styles.headerAction}
          >
            <Ionicons name="analytics-outline" size={22} color={colors.primary} />
          </TouchableOpacity>
        </View>

        {/* Hero Stats */}
        <Animated.View entering={FadeInDown.delay(100).duration(400)} style={styles.quickStatsRow}>
          <QuickStat icon="shirt-outline" value={items.length} label={t('weeklyInsights.totalItems')} color={colors.primary} index={0} />
          <QuickStat icon="flame-outline" value={streak} label={t('weeklyInsights.dayStreak')} color="#F97316" index={1} />
          <QuickStat icon="repeat-outline" value={totalWears} label={t('weeklyInsights.totalWears')} color="#8B5CF6" index={2} />
        </Animated.View>

        {/* Closet Utilization */}
        <Animated.View entering={FadeInDown.delay(200).duration(400)} style={styles.sectionContainer}>
          <LiquidGlassCard variant="opaque" radius="xl">
            <StatsCard
              title={t('weeklyInsights.closetUtilization')}
              value={utilization}
              subtitle={`${utilization >= 80 ? 'Excellent rotation!' : utilization >= 50 ? 'Getting better!' : 'Room for improvement'} — ${items.length - unwornItems.length}/${items.length} items active`}
              progress={utilization / 100}
              icon="pie-chart"
              trend={utilization >= 50 ? 'up' : 'down'}
              trendValue={`${utilization}% worn in 30 days`}
              gradient
              gradientColors={['#0A1931', '#1E3A5F']}
            />
          </LiquidGlassCard>
        </Animated.View>

        {/* Unworn Items */}
        {unwornItems.length > 0 && (
          <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: 'rgba(249,115,22,0.12)' }]}>
                    <Ionicons name="sparkles-outline" size={20} color="#F97316" />
                  </View>
                  <ScaledText style={styles.cardTitle}>
                    {t('weeklyInsights.unwornItems', { count: unwornItems.length })}
                  </ScaledText>
                </View>
              </View>
              <ScaledText style={styles.cardSubtext}>
                {t('weeklyInsights.unwornItemsSubtext')}
              </ScaledText>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.horizontalScroll}>
                {unwornItems.slice(0, 8).map((item) => (
                  <UnwornItemCard
                    key={item.id}
                    item={item}
                    onPress={() => navigation.navigate('ClothingDetail', { itemId: item.id })}
                  />
                ))}
              </ScrollView>
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Most Worn */}
        {mostWorn.length > 0 && (
          <Animated.View entering={FadeInDown.delay(400).duration(400)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: 'rgba(16,185,129,0.12)' }]}>
                    <Ionicons name="trending-up-outline" size={20} color="#10B981" />
                  </View>
                  <ScaledText style={styles.cardTitle}>{t('weeklyInsights.mostWorn')}</ScaledText>
                </View>
              </View>
              {mostWorn.map((item, index) => (
                <TouchableOpacity
                  key={item.id}
                  style={styles.rankRow}
                  onPress={() => navigation.navigate('ClothingDetail', { itemId: item.id })}
                >
                  <ScaledText style={styles.rankNumber}>{index + 1}</ScaledText>
                  {item.imageUrl ? (
                    <CachedImage uri={item.imageUrl} style={styles.rankImage} contentFit="cover" fadeIn={false} />
                  ) : (
                    <View style={[styles.rankImage, styles.rankPlaceholder]}>
                      <View style={[styles.rankColor, { backgroundColor: item.colorHex || '#CCC' }]} />
                    </View>
                  )}
                  <View style={styles.rankInfo}>
                    <ScaledText style={styles.rankName} numberOfLines={1}>
                      {item.subCategory || item.category}
                    </ScaledText>
                    <ScaledText style={styles.rankMeta}>{item.primaryColor}</ScaledText>
                  </View>
                  <ScaledText style={styles.rankCount}>{item.wearCount}×</ScaledText>
                </TouchableOpacity>
              ))}
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Category Breakdown */}
        {categoryBreakdown.length > 0 && (
          <Animated.View entering={FadeInDown.delay(500).duration(400)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: 'rgba(139,92,246,0.12)' }]}>
                    <Ionicons name="grid-outline" size={20} color="#8B5CF6" />
                  </View>
                  <ScaledText style={styles.cardTitle}>{t('weeklyInsights.wardrobeBreakdown')}</ScaledText>
                </View>
              </View>
              {categoryBreakdown.map(([category, count]) => {
                const percentage = Math.round((count / items.length) * 100);
                return (
                  <View key={category} style={styles.breakdownRow}>
                    <ScaledText style={styles.breakdownLabel}>
                      {category.charAt(0).toUpperCase() + category.slice(1)}
                    </ScaledText>
                    <View style={styles.breakdownTrack}>
                      <View style={[styles.breakdownFill, { width: `${percentage}%` }]} />
                    </View>
                    <ScaledText style={styles.breakdownCount}>{count}</ScaledText>
                  </View>
                );
              })}
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Style Insights */}
        {insights.length > 0 && (
          <Animated.View entering={FadeInDown.delay(600).duration(400)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: 'rgba(239,68,68,0.12)' }]}>
                    <Ionicons name="bulb-outline" size={20} color="#EF4444" />
                  </View>
                  <ScaledText style={styles.cardTitle}>{t('weeklyInsights.patternsHabits')}</ScaledText>
                </View>
              </View>
              {insights
                .filter((i) => i.type !== 'utilization')
                .slice(0, 4)
                .map((insight, index) => (
                  <View key={`${insight.type}_${index}`} style={styles.insightRow}>
                    <ScaledText style={styles.insightTitle}>{insight.title}</ScaledText>
                    <ScaledText style={styles.insightDesc}>{insight.description}</ScaledText>
                  </View>
                ))}
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Navigate to Full Analytics */}
        <Animated.View entering={FadeInDown.delay(700).duration(400)} style={styles.sectionContainer}>
          <TouchableOpacity
            style={styles.fullAnalyticsButton}
            onPress={() => navigation.navigate('WardrobeAnalytics')}
          >
            <Ionicons name="bar-chart-outline" size={22} color="#FFF" />
            <ScaledText style={styles.fullAnalyticsText}>View Full Analytics Dashboard</ScaledText>
            <Ionicons name="arrow-forward" size={20} color="#FFF" />
          </TouchableOpacity>
        </Animated.View>

        {/* Empty State */}
        {items.length === 0 && (
          <View style={styles.emptyState}>
            <Ionicons name="analytics-outline" size={56} color={colors.text.muted} />
            <ScaledText style={styles.emptyTitle}>{t('weeklyInsights.noDataYet')}</ScaledText>
            <ScaledText style={styles.emptySubtext}>{t('weeklyInsights.emptySubtext')}</ScaledText>
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </ScreenWrapper>
  );
};

const styles = StyleSheet.create({
  backgroundOrbTop: {
    position: 'absolute',
    top: -120,
    left: -60,
    width: 240,
    height: 240,
    borderRadius: 120,
    backgroundColor: 'rgba(188, 210, 245, 0.35)',
  },
  backgroundOrbBottom: {
    position: 'absolute',
    right: -100,
    bottom: 160,
    width: 260,
    height: 260,
    borderRadius: 130,
    backgroundColor: 'rgba(216, 229, 252, 0.28)',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing.l,
    paddingVertical: spacing.m,
  },
  backButton: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.84)',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  headerTitle: {
    ...typography.h2,
    color: colors.text.primary,
  },
  headerAction: {
    width: 36,
    height: 36,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 18,
    backgroundColor: 'rgba(255,255,255,0.84)',
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.08)',
  },
  scrollContent: {
    paddingHorizontal: spacing.l,
    paddingBottom: spacing.xxl,
  },
  quickStatsRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: spacing.m,
    gap: spacing.s,
  },
  sectionContainer: {
    marginBottom: spacing.m,
  },
  cardHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.s,
    gap: spacing.s,
  },
  cardHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.s,
  },
  iconBadge: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  cardTitle: {
    ...typography.h3,
    color: colors.text.primary,
  },
  cardSubtext: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    lineHeight: 18,
    marginBottom: spacing.s,
  },
  horizontalScroll: {
    marginHorizontal: -spacing.xs,
  },
  // Unworn
  unwornCard: {
    width: 80,
    alignItems: 'center',
    marginHorizontal: spacing.xs,
    gap: spacing.xs,
  },
  unwornImage: {
    width: 64,
    height: 64,
    borderRadius: 16,
  },
  unwornPlaceholder: {
    backgroundColor: 'rgba(0,0,0,0.05)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  unwornColorDot: {
    width: 24,
    height: 24,
    borderRadius: 12,
  },
  unwornLabel: {
    fontSize: 11,
    color: colors.text.muted,
    textAlign: 'center',
  },
  // Most worn
  rankRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.s,
    paddingVertical: spacing.xs,
  },
  rankNumber: {
    fontSize: 14,
    color: colors.text.muted,
    fontWeight: '700',
    width: 24,
  },
  rankImage: {
    width: 40,
    height: 40,
    borderRadius: 12,
  },
  rankPlaceholder: {
    backgroundColor: 'rgba(0,0,0,0.05)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  rankColor: {
    width: 20,
    height: 20,
    borderRadius: 10,
  },
  rankInfo: {
    flex: 1,
  },
  rankName: {
    fontSize: 14,
    color: colors.text.primary,
    fontWeight: '600',
  },
  rankMeta: {
    fontSize: 12,
    color: colors.text.muted,
  },
  rankCount: {
    fontSize: 14,
    color: colors.text.secondary,
    fontWeight: '700',
  },
  // Category breakdown
  breakdownRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.s,
    marginBottom: spacing.xs,
  },
  breakdownLabel: {
    fontSize: 12,
    color: colors.text.secondary,
    width: 80,
  },
  breakdownTrack: {
    flex: 1,
    height: 6,
    backgroundColor: 'rgba(0,0,0,0.08)',
    borderRadius: 3,
    overflow: 'hidden',
  },
  breakdownFill: {
    height: '100%',
    backgroundColor: '#8B5CF6',
    borderRadius: 3,
  },
  breakdownCount: {
    fontSize: 12,
    color: colors.text.muted,
    width: 28,
    textAlign: 'right',
  },
  // Insights
  insightRow: {
    gap: 2,
    marginBottom: spacing.s,
    paddingBottom: spacing.s,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(0,0,0,0.06)',
  },
  insightTitle: {
    fontSize: 14,
    color: colors.text.primary,
    fontWeight: '600',
  },
  insightDesc: {
    fontSize: 12,
    color: colors.text.secondary,
    lineHeight: 18,
  },
  // Full analytics button
  fullAnalyticsButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.s,
    backgroundColor: colors.primary,
    paddingVertical: spacing.m,
    borderRadius: 28,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 6,
  },
  fullAnalyticsText: {
    fontSize: 15,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  cardShadow: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.12,
    shadowRadius: 16,
    elevation: 6,
  },
  // Empty
  emptyState: {
    alignItems: 'center',
    paddingVertical: spacing.xxl,
    gap: spacing.m,
  },
  emptyTitle: {
    ...typography.h2,
    color: colors.text.secondary,
  },
  emptySubtext: {
    ...typography.body,
    color: colors.text.muted,
    textAlign: 'center',
    paddingHorizontal: spacing.xl,
  },
});

export default WeeklyInsightsScreen;
