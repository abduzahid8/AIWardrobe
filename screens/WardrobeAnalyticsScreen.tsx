import React, { useMemo } from 'react';
import { View, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import { ScaledText } from '../components/ui/ScaledText';
import { ScreenWrapper } from '../components/ui/ScreenWrapper';
import { LiquidGlassCard, PressableGlassCard } from '../components/ui/LiquidGlassCard';
import { QuickStat } from '../components/ui/QuickStat';
import { StatsCard } from '../components/ui/StatsCard';
import { CachedImage } from '../components/ui/CachedImage';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeInRight } from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing, borderRadius, shadows, typography } from '../src/theme';
import useWardrobeStore from '../store/wardrobeStore';
import { useSubscriptionGate } from '../src/hooks/useSubscriptionGate';
import FeatureLockOverlay from '../components/paywall/FeatureLockOverlay';
import { scoreDiversity, getColorDistribution, getCategoryBreakdown } from '../src/services/diversityEngine';
import type { ColorDistEntry, CategoryBreakdownEntry } from '../src/services/diversityEngine';
import { useTranslation } from 'react-i18next';

const CATEGORY_COLORS: Record<string, string> = {
  top: '#60A5FA',
  bottom: '#A78BFA',
  shoes: '#F472B6',
  outerwear: '#34D399',
  dress: '#FBBF24',
  accessory: '#FB923C',
  other: '#94A3B8',
};

const ColorSwatch = ({ color, name, count, total }: { color: string; name: string; count: number; total: number }) => {
  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
  return (
    <View style={styles.swatchContainer}>
      <View style={[styles.swatchCircle, { backgroundColor: color }]} />
      <ScaledText style={styles.swatchName} numberOfLines={1}>{name}</ScaledText>
      <ScaledText style={styles.swatchPct}>{pct}%</ScaledText>
    </View>
  );
};

const CategoryBar = ({ label, count, total, color }: { label: string; count: number; total: number; color: string }) => {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <View style={styles.categoryBarContainer}>
      <View style={styles.categoryBarLabelRow}>
        <ScaledText style={styles.categoryBarLabel}>{label}</ScaledText>
        <ScaledText style={styles.categoryBarCount}>{count}</ScaledText>
      </View>
      <View style={styles.categoryBarTrack}>
        <Animated.View
          entering={FadeInRight.duration(600).delay(200)}
          style={[styles.categoryBarFill, { width: `${Math.max(pct, 2)}%`, backgroundColor: color }]}
        />
      </View>
    </View>
  );
};

export default function WardrobeAnalyticsScreen({ navigation }: any) {
  const { canAccess } = useSubscriptionGate();
  const { t } = useTranslation();
  const hasAccess = canAccess('analytics');

  const items = useWardrobeStore((s) => s.items);
  const wearLogs = useWardrobeStore((s) => s.wearLogs);
  const streak = useWardrobeStore((s) => s.streak);

  const analytics = useMemo(() => {
    const utilization = items.length > 0
      ? (() => {
        const cutoff = new Date(Date.now() - 30 * 86400000).toISOString().split('T')[0];
        const recentLogs = wearLogs.filter((log) => log.date >= cutoff);
        const wornItemIds = new Set(recentLogs.flatMap((log) => log.itemIds));
        return Math.round((wornItemIds.size / items.length) * 100);
      })()
      : 0;

    const unwornItems = (() => {
      const cutoff = new Date(Date.now() - 30 * 86400000).toISOString().split('T')[0];
      const recentLogs = wearLogs.filter((log) => log.date >= cutoff);
      const wornItemIds = new Set(recentLogs.flatMap((log) => log.itemIds));
      return items.filter((item) => !wornItemIds.has(item.id));
    })();

    const diversity = scoreDiversity(items, wearLogs);
    const colorDist = getColorDistribution(items);
    const categoryBreakdown = getCategoryBreakdown(items);

    const mostWorn = [...items]
      .sort((a, b) => b.wearCount - a.wearCount)
      .filter(i => i.wearCount > 0)
      .slice(0, 5);

    const totalWears = wearLogs.length;
    const avgWears = items.length > 0
      ? Math.round((items.reduce((sum, i) => sum + i.wearCount, 0) / items.length) * 10) / 10
      : 0;
    const totalWearCounts = items.reduce((sum, i) => sum + i.wearCount, 0);

    const seasonBreakdown = (() => {
      const counts: Record<string, number> = {};
      items.forEach((item) => {
        if (item.seasons) {
          item.seasons.forEach((s) => {
            counts[s] = (counts[s] || 0) + 1;
          });
        }
      });
      return Object.entries(counts).sort((a, b) => b[1] - a[1]);
    })();

    const occasionBreakdown = (() => {
      const counts: Record<string, number> = {};
      items.forEach((item) => {
        if (item.occasions) {
          item.occasions.forEach((o) => {
            counts[o] = (counts[o] || 0) + 1;
          });
        }
      });
      return Object.entries(counts).sort((a, b) => b[1] - a[1]);
    })();

    const favoritesCount = items.filter((i) => i.isFavorite).length;
    const favoriteUtilization = favoritesCount > 0
      ? items.filter((i) => i.isFavorite && i.wearCount > 0).length / favoritesCount
      : 0;

    return {
      utilization, unwornItems, diversity, colorDist, categoryBreakdown,
      mostWorn, totalWears, avgWears, totalWearCounts,
      seasonBreakdown, occasionBreakdown, favoritesCount, favoriteUtilization,
    };
  }, [items, wearLogs]);

  if (!hasAccess) {
    return (
      <FeatureLockOverlay
        requiredTier="Pro"
        featureName={t('wardrobeAnalytics.featureName')}
        tagline={t('wardrobeAnalytics.tagline')}
        icon="bar-chart"
        bullets={[
          t('wardrobeAnalytics.bullet1'),
          t('wardrobeAnalytics.bullet2'),
          t('wardrobeAnalytics.bullet3'),
          t('wardrobeAnalytics.bullet4'),
        ]}
      />
    );
  }

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
          <ScaledText style={styles.headerTitle}>{t('analytics.title')}</ScaledText>
          <TouchableOpacity
            onPress={() => navigation.navigate('WeeklyInsights')}
            style={styles.headerAction}
          >
            <Ionicons name="calendar-outline" size={22} color={colors.primary} />
          </TouchableOpacity>
        </View>

        {/* Hero Utilization */}
        <Animated.View entering={FadeInDown.duration(500)} style={styles.heroSection}>
          <LiquidGlassCard variant="opaque" radius="xl">
            <StatsCard
              title={t('analytics.closetUtilization')}
              value={analytics.utilization}
              subtitle={`${analytics.totalWearCounts} total wears across ${items.length} items`}
              progress={analytics.utilization / 100}
              icon="pie-chart"
              trend={analytics.utilization >= 50 ? 'up' : analytics.utilization >= 20 ? 'neutral' : 'down'}
              trendValue={analytics.utilization >= 50 ? 'Good rotation' : 'Needs attention'}
              gradient
              gradientColors={['#0A1931', '#1E3A5F']}
            />
          </LiquidGlassCard>
        </Animated.View>

        {/* Quick Stats Row */}
        <Animated.View entering={FadeInDown.duration(500).delay(80)} style={styles.quickStatsRow}>
          <QuickStat icon="shirt-outline" value={items.length} label={t('analytics.totalItems')} color={colors.primary} index={0} />
          <QuickStat icon="flame-outline" value={streak} label={t('analytics.dayStreak')} color="#F97316" index={1} />
          <QuickStat icon="repeat-outline" value={analytics.totalWears} label={t('analytics.totalWears')} color="#8B5CF6" index={2} />
        </Animated.View>

        {/* Diversity Score */}
        <Animated.View entering={FadeInDown.duration(500).delay(120)} style={styles.sectionContainer}>
          <LiquidGlassCard variant="frosted" radius="xl">
            <View style={styles.cardHeaderRow}>
              <View style={styles.cardHeaderLeft}>
                <View style={[styles.iconBadge, { backgroundColor: '#3B82F615' }]}>
                  <Ionicons name="color-palette-outline" size={20} color="#3B82F6" />
                </View>
                <ScaledText style={styles.cardTitle}>{t('analytics.styleDiversity')}</ScaledText>
              </View>
              <View style={[styles.scoreBadge, {
                backgroundColor: analytics.diversity >= 70 ? '#DCFCE7' : analytics.diversity >= 40 ? '#FEF9C3' : '#FEE2E2',
              }]}>
                <ScaledText style={[styles.scoreBadgeText, {
                  color: analytics.diversity >= 70 ? '#166534' : analytics.diversity >= 40 ? '#854D0E' : '#991B1B',
                }]}>
                  {analytics.diversity}/100
                </ScaledText>
              </View>
            </View>
            <ScaledText style={styles.cardSubtext}>
              {analytics.diversity >= 70
                ? 'Great variety! You use your wardrobe well.'
                : analytics.diversity >= 40
                  ? 'Good start. Try mixing in some unworn items!'
                  : 'You tend to repeat the same items. Try the Surprise Me feature!'}
            </ScaledText>
            <View style={styles.diversityBar}>
              <Animated.View
                entering={FadeInRight.duration(800).delay(300)}
                style={[styles.diversityFill, { width: `${analytics.diversity}%` }]}
              />
            </View>
          </LiquidGlassCard>
        </Animated.View>

        {/* Most Worn Items */}
        {analytics.mostWorn.length > 0 && (
          <Animated.View entering={FadeInDown.duration(500).delay(160)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: '#10B98115' }]}>
                    <Ionicons name="trending-up-outline" size={20} color="#10B981" />
                  </View>
                  <ScaledText style={styles.cardTitle}>{t('analytics.mostWorn')}</ScaledText>
                </View>
              </View>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.horizontalScroll}>
                {analytics.mostWorn.map((item, idx) => (
                  <TouchableOpacity
                    key={item.id}
                    style={styles.mostWornItem}
                    onPress={() => navigation.navigate('ClothingDetail', { itemId: item.id })}
                  >
                    <View style={styles.mostWornRank}>
                      <ScaledText style={styles.mostWornRankText}>{idx + 1}</ScaledText>
                    </View>
                    <CachedImage
                      uri={item.imageUrl || item.thumbnailUrl || ''}
                      style={styles.mostWornImage}
                      contentFit="cover"
                      fadeIn={false}
                    />
                    <ScaledText style={styles.mostWornCount}>{item.wearCount}×</ScaledText>
                    <ScaledText style={styles.mostWornName} numberOfLines={1}>
                      {item.name || item.subCategory || item.category}
                    </ScaledText>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Color Palette */}
        {analytics.colorDist.length > 0 && (
          <Animated.View entering={FadeInDown.duration(500).delay(200)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: '#EC489915' }]}>
                    <Ionicons name="color-fill-outline" size={20} color="#EC4899" />
                  </View>
                  <ScaledText style={styles.cardTitle}>{t('analytics.colorPalette')}</ScaledText>
                </View>
              </View>
              <View style={styles.swatchRow}>
                {analytics.colorDist.slice(0, 8).map(({ color, name, count }: ColorDistEntry) => (
                  <ColorSwatch key={name} color={color} name={name} count={count} total={items.length} />
                ))}
              </View>
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Category Breakdown */}
        {analytics.categoryBreakdown.length > 0 && (
          <Animated.View entering={FadeInDown.duration(500).delay(240)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: '#8B5CF615' }]}>
                    <Ionicons name="grid-outline" size={20} color="#8B5CF6" />
                  </View>
                  <ScaledText style={styles.cardTitle}>{t('analytics.categoryBreakdown')}</ScaledText>
                </View>
              </View>
              {analytics.categoryBreakdown.map(({ category, count }: CategoryBreakdownEntry) => (
                <CategoryBar
                  key={category}
                  label={category.charAt(0).toUpperCase() + category.slice(1)}
                  count={count}
                  total={items.length}
                  color={CATEGORY_COLORS[category] || '#94A3B8'}
                />
              ))}
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Season Breakdown */}
        {analytics.seasonBreakdown.length > 0 && (
          <Animated.View entering={FadeInDown.duration(500).delay(280)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="frosted" radius="xl">
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: '#06B6D415' }]}>
                    <Ionicons name="thermometer-outline" size={20} color="#06B6D4" />
                  </View>
                  <ScaledText style={styles.cardTitle}>Season Breakdown</ScaledText>
                </View>
              </View>
              {analytics.seasonBreakdown.map(([season, count]) => {
                const pct = Math.round((count / items.length) * 100);
                const seasonColors: Record<string, string> = {
                  spring: '#34D399', summer: '#FBBF24', fall: '#FB923C', winter: '#60A5FA',
                };
                return (
                  <View key={season} style={styles.categoryBarContainer}>
                    <View style={styles.categoryBarLabelRow}>
                      <ScaledText style={styles.categoryBarLabel}>{season.charAt(0).toUpperCase() + season.slice(1)}</ScaledText>
                      <ScaledText style={styles.categoryBarCount}>{count}</ScaledText>
                    </View>
                    <View style={styles.categoryBarTrack}>
                      <Animated.View
                        entering={FadeInRight.duration(600).delay(200)}
                        style={[styles.categoryBarFill, { width: `${Math.max(pct, 2)}%`, backgroundColor: seasonColors[season] || '#94A3B8' }]}
                      />
                    </View>
                  </View>
                );
              })}
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Unworn Alert */}
        {analytics.unwornItems.length > 0 && (
          <Animated.View entering={FadeInDown.duration(500).delay(320)} style={styles.sectionContainer}>
            <LiquidGlassCard variant="light" radius="xl" style={styles.unwornCard}>
              <View style={styles.cardHeaderRow}>
                <View style={styles.cardHeaderLeft}>
                  <View style={[styles.iconBadge, { backgroundColor: '#F59E0B15' }]}>
                    <Ionicons name="alert-circle-outline" size={20} color="#F59E0B" />
                  </View>
                  <ScaledText style={[styles.cardTitle, { color: '#D97706' }]}>
                    {analytics.unwornItems.length} items unworn
                  </ScaledText>
                </View>
              </View>
              <ScaledText style={styles.unwornSubtext}>
                That's {Math.round((analytics.unwornItems.length / items.length) * 100)}% of your wardrobe sitting idle.
              </ScaledText>
              <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.horizontalScroll}>
                {analytics.unwornItems.slice(0, 10).map((item) => (
                  <TouchableOpacity
                    key={item.id}
                    onPress={() => navigation.navigate('ClothingDetail', { itemId: item.id })}
                  >
                    <CachedImage
                      uri={item.imageUrl || item.thumbnailUrl || ''}
                      style={styles.unwornImage}
                      contentFit="cover"
                      fadeIn={false}
                    />
                  </TouchableOpacity>
                ))}
              </ScrollView>
              <TouchableOpacity
                style={styles.unwornAction}
                onPress={() => navigation.navigate('WeeklyInsights')}
              >
                <ScaledText style={styles.unwornActionText}>View Weekly Insights</ScaledText>
                <Ionicons name="arrow-forward" size={16} color="#D97706" />
              </TouchableOpacity>
            </LiquidGlassCard>
          </Animated.View>
        )}

        {/* Quick Actions */}
        <Animated.View entering={FadeInDown.duration(500).delay(360)} style={styles.sectionContainer}>
          <LiquidGlassCard variant="frosted" radius="xl">
            <View style={styles.cardHeaderRow}>
              <ScaledText style={styles.cardTitle}>Quick Actions</ScaledText>
            </View>
            <View style={styles.quickActionsRow}>
              <TouchableOpacity
                style={styles.quickAction}
                onPress={() => navigation.navigate('MyCloset')}
              >
                <View style={[styles.quickActionIcon, { backgroundColor: '#3B82F615' }]}>
                  <Ionicons name="folder-open-outline" size={24} color="#3B82F6" />
                </View>
                <ScaledText style={styles.quickActionLabel}>My Closet</ScaledText>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.quickAction}
                onPress={() => navigation.navigate('Calendar')}
              >
                <View style={[styles.quickActionIcon, { backgroundColor: '#8B5CF615' }]}>
                  <Ionicons name="calendar-outline" size={24} color="#8B5CF6" />
                </View>
                <ScaledText style={styles.quickActionLabel}>Calendar</ScaledText>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.quickAction}
                onPress={() => navigation.navigate('WeeklyInsights')}
              >
                <View style={[styles.quickActionIcon, { backgroundColor: '#10B98115' }]}>
                  <Ionicons name="stats-chart-outline" size={24} color="#10B981" />
                </View>
                <ScaledText style={styles.quickActionLabel}>Weekly Insights</ScaledText>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.quickAction}
                onPress={() => navigation.navigate('AIOutfit')}
              >
                <View style={[styles.quickActionIcon, { backgroundColor: '#F9731615' }]}>
                  <Ionicons name="sparkles-outline" size={24} color="#F97316" />
                </View>
                <ScaledText style={styles.quickActionLabel}>AI Outfits</ScaledText>
              </TouchableOpacity>
            </View>
          </LiquidGlassCard>
        </Animated.View>

        {/* Empty State */}
        {items.length === 0 && (
          <View style={styles.emptyState}>
            <Ionicons name="analytics-outline" size={64} color={colors.text.muted} />
            <ScaledText style={styles.emptyTitle}>{t('analytics.noDataYet')}</ScaledText>
            <ScaledText style={styles.emptySubtext}>
              {t('analytics.emptySubtext')}
            </ScaledText>
          </View>
        )}

        <View style={{ height: 40 }} />
      </ScrollView>
    </ScreenWrapper>
  );
}

const styles = StyleSheet.create({
  backgroundOrbTop: {
    position: 'absolute',
    top: -100,
    right: -80,
    width: 280,
    height: 280,
    borderRadius: 140,
    backgroundColor: 'rgba(188, 210, 245, 0.42)',
  },
  backgroundOrbBottom: {
    position: 'absolute',
    left: -120,
    bottom: 140,
    width: 300,
    height: 300,
    borderRadius: 150,
    backgroundColor: 'rgba(216, 229, 252, 0.34)',
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
  heroSection: {
    marginBottom: spacing.m,
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
  scoreBadge: {
    paddingHorizontal: spacing.m,
    paddingVertical: spacing.xs,
    borderRadius: 99,
  },
  scoreBadgeText: {
    fontSize: 13,
    fontWeight: '700',
  },
  diversityBar: {
    height: 8,
    backgroundColor: 'rgba(0,0,0,0.08)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  diversityFill: {
    height: '100%',
    borderRadius: 4,
    backgroundColor: '#3B82F6',
  },
  horizontalScroll: {
    marginHorizontal: -spacing.s,
  },
  mostWornItem: {
    alignItems: 'center',
    marginHorizontal: spacing.s,
    width: 72,
  },
  mostWornRank: {
    position: 'absolute',
    top: 0,
    left: 0,
    zIndex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    borderRadius: 8,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  mostWornRankText: {
    fontSize: 10,
    color: '#FFF',
    fontWeight: '700',
  },
  mostWornImage: {
    width: 64,
    height: 64,
    borderRadius: 16,
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  mostWornCount: {
    fontSize: 12,
    color: colors.text.primary,
    fontWeight: '700',
    marginTop: 4,
  },
  mostWornName: {
    fontSize: 11,
    color: colors.text.muted,
    textAlign: 'center',
  },
  swatchRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.m,
  },
  swatchContainer: {
    alignItems: 'center',
    width: 52,
  },
  swatchCircle: {
    width: 32,
    height: 32,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.5)',
    marginBottom: 4,
  },
  swatchName: {
    fontSize: 10,
    color: colors.text.secondary,
    textAlign: 'center',
  },
  swatchPct: {
    fontSize: 10,
    color: colors.text.muted,
  },
  categoryBarContainer: {
    marginBottom: spacing.s,
  },
  categoryBarLabelRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  categoryBarLabel: {
    fontSize: 13,
    color: colors.text.secondary,
    fontWeight: '500',
  },
  categoryBarCount: {
    fontSize: 13,
    color: colors.text.primary,
    fontWeight: '600',
  },
  categoryBarTrack: {
    height: 8,
    backgroundColor: 'rgba(0,0,0,0.08)',
    borderRadius: 4,
    overflow: 'hidden',
  },
  categoryBarFill: {
    height: '100%',
    borderRadius: 4,
  },
  unwornCard: {
    borderWidth: 1,
    borderColor: 'rgba(245, 158, 11, 0.25)',
  },
  unwornSubtext: {
    ...typography.bodySmall,
    color: colors.text.secondary,
    marginBottom: spacing.s,
  },
  unwornImage: {
    width: 56,
    height: 56,
    borderRadius: 14,
    marginHorizontal: spacing.xs,
    backgroundColor: 'rgba(0,0,0,0.05)',
  },
  unwornAction: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: spacing.xs,
    marginTop: spacing.s,
    paddingVertical: spacing.s,
    borderRadius: 20,
    backgroundColor: 'rgba(245, 158, 11, 0.1)',
  },
  unwornActionText: {
    fontSize: 13,
    fontWeight: '600',
    color: '#D97706',
  },
  quickActionsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.s,
    marginTop: spacing.xs,
  },
  quickAction: {
    flex: 1,
    minWidth: 70,
    alignItems: 'center',
    gap: spacing.xs,
    paddingVertical: spacing.s,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.5)',
  },
  quickActionIcon: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  quickActionLabel: {
    fontSize: 10,
    fontWeight: '600',
    color: colors.text.secondary,
    textAlign: 'center',
  },
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
