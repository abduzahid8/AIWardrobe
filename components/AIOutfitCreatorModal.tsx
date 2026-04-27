/**
 * AIOutfitCreatorModal — Bottom sheet for AI-powered outfit creation
 * Lets user pick an event/occasion, then generates an outfit from their wardrobe.
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  Image,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
} from 'react-native';
import { Ionicons, MaterialCommunityIcons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import Animated, {
  FadeIn,
  FadeInDown,
  FadeInUp,
  SlideInRight,
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import * as Haptics from 'expo-haptics';

import { BottomSheet } from './ui/BottomSheet';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import { generateSuggestions, ScoredOutfit } from '../src/services/suggestionEngine';
import useWardrobeStore from '../store/wardrobeStore';
import type { Occasion } from '../src/types/domain';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

// ─── Event definitions with icons ────────────────────────────
interface EventOption {
  id: string;
  occasion: Occasion;
  label: string;
  icon: string;
  iconFamily: 'ionicons' | 'material';
  gradient: [string, string];
}

// ─── Component ───────────────────────────────────────────────

interface AIOutfitCreatorModalProps {
  visible: boolean;
  onClose: () => void;
}

const AIOutfitCreatorModal: React.FC<AIOutfitCreatorModalProps> = ({
  visible,
  onClose,
}) => {
  const { t } = useTranslation();
  const items = useWardrobeStore((s) => s.items);
  const wearLogs = useWardrobeStore((s) => s.wearLogs);

  const EVENT_OPTIONS: EventOption[] = [
    {
      id: 'business-meeting',
      occasion: 'work',
      label: t('aiOutfitCreator.events.businessMeeting'),
      icon: 'briefcase-outline',
      iconFamily: 'ionicons',
      gradient: ['#1a1a2e', '#16213e'],
    },
    {
      id: 'alpine-skiing',
      occasion: 'sport',
      label: t('aiOutfitCreator.events.alpineSkiing'),
      icon: 'snow-outline',
      iconFamily: 'ionicons',
      gradient: ['#0f4c75', '#3282b8'],
    },
    {
      id: 'casual',
      occasion: 'casual',
      label: t('aiOutfitCreator.events.casualDay'),
      icon: 'sunny-outline',
      iconFamily: 'ionicons',
      gradient: ['#f39c12', '#e67e22'],
    },
    {
      id: 'date',
      occasion: 'date',
      label: t('aiOutfitCreator.events.dateNight'),
      icon: 'heart-outline',
      iconFamily: 'ionicons',
      gradient: ['#c0392b', '#e74c3c'],
    },
    {
      id: 'workout',
      occasion: 'sport',
      label: t('aiOutfitCreator.events.workout'),
      icon: 'fitness-outline',
      iconFamily: 'ionicons',
      gradient: ['#27ae60', '#2ecc71'],
    },
    {
      id: 'beach',
      occasion: 'travel',
      label: t('aiOutfitCreator.events.beachDay'),
      icon: 'water-outline',
      iconFamily: 'ionicons',
      gradient: ['#00b4d8', '#48cae4'],
    },
    {
      id: 'wedding',
      occasion: 'formal',
      label: t('aiOutfitCreator.events.wedding'),
      icon: 'sparkles-outline',
      iconFamily: 'ionicons',
      gradient: ['#8e44ad', '#9b59b6'],
    },
    {
      id: 'interview',
      occasion: 'work',
      label: t('aiOutfitCreator.events.jobInterview'),
      icon: 'person-outline',
      iconFamily: 'ionicons',
      gradient: ['#2c3e50', '#34495e'],
    },
    {
      id: 'night-out',
      occasion: 'date',
      label: t('aiOutfitCreator.events.nightOut'),
      icon: 'moon-outline',
      iconFamily: 'ionicons',
      gradient: ['#6c5ce7', '#a29bfe'],
    },
    {
      id: 'travel',
      occasion: 'travel',
      label: t('aiOutfitCreator.events.travel'),
      icon: 'airplane-outline',
      iconFamily: 'ionicons',
      gradient: ['#00cec9', '#55efc4'],
    },
  ];

  const [selectedEvent, setSelectedEvent] = useState<Occasion | null>(null);
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<ScoredOutfit | null>(null);
  const [resultIndex, setResultIndex] = useState(0);
  const [allResults, setAllResults] = useState<ScoredOutfit[]>([]);

  const hasItems = items.length >= 3;

  const handleSelectEvent = useCallback((eventId: Occasion) => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    setSelectedEvent(eventId);
    setResult(null);
    setResultIndex(0);
    setAllResults([]);
  }, []);

  const handleGenerate = useCallback(() => {
    if (!selectedEvent || !hasItems) return;

    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
    setGenerating(true);

    // Small delay for animation feel
    setTimeout(() => {
      const suggestions = generateSuggestions({
        items,
        wearLogs,
        occasion: selectedEvent,
      });

      setAllResults(suggestions);
      setResult(suggestions[0] || null);
      setResultIndex(0);
      setGenerating(false);
    }, 600);
  }, [selectedEvent, items, wearLogs, hasItems]);

  const handleTryAnother = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    const nextIndex = (resultIndex + 1) % allResults.length;
    setResultIndex(nextIndex);
    setResult(allResults[nextIndex] || null);
  }, [resultIndex, allResults]);

  const handleClose = useCallback(() => {
    setSelectedEvent(null);
    setResult(null);
    setResultIndex(0);
    setAllResults([]);
    onClose();
  }, [onClose]);

  // ─── Render ──────────────────────────────────────

  const renderEventChips = () => (
    <View style={styles.chipsSection}>
      <Text style={styles.sectionLabel}>{t('aiOutfitCreator.chooseEvent')}</Text>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.chipsScroll}
      >
        {EVENT_OPTIONS.map((event, index) => {
          const isSelected = selectedEvent === event.occasion;
          return (
            <Animated.View
              key={event.id}
              entering={FadeInDown.delay(index * 40).duration(300)}
            >
              <TouchableOpacity
                style={[
                  styles.eventChip,
                  isSelected && styles.eventChipSelected,
                  isSelected && { backgroundColor: event.gradient[0] },
                ]}
                onPress={() => handleSelectEvent(event.occasion)}
                activeOpacity={0.7}
                accessibilityLabel={t('aiOutfitCreator.selectEvent', { event: event.label })}
                accessibilityRole="button"
                accessibilityState={{ selected: isSelected }}
              >
                <View style={[
                  styles.chipIconContainer,
                  isSelected && { backgroundColor: 'rgba(255,255,255,0.2)' },
                ]}>
                  <Ionicons
                    name={event.icon as any}
                    size={18}
                    color={isSelected ? '#FFF' : colors.text.secondary}
                  />
                </View>
                <Text
                  style={[
                    styles.chipLabel,
                    isSelected && styles.chipLabelSelected,
                  ]}
                  numberOfLines={1}
                >
                  {event.label}
                </Text>
              </TouchableOpacity>
            </Animated.View>
          );
        })}
      </ScrollView>
    </View>
  );

  const renderGenerateButton = () => (
    <Animated.View entering={FadeInUp.delay(200).duration(300)}>
      <TouchableOpacity
        style={[
          styles.generateButton,
          (!selectedEvent || !hasItems) && styles.generateButtonDisabled,
        ]}
        onPress={handleGenerate}
        disabled={!selectedEvent || !hasItems || generating}
        activeOpacity={0.8}
        accessibilityLabel={t('aiOutfitCreator.generateOutfitWithAI')}
        accessibilityRole="button"
      >
        {generating ? (
          <ActivityIndicator size="small" color="#FFF" />
        ) : (
          <>
            <Ionicons name="sparkles" size={18} color="#FFF" />
            <Text style={styles.generateButtonText}>
              {t('aiOutfitCreator.generateWithAI')}
            </Text>
          </>
        )}
      </TouchableOpacity>
    </Animated.View>
  );

  const renderResult = () => {
    if (!result) return null;

    const outfitItems = result.outfit.itemIds
      .map((id) => items.find((i) => i.id === id))
      .filter(Boolean);

    return (
      <Animated.View entering={FadeIn.duration(400)} style={styles.resultSection}>
        {/* Outfit Grid */}
        <View style={styles.outfitGrid}>
          {outfitItems.map((item, index) => (
            <Animated.View
              key={item!.id}
              entering={SlideInRight.delay(index * 80).duration(300)}
              style={styles.outfitItemCard}
            >
              <Image
                source={{ uri: item!.imageUrl }}
                style={styles.outfitItemImage as any}
                resizeMode="cover"
              />
              <Text style={styles.outfitItemLabel} numberOfLines={1}>
                {item!.subCategory || item!.category}
              </Text>
            </Animated.View>
          ))}
        </View>

        {/* Reasoning */}
        <View style={styles.reasoningContainer}>
          <Ionicons name="bulb-outline" size={16} color={colors.text.secondary} />
          <Text style={styles.reasoningText}>
            {result.reasoning}
          </Text>
        </View>

        {/* Actions */}
        <View style={styles.resultActions}>
          {allResults.length > 1 && (
            <TouchableOpacity
              style={styles.tryAnotherButton}
              onPress={handleTryAnother}
              activeOpacity={0.7}
            >
              <Ionicons name="refresh" size={18} color={colors.text.primary} />
              <Text style={styles.tryAnotherText}>{t('aiOutfitCreator.tryAnother')}</Text>
            </TouchableOpacity>
          )}
          <TouchableOpacity
            style={styles.saveButton}
            onPress={handleClose}
            activeOpacity={0.8}
          >
            <Ionicons name="checkmark" size={18} color="#FFF" />
            <Text style={styles.saveButtonText}>{t('aiOutfitCreator.looksGood')}</Text>
          </TouchableOpacity>
        </View>
      </Animated.View>
    );
  };

  const renderEmptyState = () => (
    <Animated.View entering={FadeIn.duration(300)} style={styles.emptyState}>
      <Ionicons name="shirt-outline" size={40} color={colors.text.tertiary} />
      <Text style={styles.emptyTitle}>{t('aiOutfitCreator.addMoreItemsFirst')}</Text>
      <Text style={styles.emptySubtitle}>
        {t('aiOutfitCreator.needAtLeastThreeItems')}
      </Text>
    </Animated.View>
  );

  return (
    <BottomSheet
      visible={visible}
      onClose={handleClose}
      snapPoint={0.65}
      enableBlur
      style={styles.sheet}
    >
      {/* Title */}
      <View style={styles.titleRow}>
        <View>
          <Text style={styles.title}>{t('aiOutfitCreator.title')}</Text>
          <Text style={styles.subtitle}>
            {t('aiOutfitCreator.subtitle')}
          </Text>
        </View>
        <View style={styles.aiBadge}>
          <Ionicons name="sparkles" size={14} color="#FFF" />
        </View>
      </View>

      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        {/* Event Chips */}
        {renderEventChips()}

        {/* Generate or Empty State */}
        {hasItems ? (
          <>
            {!result && renderGenerateButton()}
            {renderResult()}
          </>
        ) : (
          renderEmptyState()
        )}
      </ScrollView>
    </BottomSheet>
  );
};

// ─── Styles ──────────────────────────────────────────────────

const styles = StyleSheet.create({
  sheet: {
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
  },
  titleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.lg,
  },
  title: {
    ...typography.scale.titleLarge,
    color: colors.text.primary,
    fontWeight: '700',
  },
  subtitle: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
    marginTop: 2,
  },
  aiBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#0A1931',
    alignItems: 'center',
    justifyContent: 'center',
  },
  scrollContent: {
    paddingBottom: 40,
  },

  // ── Chips ───────────
  chipsSection: {
    marginBottom: spacing.lg,
  },
  sectionLabel: {
    ...typography.scale.labelSmall,
    color: colors.text.tertiary,
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: spacing.sm,
  },
  chipsScroll: {
    gap: 10,
    paddingRight: spacing.md,
  },
  eventChip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 20,
    backgroundColor: colors.background.secondary,
    borderWidth: 1.5,
    borderColor: colors.background.tertiary || '#E8E8E8',
    gap: 8,
  },
  eventChipSelected: {
    borderColor: 'transparent',
  },
  chipIconContainer: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.background.tertiary || '#F0F0F0',
    alignItems: 'center',
    justifyContent: 'center',
  },
  chipLabel: {
    ...typography.scale.bodySmall,
    color: colors.text.primary,
    fontWeight: '600',
  },
  chipLabelSelected: {
    color: '#FFF',
  },

  // ── Generate Button ─
  generateButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0A1931',
    paddingVertical: 14,
    borderRadius: radius.pill,
    gap: 8,
    marginBottom: spacing.lg,
  },
  generateButtonDisabled: {
    opacity: 0.4,
  },
  generateButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },

  // ── Result ──────────
  resultSection: {
    marginTop: spacing.sm,
  },
  outfitGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    justifyContent: 'center',
  },
  outfitItemCard: {
    width: (SCREEN_WIDTH - 100) / 3,
    backgroundColor: '#F7F7F7',
    borderRadius: radius.lg,
    overflow: 'hidden',
    alignItems: 'center',
  },
  outfitItemImage: {
    width: '100%',
    aspectRatio: 0.85,
  },
  outfitItemLabel: {
    ...typography.scale.labelSmall,
    color: colors.text.secondary,
    paddingVertical: 6,
    textTransform: 'capitalize',
  },
  reasoningContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginTop: spacing.md,
    paddingHorizontal: spacing.sm,
  },
  reasoningText: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
    flex: 1,
    lineHeight: 18,
  },

  // ── Actions ─────────
  resultActions: {
    flexDirection: 'row',
    gap: 10,
    marginTop: spacing.lg,
  },
  tryAnotherButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: radius.pill,
    backgroundColor: colors.background.secondary,
    gap: 6,
  },
  tryAnotherText: {
    ...typography.scale.bodySmall,
    color: colors.text.primary,
    fontWeight: '600',
  },
  saveButton: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: radius.pill,
    backgroundColor: '#0A1931',
    gap: 6,
  },
  saveButtonText: {
    ...typography.scale.bodySmall,
    color: '#FFF',
    fontWeight: '600',
  },

  // ── Empty State ─────
  emptyState: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing.xxl,
    gap: spacing.sm,
  },
  emptyTitle: {
    ...typography.scale.titleSmall,
    color: colors.text.primary,
    fontWeight: '600',
  },
  emptySubtitle: {
    ...typography.scale.bodySmall,
    color: colors.text.secondary,
    textAlign: 'center',
    maxWidth: 220,
  },
});

export default AIOutfitCreatorModal;
