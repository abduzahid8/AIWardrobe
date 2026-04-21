import React from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Image,
  ActivityIndicator,
  StyleSheet,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import LiquidGlass2026Theme, { SpatialElevation } from '../../../constants/LiquidGlass2026Theme';
import QuotaBadge from '../../../components/paywall/QuotaBadge';
import { BASIC_CLOTHING_ITEMS } from '../../../data/basicClothingItems';
import OutfitCollageDisplay from './OutfitCollageDisplay';
import type { GeneratedOutfit, OutfitItem, OutfitVisual } from '../types';
import { getOutfitItemMacroCategory } from '../utils/outfitPreview';

function resolveClothingImage(src: string | number | undefined | null): string | number | null {
  if (src === null || src === undefined) return null;
  if (typeof src === 'number') return src;
  if (typeof src === 'string' && src.startsWith('basic_clothing_')) {
    const id = src.replace('basic_clothing_', '');
    const found = BASIC_CLOTHING_ITEMS.find(b => b.id === id);
    return found ? found.image : null;
  }
  return src.length > 0 ? src : null;
}

interface OutfitResultsViewProps {
  outfits: GeneratedOutfit[];
  outfitVisuals: Record<string, OutfitVisual>;
  calendarDate?: string;
  onBack: () => void;
  onSave: (outfit: GeneratedOutfit) => void;
  onAddToCalendar: (outfit: GeneratedOutfit) => void;
  onTryOn: (outfit: GeneratedOutfit) => void;
  onRetryVisual: (outfit: GeneratedOutfit) => void;
}

const OutfitResultsView: React.FC<OutfitResultsViewProps> = ({
  outfits,
  outfitVisuals,
  calendarDate,
  onBack,
  onSave,
  onAddToCalendar,
  onTryOn,
  onRetryVisual,
}) => {
  const getCategoryLabel = (item: OutfitItem) => {
    const cat = getOutfitItemMacroCategory(item);
    if (cat === 'top') return 'Top';
    if (cat === 'outerwear') return 'Outerwear';
    if (cat === 'bottom') return 'Bottom';
    if (cat === 'shoes') return 'Shoes';
    if (cat === 'accessory') return 'Accessory';
    return item.type || item.name || 'Item';
  };

  return (
    <View style={styles.container}>
      <SafeAreaView style={{ flex: 1 }}>
        <View style={styles.header}>
          <TouchableOpacity onPress={onBack} style={styles.backButton}>
            <Ionicons name="chevron-back" size={28} color="#1a1a1a" />
          </TouchableOpacity>
          <Text style={styles.headerTitle}>Your Outfits</Text>
          <View style={{ width: 40 }} />
        </View>

        <View style={styles.quotaBadgeWrap}>
          <QuotaBadge feature="aiOutfits" label="outfits" />
        </View>

        <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 20 }}>
          {outfits.map((outfit) => (
            <View key={outfit.id} style={styles.outfitCard}>
              {/* Visual area */}
              {outfitVisuals[outfit.id]?.loading ? (
                <View style={[styles.outfitCollage, { alignItems: 'center', justifyContent: 'center', backgroundColor: '#F0F4FF' }]}>
                  <ActivityIndicator color="#0A1931" size="large" />
                  <Text style={{ marginTop: 14, color: '#0A1931', fontSize: 13, fontWeight: '600' }}>✨ AI rendering outfit...</Text>
                  <Text style={{ marginTop: 4, color: '#6B7280', fontSize: 11 }}>Powered by NVIDIA Flux.1-Kontext</Text>
                </View>
              ) : outfitVisuals[outfit.id]?.image ? (
                <Image
                  source={{ uri: outfitVisuals[outfit.id]!.image! }}
                  style={styles.outfitImage}
                  resizeMode="cover"
                />
              ) : (
                <OutfitCollageDisplay items={outfit.items} />
              )}

              {/* AI badge */}
              {outfitVisuals[outfit.id]?.image && !outfitVisuals[outfit.id]?.loading && (
                <View style={styles.aiGeneratedBadge}>
                  <Ionicons name="sparkles" size={11} color="#FFF" />
                  <Text style={styles.aiGeneratedBadgeText}>AI Generated</Text>
                </View>
              )}

              {/* Retry button */}
              {!outfitVisuals[outfit.id]?.loading && !outfitVisuals[outfit.id]?.image && outfit.items.some(i => typeof i.image === 'string') && (
                <TouchableOpacity
                  style={styles.retryAiBadge}
                  onPress={() => onRetryVisual(outfit)}
                  activeOpacity={0.8}
                >
                  <Ionicons name="sparkles-outline" size={12} color="#0A1931" />
                  <Text style={styles.retryAiBadgeText}>Generate AI Preview</Text>
                </TouchableOpacity>
              )}

              {/* Match badge */}
              <View style={styles.matchBadge}>
                <Text style={styles.matchBadgeText}>{Math.round((outfit.matchScore || 0.78) * 100)}% Match</Text>
              </View>

              {/* Details area */}
              <View style={{ padding: 20 }}>
                <Text style={styles.outfitDesc}>{outfit.description}</Text>
                <Text style={styles.itemsLabel}>Items in this outfit:</Text>

                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  contentContainerStyle={{ paddingHorizontal: 2, paddingVertical: 4, marginBottom: 16 }}
                >
                  {outfit.items.length === 0 && (
                    <Text style={{ color: '#999', fontStyle: 'italic' }}>No items available</Text>
                  )}
                  {outfit.items.map((item: OutfitItem, idx: number) => (
                    <View key={String(item.id || idx)} style={styles.individualItemCard}>
                      {(() => {
                        const resolved = resolveClothingImage(item.image);
                        return resolved !== null ? (
                          <Image
                            source={typeof resolved === 'number' ? resolved : { uri: resolved }}
                            style={styles.individualItemImage}
                            resizeMode="contain"
                          />
                        ) : (
                          <View style={[styles.individualItemImage, { backgroundColor: '#f0f0f0', alignItems: 'center', justifyContent: 'center' }]}>
                            <Ionicons name="shirt-outline" size={40} color="#ccc" />
                          </View>
                        );
                      })()}
                      <Text style={styles.individualItemType} numberOfLines={1}>
                        {getCategoryLabel(item)}
                      </Text>
                      {item.isShopItem && item.price && (
                        <Text style={{ fontSize: 11, color: '#0A1931', fontWeight: '600' }}>
                          ${item.price}
                        </Text>
                      )}
                      {item.color ? (
                        <View style={styles.individualItemColorRow}>
                          <View style={[styles.individualItemColorDot, { backgroundColor: item.color }]} />
                          <Text style={styles.individualItemColorText} numberOfLines={1}>{item.color}</Text>
                        </View>
                      ) : null}
                    </View>
                  ))}
                </ScrollView>

                {/* Styling tips */}
                <View style={styles.stylingTipsBox}>
                  <Text style={styles.stylingTipsText}>
                    💡 <Text style={{ fontWeight: '600' }}>Styling Tip:</Text>{' '}
                    {Array.isArray(outfit.stylingTips) ? outfit.stylingTips[0] : outfit.stylingTips}
                  </Text>
                </View>

                {/* Action buttons */}
                <TouchableOpacity
                  style={styles.wishlistButton}
                  onPress={() => onSave(outfit)}
                  activeOpacity={0.8}
                >
                  <Ionicons name="bookmark-outline" size={20} color={LiquidGlass2026Theme.colors.text.primary} />
                  <Text style={styles.wishlistButtonText}>Save outfit</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={styles.wishlistButton}
                  onPress={() => onTryOn(outfit)}
                  activeOpacity={0.8}
                >
                  <Ionicons name="sparkles-outline" size={20} color={LiquidGlass2026Theme.colors.text.primary} />
                  <Text style={styles.wishlistButtonText}>Validate in try-on</Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.wishlistButton, styles.calendarButton]}
                  onPress={() => onAddToCalendar(outfit)}
                  activeOpacity={0.8}
                >
                  <Ionicons name="calendar-outline" size={20} color="#FFFFFF" />
                  <Text style={[styles.wishlistButtonText, { color: '#FFFFFF' }]}>
                    {calendarDate ? `Add to ${calendarDate}` : 'Add to Calendar'}
                  </Text>
                </TouchableOpacity>
              </View>
            </View>
          ))}
        </ScrollView>
      </SafeAreaView>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: LiquidGlass2026Theme.colors.background.primary,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 12,
  },
  backButton: {
    marginRight: 16,
    width: 44,
    height: 44,
    borderRadius: 22,
    backgroundColor: 'rgba(255,255,255,0.7)',
    alignItems: 'center',
    justifyContent: 'center',
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  headerTitle: {
    ...LiquidGlass2026Theme.typography.scale.headlineMedium,
    color: LiquidGlass2026Theme.colors.text.primary,
    flex: 1,
  },
  quotaBadgeWrap: {
    paddingHorizontal: 20,
    paddingBottom: 8,
  },
  outfitCard: {
    marginBottom: 32,
    backgroundColor: LiquidGlass2026Theme.colors.background.elevated,
    borderRadius: LiquidGlass2026Theme.radius.card,
    overflow: 'hidden',
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },
  outfitImage: {
    width: '100%',
    height: 400,
  },
  outfitCollage: {
    width: '100%',
    height: 400,
    backgroundColor: '#F7F8FA',
    overflow: 'hidden',
    flexDirection: 'row',
  },
  matchBadge: {
    position: 'absolute',
    top: 16,
    right: 16,
    backgroundColor: 'rgba(0,0,0,0.7)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  matchBadgeText: {
    color: LiquidGlass2026Theme.colors.text.onDark,
    ...LiquidGlass2026Theme.typography.scale.labelMedium,
    fontWeight: '700',
  },
  outfitDesc: {
    ...LiquidGlass2026Theme.typography.scale.bodyLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    marginBottom: 16,
  },
  itemsLabel: {
    ...LiquidGlass2026Theme.typography.scale.titleSmall,
    color: LiquidGlass2026Theme.colors.text.secondary,
    marginBottom: 12,
  },
  individualItemCard: {
    width: 120,
    marginRight: 14,
    backgroundColor: '#FFFFFF',
    borderRadius: 18,
    overflow: 'hidden',
    alignItems: 'center',
    paddingBottom: 12,
    ...SpatialElevation.getShadow(SpatialElevation.levels.card),
  },
  individualItemImage: {
    width: 120,
    height: 120,
    borderRadius: 18,
  },
  individualItemType: {
    fontSize: 12,
    color: '#1a1a2e',
    fontWeight: '700',
    marginTop: 8,
    paddingHorizontal: 8,
    textAlign: 'center',
    textTransform: 'capitalize',
    letterSpacing: 0.1,
  },
  individualItemColorRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 4,
    gap: 5,
  },
  individualItemColorDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.1)',
  },
  individualItemColorText: {
    fontSize: 11,
    color: '#6B7280',
    textTransform: 'capitalize',
    maxWidth: 80,
  },
  stylingTipsBox: {
    backgroundColor: 'rgba(255,255,255,0.7)',
    padding: 16,
    borderRadius: LiquidGlass2026Theme.radius.md,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.glass,
    marginTop: 8,
  },
  stylingTipsText: {
    ...LiquidGlass2026Theme.typography.scale.bodyMedium,
    color: LiquidGlass2026Theme.colors.text.secondary,
  },
  wishlistButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 16,
    paddingVertical: 14,
    borderRadius: LiquidGlass2026Theme.radius.pill,
    borderWidth: 1,
    borderColor: LiquidGlass2026Theme.colors.border.subtle,
    backgroundColor: 'rgba(255,255,255,0.5)',
    gap: 8,
  },
  wishlistButtonText: {
    ...LiquidGlass2026Theme.typography.scale.labelLarge,
    color: LiquidGlass2026Theme.colors.text.primary,
    fontWeight: '600',
  },
  calendarButton: {
    backgroundColor: '#0F172A',
    borderColor: '#0F172A',
  },
  aiGeneratedBadge: {
    position: 'absolute',
    top: 16,
    left: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(10,25,49,0.82)',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    gap: 5,
  },
  aiGeneratedBadgeText: {
    color: '#FFF',
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 0.2,
  },
  retryAiBadge: {
    position: 'absolute',
    bottom: 16,
    right: 16,
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255,255,255,0.92)',
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 20,
    gap: 5,
    borderWidth: 1,
    borderColor: 'rgba(10,25,49,0.15)',
    ...SpatialElevation.getShadow(SpatialElevation.levels.raised),
  },
  retryAiBadgeText: {
    color: '#0A1931',
    fontSize: 11,
    fontWeight: '700',
  },
});

export default OutfitResultsView;
