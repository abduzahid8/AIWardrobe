/**
 * ClothingDetailScreen — View & edit an existing wardrobe item.
 * Navigated to from MyClosetScreen and OutfitInspoScreen with
 * { itemId, fullItem? } route params.
 */

import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  Image,
  Alert,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';

import useWardrobeStore from '../store/wardrobeStore';
import { wardrobeApi } from '../src/lib/api';
import type { ClothingItem, Season, Occasion } from '../src/types/domain';
import type { RootStackParamList } from '../navigation/types';

// ── Constants ──────────────────────────────────────────────────────────────

const TYPES = [
  { id: 'tops',        label: 'Tops' },
  { id: 'bottoms',     label: 'Bottoms' },
  { id: 'shoes',       label: 'Shoes' },
  { id: 'accessories', label: 'Accessories' },
  { id: 'outerwear',   label: 'Outerwear' },
  { id: 'sportswear',  label: 'Sportswear' },
  { id: 'homewear',    label: 'Homewear' },
  { id: 'other',       label: 'Other' },
];

const COLORS = [
  { id: 'black',      label: 'Black',      hex: '#1C1C1E' },
  { id: 'white',      label: 'White',      hex: '#FFFFFF' },
  { id: 'grey',       label: 'Grey',       hex: '#8E8E93' },
  { id: 'beige',      label: 'Beige',      hex: '#C7B299' },
  { id: 'cream',      label: 'Cream',      hex: '#FFFDD0' },
  { id: 'brown',      label: 'Brown',      hex: '#8B4513' },
  { id: 'red',        label: 'Red',        hex: '#FF3B30' },
  { id: 'pink',       label: 'Pink',       hex: '#FF6B9D' },
  { id: 'orange',     label: 'Orange',     hex: '#FF9500' },
  { id: 'yellow',     label: 'Yellow',     hex: '#FFCC00' },
  { id: 'green',      label: 'Green',      hex: '#34C759' },
  { id: 'teal',       label: 'Teal',       hex: '#5AC8FA' },
  { id: 'blue',       label: 'Blue',       hex: '#007AFF' },
  { id: 'navy',       label: 'Navy',       hex: '#1B2A4A' },
  { id: 'purple',     label: 'Purple',     hex: '#AF52DE' },
  { id: 'multicolor', label: 'Multi',      hex: '#FF6B6B' },
];

const SEASONS: { id: Season; label: string }[] = [
  { id: 'spring', label: 'Spring' },
  { id: 'summer', label: 'Summer' },
  { id: 'fall',   label: 'Autumn' },
  { id: 'winter', label: 'Winter' },
];

const OCCASIONS: { id: Occasion; label: string; icon: string }[] = [
  { id: 'casual', label: 'Casual',  icon: 'cafe-outline' },
  { id: 'work',   label: 'Work',    icon: 'briefcase-outline' },
  { id: 'formal', label: 'Formal',  icon: 'ribbon-outline' },
  { id: 'sport',  label: 'Sport',   icon: 'fitness-outline' },
  { id: 'date',   label: 'Date',    icon: 'heart-outline' },
  { id: 'travel', label: 'Travel',  icon: 'airplane-outline' },
];

// ── Helpers ────────────────────────────────────────────────────────────────

/** Map a primaryColor label or hex to a COLORS entry id */
const resolveColorId = (item: ClothingItem): string => {
  const label = (item.primaryColor || '').toLowerCase();
  const hex   = (item.colorHex || '').toLowerCase();
  const match = COLORS.find(
    c => c.id === label || c.label.toLowerCase() === label || c.hex.toLowerCase() === hex
  );
  return match?.id ?? 'grey';
};

/** Map a ClothingCategory to a TYPES entry id */
const resolveCategoryId = (item: ClothingItem): string => {
  const cat = (item.subCategory || item.category || '').toLowerCase();
  const match = TYPES.find(t => t.id === cat || cat.includes(t.id));
  return match?.id ?? 'other';
};

// ── Component ──────────────────────────────────────────────────────────────

type RouteProps = RouteProp<RootStackParamList, 'ClothingDetail'>;

const ClothingDetailScreen: React.FC = () => {
  const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
  const route      = useRoute<RouteProps>();
  const { t }      = useTranslation();

  const { itemId, fullItem: routeItem } = route.params ?? {};

  // Prefer store item so edits from other screens are reflected
  const storeItem = useWardrobeStore(s => s.items.find(i => i.id === itemId));
  const item: ClothingItem | undefined = storeItem ?? (routeItem as ClothingItem | undefined);

  const updateItemInStore = useWardrobeStore(s => s.updateItem);
  const removeItemFromStore = useWardrobeStore(s => s.removeItem);
  const toggleFavoriteInStore = useWardrobeStore(s => s.toggleFavorite);

  // ── Edit state ────────────────────────────────────────────────────────
  const [isEditing,  setIsEditing]  = useState(false);
  const [isSaving,   setIsSaving]   = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const [selectedType,     setSelectedType]     = useState(() => item ? resolveCategoryId(item) : 'other');
  const [selectedColor,    setSelectedColor]     = useState(() => item ? resolveColorId(item) : 'grey');
  const [selectedSeasons,  setSelectedSeasons]   = useState<Season[]>(() => item?.seasons ?? []);
  const [selectedOccasions, setSelectedOccasions] = useState<Occasion[]>(() => item?.occasions ?? []);

  // ── Guard: item not found ─────────────────────────────────────────────
  if (!item) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.centered}>
          <Ionicons name="alert-circle-outline" size={48} color="#8E8E93" />
          <Text style={styles.notFoundText}>{t('clothingDetail.notFound')}</Text>
          <TouchableOpacity style={styles.backButton} onPress={() => navigation.goBack()}>
            <Text style={styles.backButtonText}>{t('common.goBack')}</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  // ── Handlers ──────────────────────────────────────────────────────────

  const handleFavorite = useCallback(() => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    toggleFavoriteInStore(item.id);
  }, [item.id, toggleFavoriteInStore]);

  const handleDelete = useCallback(() => {
    Alert.alert(
      'Remove Item',
      'Remove this item from your wardrobe? This cannot be undone.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            setIsDeleting(true);
            try {
              await removeItemFromStore(item.id);
              navigation.goBack();
            } catch {
              Alert.alert('Error', 'Failed to remove item. Please try again.');
            } finally {
              setIsDeleting(false);
            }
          },
        },
      ]
    );
  }, [item.id, navigation, removeItemFromStore]);

  const toggleSeason = useCallback((season: Season) => {
    Haptics.selectionAsync();
    setSelectedSeasons(prev =>
      prev.includes(season) ? prev.filter(s => s !== season) : [...prev, season]
    );
  }, []);

  const toggleOccasion = useCallback((occasion: Occasion) => {
    Haptics.selectionAsync();
    setSelectedOccasions(prev =>
      prev.includes(occasion) ? prev.filter(o => o !== occasion) : [...prev, occasion]
    );
  }, []);

  const handleSave = useCallback(async () => {
    setIsSaving(true);
    const colorData = COLORS.find(c => c.id === selectedColor) ?? COLORS[2];
    const updates: Partial<ClothingItem> = {
      subCategory:  selectedType,
      primaryColor: colorData.label,
      colorHex:     colorData.hex,
      seasons:      selectedSeasons,
      occasions:    selectedOccasions,
    };
    try {
      // Optimistic local update
      updateItemInStore(item.id, updates);
      // Persist to server
      await wardrobeApi.update(item.id, {
        subCategory:  selectedType,
        primaryColor: colorData.label,
        colorHex:     colorData.hex,
        seasons:      selectedSeasons as any,
        occasions:    selectedOccasions as any,
      } as any);
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
      setIsEditing(false);
    } catch {
      Alert.alert('Error', 'Failed to save changes. Please try again.');
    } finally {
      setIsSaving(false);
    }
  }, [item.id, selectedType, selectedColor, selectedSeasons, selectedOccasions, updateItemInStore]);

  const handleEditToggle = useCallback(() => {
    if (isEditing) {
      // Discard — reset to current item values
      setSelectedType(resolveCategoryId(item));
      setSelectedColor(resolveColorId(item));
      setSelectedSeasons(item.seasons ?? []);
      setSelectedOccasions(item.occasions ?? []);
    }
    setIsEditing(e => !e);
  }, [isEditing, item]);

  // ── Derived display values ────────────────────────────────────────────
  const displayColor = COLORS.find(c => c.id === selectedColor);
  const isFavorite   = item.isFavorite;

  return (
    <SafeAreaView style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigation.goBack()} style={styles.iconButton}>
          <Ionicons name="chevron-back" size={24} color="#1C1C1E" />
        </TouchableOpacity>

        <Text style={styles.headerTitle} numberOfLines={2}>
          {item.name || item.subCategory || item.category}
        </Text>

        <View style={styles.headerActions}>
          <TouchableOpacity onPress={handleFavorite} style={styles.iconButton}>
            <Ionicons
              name={isFavorite ? 'heart' : 'heart-outline'}
              size={24}
              color={isFavorite ? '#FF3B30' : '#1C1C1E'}
            />
          </TouchableOpacity>
          <TouchableOpacity onPress={handleEditToggle} style={styles.iconButton}>
            <Ionicons
              name={isEditing ? 'close' : 'create-outline'}
              size={24}
              color="#1C1C1E"
            />
          </TouchableOpacity>
        </View>
      </View>

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Image */}
        <View style={styles.imageContainer}>
          {item.imageUrl ? (
            <Image
              source={{ uri: item.imageUrl }}
              style={styles.image}
              resizeMode="contain"
            />
          ) : (
            <View style={styles.imagePlaceholder}>
              <Ionicons name="shirt-outline" size={72} color="#C7C7CC" />
            </View>
          )}
        </View>

        {/* Stats row */}
        <View style={styles.statsRow}>
          <View style={styles.statItem}>
            <Text style={styles.statValue}>{item.wearCount ?? 0}</Text>
            <Text style={styles.statLabel}>{t('clothingDetail.worn')}</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>
              {item.lastWornAt
                ? new Date(item.lastWornAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
                : '—'}
            </Text>
            <Text style={styles.statLabel}>{t('clothingDetail.lastWorn')}</Text>
          </View>
          <View style={styles.statDivider} />
          <View style={styles.statItem}>
            <Text style={styles.statValue}>
              {item.detectionConfidence != null
                ? `${Math.round(item.detectionConfidence * 100)}%`
                : '—'}
            </Text>
            <Text style={styles.statLabel}>{t('clothingDetail.aiConfidence')}</Text>
          </View>
        </View>

        {/* Meta chips (read-only) */}
        {!isEditing && (
          <View style={styles.section}>
            <View style={styles.metaRow}>
              {item.brand ? (
                <View style={styles.metaChip}>
                  <Text style={styles.metaChipText}>{item.brand}</Text>
                </View>
              ) : null}
              {item.material ? (
                <View style={styles.metaChip}>
                  <Text style={styles.metaChipText}>{item.material}</Text>
                </View>
              ) : null}
              {item.pattern ? (
                <View style={styles.metaChip}>
                  <Text style={styles.metaChipText}>{item.pattern}</Text>
                </View>
              ) : null}
            </View>
          </View>
        )}

        {/* ── Type ── */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>{t('clothingEditor.type')}</Text>
          {isEditing ? (
            <View style={styles.chipGrid}>
              {TYPES.map(type => (
                <TouchableOpacity
                  key={type.id}
                  style={[styles.chip, selectedType === type.id && styles.chipSelected]}
                  onPress={() => { setSelectedType(type.id); Haptics.selectionAsync(); }}
                  activeOpacity={0.7}
                >
                  <Text style={[styles.chipText, selectedType === type.id && styles.chipTextSelected]}>
                    {type.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          ) : (
            <Text style={styles.valueText}>
              {TYPES.find(t => t.id === selectedType)?.label ?? selectedType}
            </Text>
          )}
        </View>

        {/* ── Colour ── */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>{t('clothingEditor.colour')}</Text>
          {isEditing ? (
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.colorRow}
            >
              {COLORS.map(color => (
                <TouchableOpacity
                  key={color.id}
                  style={[
                    styles.colorCircle,
                    { backgroundColor: color.hex },
                    (color.id === 'white' || color.id === 'cream' || color.id === 'beige') && styles.colorCircleLight,
                    selectedColor === color.id && styles.colorCircleSelected,
                  ]}
                  onPress={() => { setSelectedColor(color.id); Haptics.selectionAsync(); }}
                  activeOpacity={0.7}
                  accessibilityLabel={color.label}
                  accessibilityRole="button"
                  accessibilityState={{ selected: selectedColor === color.id }}
                >
                  {selectedColor === color.id && (
                    <Ionicons
                      name="checkmark"
                      size={16}
                      color={['white', 'cream', 'beige', 'yellow'].includes(color.id) ? '#1C1C1E' : '#FFFFFF'}
                    />
                  )}
                </TouchableOpacity>
              ))}
            </ScrollView>
          ) : (
            <View style={styles.colorDisplay}>
              {displayColor && (
                <View
                  style={[
                    styles.colorSwatch,
                    { backgroundColor: displayColor.hex },
                    ['white', 'cream', 'beige'].includes(displayColor.id) && styles.colorCircleLight,
                  ]}
                />
              )}
              <Text style={styles.valueText}>{displayColor?.label ?? item.primaryColor}</Text>
            </View>
          )}
        </View>

        {/* ── Season ── */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>{t('clothingEditor.seasonLabel')}</Text>
          <View style={styles.chipRow}>
            {SEASONS.map(season => {
              const active = selectedSeasons.includes(season.id);
              return (
                <TouchableOpacity
                  key={season.id}
                  style={[styles.chip, active && styles.chipSelected, !isEditing && styles.chipReadOnly]}
                  onPress={() => isEditing && toggleSeason(season.id)}
                  activeOpacity={isEditing ? 0.7 : 1}
                >
                  <Text style={[styles.chipText, active && styles.chipTextSelected]}>
                    {season.label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* ── Occasions ── */}
        <View style={styles.section}>
          <Text style={styles.sectionLabel}>{t('clothingDetail.occasions')}</Text>
          <View style={styles.chipGrid}>
            {OCCASIONS.map(occ => {
              const active = selectedOccasions.includes(occ.id);
              return (
                <TouchableOpacity
                  key={occ.id}
                  style={[styles.chip, active && styles.chipSelected, !isEditing && styles.chipReadOnly]}
                  onPress={() => isEditing && toggleOccasion(occ.id)}
                  activeOpacity={isEditing ? 0.7 : 1}
                >
                  <Ionicons
                    name={occ.icon as any}
                    size={14}
                    color={active ? '#FFFFFF' : '#636366'}
                    style={{ marginRight: 4 }}
                  />
                  <Text style={[styles.chipText, active && styles.chipTextSelected]}>
                    {occ.label}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
        </View>

        {/* Delete button */}
        <TouchableOpacity
          style={styles.deleteButton}
          onPress={handleDelete}
          disabled={isDeleting}
          activeOpacity={0.7}
        >
          {isDeleting ? (
            <ActivityIndicator size="small" color="#FF3B30" />
          ) : (
            <>
              <Ionicons name="trash-outline" size={18} color="#FF3B30" style={{ marginRight: 6 }} />
              <Text style={styles.deleteButtonText}>{t('clothingDetail.removeFromWardrobe')}</Text>
            </>
          )}
        </TouchableOpacity>

        <View style={{ height: isEditing ? 100 : 32 }} />
      </ScrollView>

      {/* Save bar — only visible in edit mode */}
      {isEditing && (
        <View style={styles.saveBar}>
          <TouchableOpacity
            style={styles.saveButton}
            onPress={handleSave}
            disabled={isSaving}
            activeOpacity={0.8}
          >
            {isSaving ? (
              <ActivityIndicator size="small" color="#FFFFFF" />
            ) : (
              <Text style={styles.saveButtonText}>{t('clothingEditor.save')}</Text>
            )}
          </TouchableOpacity>
        </View>
      )}
    </SafeAreaView>
  );
};

// ── Styles ─────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FFFFFF',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
  },
  notFoundText: {
    fontSize: 17,
    color: '#636366',
    marginTop: 12,
    marginBottom: 24,
  },
  backButton: {
    backgroundColor: '#1C1C1E',
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 24,
  },
  backButtonText: {
    color: '#FFFFFF',
    fontWeight: '600',
    fontSize: 15,
  },
  // Header
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: '#E5E5EA',
  },
  headerTitle: {
    flex: 1,
    fontSize: 17,
    fontWeight: '600',
    color: '#1C1C1E',
    textAlign: 'center',
    marginHorizontal: 4,
    textTransform: 'capitalize',
  },
  headerActions: {
    flexDirection: 'row',
  },
  iconButton: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 20,
  },
  // Scroll
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 20,
    paddingTop: 16,
  },
  // Image
  imageContainer: {
    height: 300,
    borderRadius: 20,
    overflow: 'hidden',
    backgroundColor: '#F2F2F7',
    marginBottom: 20,
  },
  image: {
    width: '100%',
    height: '100%',
  },
  imagePlaceholder: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  // Stats
  statsRow: {
    flexDirection: 'row',
    backgroundColor: '#F2F2F7',
    borderRadius: 16,
    paddingVertical: 16,
    marginBottom: 24,
  },
  statItem: {
    flex: 1,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 18,
    fontWeight: '700',
    color: '#1C1C1E',
  },
  statLabel: {
    fontSize: 12,
    color: '#8E8E93',
    marginTop: 2,
  },
  statDivider: {
    width: StyleSheet.hairlineWidth,
    backgroundColor: '#C7C7CC',
    marginVertical: 4,
  },
  // Sections
  section: {
    marginBottom: 24,
  },
  sectionLabel: {
    fontSize: 15,
    fontWeight: '600',
    color: '#1C1C1E',
    marginBottom: 10,
  },
  valueText: {
    fontSize: 15,
    color: '#3C3C43',
    textTransform: 'capitalize',
  },
  // Meta chips (read-only brand/material/pattern)
  metaRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  metaChip: {
    backgroundColor: '#F2F2F7',
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 5,
  },
  metaChipText: {
    fontSize: 13,
    color: '#636366',
    textTransform: 'capitalize',
  },
  // Editable chips
  chipRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chipGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#F2F2F7',
    borderWidth: 1,
    borderColor: '#E5E5EA',
  },
  chipSelected: {
    backgroundColor: '#1C1C1E',
    borderColor: '#1C1C1E',
  },
  chipReadOnly: {
    opacity: 1,
  },
  chipText: {
    fontSize: 14,
    color: '#1C1C1E',
    fontWeight: '500',
  },
  chipTextSelected: {
    color: '#FFFFFF',
    fontWeight: '600',
  },
  // Color
  colorRow: {
    flexDirection: 'row',
    gap: 10,
    paddingVertical: 4,
  },
  colorCircle: {
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.15,
    shadowRadius: 2,
  },
  colorCircleLight: {
    borderWidth: 1.5,
    borderColor: '#D1D1D6',
  },
  colorCircleSelected: {
    borderWidth: 3,
    borderColor: '#007AFF',
    shadowColor: '#007AFF',
    shadowOpacity: 0.4,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 0 },
  },
  colorDisplay: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
  },
  colorSwatch: {
    width: 28,
    height: 28,
    borderRadius: 14,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.12,
    shadowRadius: 2,
  },
  // Delete
  deleteButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 14,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: '#FFCDD2',
    backgroundColor: '#FFF5F5',
    marginBottom: 8,
  },
  deleteButtonText: {
    fontSize: 15,
    fontWeight: '600',
    color: '#FF3B30',
  },
  // Save bar
  saveBar: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    paddingHorizontal: 20,
    paddingBottom: 34,
    paddingTop: 12,
    backgroundColor: '#FFFFFF',
    borderTopWidth: StyleSheet.hairlineWidth,
    borderTopColor: '#E5E5EA',
  },
  saveButton: {
    backgroundColor: '#1C1C1E',
    paddingVertical: 16,
    borderRadius: 28,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#FFFFFF',
    fontSize: 17,
    fontWeight: '600',
  },
});

export default ClothingDetailScreen;
