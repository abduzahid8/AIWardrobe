/**
 * ClothingDetailEditor - Post-analyzer edit screen
 * Allows users to verify and edit AI-detected clothing attributes
 */

import { acknowledgeScanReviewItem } from '../src/services/scanReviews';
import { flushWardrobePersistence } from '../store/wardrobeStore';
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { View, StyleSheet, TouchableOpacity, ScrollView, StatusBar, Alert, ActivityIndicator } from 'react-native';
import { ScaledText } from './ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { useNavigation, useRoute } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import useWardrobeStore from '../store/wardrobeStore';
import useAuthStore from '../store/auth';
import { saveClothingEdits } from '../src/services/clothingEditorService';
import { normalizeCategory } from '../src/utils/categoryMapper';
import { authEvents } from '../store/authEvents';
import type { Season, Occasion, ClothingItem } from '../src/types/domain';
import { ExternalAIService, type ProcessingResult } from '../src/services/externalAIService';
import * as FileSystem from 'expo-file-system/legacy';
import { mapCategoryToType, mapColorToId } from '../src/utils/mappingUtils';
import { Image } from '../src/components/MediaImage';

interface ClothingDetailEditorProps {
    imageUri?: string;
    initialData?: {
        type?: string;
        color?: string;
        season?: string;
    };
    onSave?: (data: ClothingData) => void | Promise<void>;
    onCancel?: () => void;
}

export interface ClothingData {
    type: string;
    color: string;
    season: string;
    weather: string[];
}

// ── AI inference helpers ───────────────────────────────────────────────
const inferSeasonFromType = (type: string, material?: string | null): string => {
    const t = type.toLowerCase();
    const m = (material || '').toLowerCase();
    if (m.includes('wool') || m.includes('fur') || m.includes('fleece') || m.includes('cashmere') || m.includes('down')) return 'winter';
    if (m.includes('linen') || m.includes('silk')) return 'summer';
    if (t === 'outerwear') return 'winter';
    if (t === 'sportswear' || t === 'shoes') return 'spring';
    if (t === 'bottoms' && !m) return 'spring';
    return 'spring';
};

const inferWeatherFromType = (type: string, material?: string | null): string[] => {
    const t = type.toLowerCase();
    const m = (material || '').toLowerCase();
    if (t === 'outerwear' || m.includes('wool') || m.includes('fleece')) return ['snow', 'wind'];
    if (t === 'sportswear') return ['sun', 'wind'];
    return ['sun'];
};

const styleToOccasions = (style?: string): Occasion[] => {
    const s = (style || '').toLowerCase();
    if (s.includes('formal') || s.includes('elegant') || s.includes('business')) return ['formal', 'work'] as Occasion[];
    if (s.includes('sport')) return ['sport', 'casual'] as Occasion[];
    if (s.includes('semi_classic') || s.includes('semi-classic') || s.includes('urban')) return ['casual', 'outdoor'] as Occasion[];
    return ['casual'] as Occasion[];
};

const ClothingDetailEditor: React.FC<ClothingDetailEditorProps> = ({
    imageUri,
    initialData,
    onSave,
    onCancel,
}) => {
    const { t } = useTranslation();
    const navigation = useNavigation();
    const route = useRoute<any>();

    const TYPES = [
        { id: 'tops', label: t('clothingEditor.types.tops') },
        { id: 'bottoms', label: t('clothingEditor.types.bottoms') },
        { id: 'shoes', label: t('clothingEditor.types.shoes') },
        { id: 'accessories', label: t('clothingEditor.types.accessories') },
        { id: 'outerwear', label: t('clothingEditor.types.outerwear') },
        { id: 'sportswear', label: t('clothingEditor.types.sportswear') },
        { id: 'homewear', label: t('clothingEditor.types.homewear') },
    ];

    const COLORS = [
        { id: 'black', label: t('clothingEditor.colors.black'), hex: '#1C1C1E' },
        { id: 'white', label: t('clothingEditor.colors.white'), hex: '#FFFFFF' },
        { id: 'grey', label: t('clothingEditor.colors.grey'), hex: '#8E8E93' },
        { id: 'beige', label: t('clothingEditor.colors.beige'), hex: '#C7B299' },
        { id: 'cream', label: t('clothingEditor.colors.cream') || 'Cream', hex: '#FFFDD0' },
        { id: 'brown', label: t('clothingEditor.colors.brown'), hex: '#8B4513' },
        { id: 'red', label: t('clothingEditor.colors.red'), hex: '#FF3B30' },
        { id: 'pink', label: t('clothingEditor.colors.pink') || 'Pink', hex: '#FF6B9D' },
        { id: 'orange', label: t('clothingEditor.colors.orange') || 'Orange', hex: '#FF9500' },
        { id: 'yellow', label: t('clothingEditor.colors.yellow') || 'Yellow', hex: '#FFCC00' },
        { id: 'green', label: t('clothingEditor.colors.green'), hex: '#34C759' },
        { id: 'teal', label: t('clothingEditor.colors.teal') || 'Teal', hex: '#5AC8FA' },
        { id: 'blue', label: t('clothingEditor.colors.blue'), hex: '#007AFF' },
        { id: 'navy', label: t('clothingEditor.colors.navy') || 'Navy', hex: '#1B2A4A' },
        { id: 'purple', label: t('clothingEditor.colors.purple') || 'Purple', hex: '#AF52DE' },
        { id: 'multicolor', label: t('clothingEditor.colors.multicolor') || 'Multi', hex: '#FF6B6B' },
    ];

    const SEASONS = [
        { id: 'summer', label: t('clothingEditor.seasons.summer') },
        { id: 'winter', label: t('clothingEditor.seasons.winter') },
        { id: 'autumn', label: t('clothingEditor.seasons.autumn') },
        { id: 'spring', label: t('clothingEditor.seasons.spring') },
    ];

    const WEATHER = [
        { id: 'rain', icon: 'rainy-outline', label: t('clothingEditor.weather.rain') },
        { id: 'sun', icon: 'sunny-outline', label: t('clothingEditor.weather.sunny') },
        { id: 'snow', icon: 'snow-outline', label: t('clothingEditor.weather.snow') },
        { id: 'wind', icon: 'flag-outline', label: t('clothingEditor.weather.wind') },
    ];

    // Existing item editing mode
    const existingItem = route.params?.existingItem as ClothingItem | undefined;
    const isEditingExisting = !!existingItem;
    const [isSaving, setIsSaving] = useState(false);
    const saveInFlight = useRef(false);
    const retainedCreate = useRef<{ owner: string; source: string | undefined; id: string } | null>(null);

    const domainSeasonToEditor = (season?: string): string => {
        if (season === 'fall') return 'autumn';
        return season || 'spring';
    };
    const editorSeasonToDomain = (season: string): Season => {
        if (season === 'autumn') return 'fall';
        return season as Season;
    };

    const [analysisMetadata, setAnalysisMetadata] = useState<ProcessingResult | null>(null);

    // Get data from route params, props, or existing item
    const itemImageUri = existingItem?.imageUrl || route.params?.imageUri || imageUri;
    const detectedType = existingItem?.subCategory || route.params?.detectedType || initialData?.type || 'tops';
    const detectedColor = (existingItem?.primaryColor ? mapColorToId(existingItem.primaryColor) : undefined)
        || route.params?.detectedColor || initialData?.color || 'beige';
    const detectedStyle: string | undefined = route.params?.detectedStyle ?? analysisMetadata?.classification?.attributes.style;
    const detectedMaterial: string | undefined = existingItem?.material ?? route.params?.detectedMaterial ?? analysisMetadata?.classification?.attributes.material ?? undefined;
    const aiConfidence: number | undefined = existingItem?.detectionConfidence ?? route.params?.aiConfidence ?? analysisMetadata?.classification?.confidence;
    const detectedDescription: string | undefined = existingItem?.name || route.params?.detectedDescription || analysisMetadata?.description || undefined;
    const detectedPattern = existingItem?.pattern ?? route.params?.detectedItem?.pattern ?? '';
    const detectedSpecificType = route.params?.detectedItem?.specificType || route.params?.detectedItem?.itemType || analysisMetadata?.classification?.category;

    // AI analysis for images opened without pre-detected data (skip for existing items)
    const [isAnalyzing, setIsAnalyzing] = useState(!isEditingExisting && !route.params?.detectedType && !!itemImageUri);

    const [selectedType, setSelectedType] = useState(
        isEditingExisting && existingItem?.subCategory
            ? mapCategoryToType(existingItem.subCategory, existingItem.category || '')
            : detectedType
    );
    const [selectedColor, setSelectedColor] = useState(detectedColor);
    const [selectedSeason, setSelectedSeason] = useState(
        isEditingExisting && existingItem?.seasons?.length
            ? domainSeasonToEditor(existingItem.seasons[0])
            : (route.params?.detectedType
                ? inferSeasonFromType(detectedType, detectedMaterial)
                : 'spring')
    );
    const [selectedWeather, setSelectedWeather] = useState<string[]>(
        isEditingExisting && existingItem
            ? inferWeatherFromType(detectedType, detectedMaterial)
            : (route.params?.detectedType
                ? inferWeatherFromType(detectedType, detectedMaterial)
                : ['sun'])
    );

    // Live AI analysis when editor opened with just imageUri (no pre-classified data)
    useEffect(() => {
        if (!isAnalyzing || !itemImageUri) return;

        const revision = authEvents.getRevision();
        let cancelled = false;
        const canApply = () => !cancelled && revision === authEvents.getRevision();
        const analyzeEditorImage = async () => {
            try {
                const b64 = await FileSystem.readAsStringAsync(itemImageUri, { encoding: 'base64' });
                if (!canApply()) return;
                const aiResult = await ExternalAIService.classifyOnly(b64);
                if (!canApply()) return;
                if (!aiResult.success || !aiResult.classification) throw new Error('classification_unavailable');
                const cat = aiResult.classification.category.toLowerCase();
                const sec = aiResult.classification.section.toLowerCase();
                const newType = mapCategoryToType(cat, sec);
                const newColor = mapColorToId(aiResult.classification.attributes?.color || '');
                setAnalysisMetadata(aiResult);
                setSelectedType(newType);
                setSelectedColor(newColor);
                setSelectedSeason(inferSeasonFromType(newType, aiResult.classification.attributes?.material));
                setSelectedWeather(inferWeatherFromType(newType, aiResult.classification.attributes?.material));
            } catch {
                if (canApply()) Alert.alert(t('videoAnalysis.analysisFailed'), t('videoAnalysis.somethingWrong'));
            } finally {
                if (canApply()) setIsAnalyzing(false);
            }
        };
        void analyzeEditorImage();
        return () => { cancelled = true; };
    }, []);

    const handleTypeSelect = (typeId: string) => {
        setSelectedType(typeId);
        Haptics.selectionAsync();
    };

    const handleColorSelect = (colorId: string) => {
        setSelectedColor(colorId);
        Haptics.selectionAsync();
    };

    const handleSeasonSelect = (seasonId: string) => {
        setSelectedSeason(seasonId);
        Haptics.selectionAsync();
    };

    const toggleWeather = (weatherId: string) => {
        setSelectedWeather(prev =>
            prev.includes(weatherId)
                ? prev.filter(w => w !== weatherId)
                : [...prev, weatherId]
        );
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    };

    const saveChanges = async () => {
        const saveRevision = authEvents.getRevision();
        const data: ClothingData = {
            type: selectedType,
            color: selectedColor,
            season: selectedSeason,
            weather: selectedWeather,
        };

        // If onSave callback provided, use it
        if (onSave) {
            await onSave(data);
            if (saveRevision === authEvents.getRevision()) Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            return;
        }

        const selectedColorData = COLORS.find(c => c.id === selectedColor) || COLORS[2];

        // Editing existing item
        if (isEditingExisting && existingItem) {
            const revision = authEvents.getRevision();
            try {
                const updateItemInStore = useWardrobeStore.getState().updateItem;
                const updates: Partial<ClothingItem> = {
                    ...(selectedType !== mapCategoryToType(existingItem.subCategory, existingItem.category || '') ? {
                        category: selectedType === 'sportswear' || selectedType === 'homewear' ? existingItem.category : normalizeCategory(selectedType),
                        subCategory: selectedType,
                    } : {}),
                    primaryColor: selectedColorData.id,
                    colorHex: selectedColorData.hex,
                    seasons: [editorSeasonToDomain(selectedSeason)] as Season[],
                };
                await saveClothingEdits(existingItem.id, updates);
                if (revision !== authEvents.getRevision()) return;
                updateItemInStore(existingItem.id, updates);
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                Alert.alert(
                    t('clothingEditor.saved'),
                    t('clothingEditor.itemAddedToWardrobe'),
                    [{ text: t('clothingEditor.ok'), onPress: () => navigation.goBack() }]
                );
            } catch (error: any) {
                console.error('Failed to update item:', error);
                if (revision === authEvents.getRevision()) Alert.alert(t('clothingEditor.error'), t('clothingEditor.failedToSaveItem'));
            }
            return;
        }

        // Otherwise save new item to wardrobe
        const { user } = useAuthStore.getState();
        if (!user) {
            Alert.alert(t('clothingEditor.loginRequired'), t('clothingEditor.pleaseLoginToSave'));
            return;
        }

        const addItem = useWardrobeStore.getState().addItem;

        try {
            const itemInput = {
                userId: user.id,
                imageUrl: itemImageUri || '',
                category: selectedType as any,
                subCategory: detectedSpecificType && mapCategoryToType(detectedSpecificType) === selectedType
                    ? detectedSpecificType : selectedType,
                primaryColor: selectedColorData.id,
                colorHex: selectedColorData.hex,
                pattern: typeof detectedPattern === 'string' ? detectedPattern : '',
                material: detectedMaterial || '',
                brand: '',
                name: detectedDescription
                    ? detectedDescription.slice(0, 60)
                    : `${selectedColorData.label} ${selectedType}`,
                seasons: [editorSeasonToDomain(selectedSeason)] as Season[],
                occasions: styleToOccasions(detectedStyle),
                ...(typeof aiConfidence === 'number' && Number.isFinite(aiConfidence) && aiConfidence >= 0 && aiConfidence <= 1
                    ? { detectionConfidence: aiConfidence } : {}),
            };
            if (route.params?.scanReviewItemId) await addItem(itemInput, { idempotencyId: route.params.scanReviewItemId });
            else if (retainedCreate.current?.owner === user.id && retainedCreate.current.source === itemImageUri) {
                await addItem(itemInput, { idempotencyId: retainedCreate.current.id });
            } else await addItem(itemInput);
            if (saveRevision !== authEvents.getRevision()) return;
            if (route.params?.scanReviewId && route.params.scanReviewItemId) {
                await flushWardrobePersistence();
                if (saveRevision !== authEvents.getRevision()) return;
                try { await acknowledgeScanReviewItem(user.id, route.params.scanReviewId, route.params.scanReviewItemId); }
                catch { console.warn('Saved wardrobe item; scan-review acknowledgement will need retry'); }
            }
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

            Alert.alert(
                t('clothingEditor.saved'),
                t('clothingEditor.itemAddedToWardrobe'),
                [{ text: t('clothingEditor.ok'), onPress: () => navigation.goBack() }]
            );
        } catch (error: any) {
            if (saveRevision !== authEvents.getRevision()) return;
            if (error?.code === 'WARDROBE_LIMIT_REACHED') {
                if (error.savedLocally && typeof error.itemId === 'string') {
                    retainedCreate.current = { owner: user.id, source: itemImageUri, id: error.itemId };
                }
                Alert.alert(
                    t('clothingEditor.wardrobeFull'),
                    error.savedLocally ? t('wardrobeSync.capacityMessage', { limit: error.limit ?? 20 })
                        : t('clothingEditor.freePlanLimit', { limit: error.limit ?? 20 }),
                    [
                        { text: t('clothingEditor.maybeLater'), style: 'cancel' },
                        {
                            text: t('clothingEditor.upgrade'),
                            style: 'default',
                            onPress: () => (navigation as any).navigate('Paywall'),
                        },
                    ]
                );
                return;
            }
            console.error('Failed to save item:', error);
            Alert.alert(t('clothingEditor.error'), t('clothingEditor.failedToSaveItem'));
        }
    };

    const handleSave = async () => {
        // A ref closes the same-tick window before React renders disabled=true.
        if (isAnalyzing || saveInFlight.current) return;
        saveInFlight.current = true;
        setIsSaving(true);
        const revision = authEvents.getRevision();
        try {
            await saveChanges();
        } catch (error) {
            console.error('Failed to save item:', error);
            if (revision === authEvents.getRevision()) Alert.alert(t('clothingEditor.error'), t('clothingEditor.failedToSaveItem'));
        } finally {
            saveInFlight.current = false;
            setIsSaving(false);
        }
    };

    const handleDelete = useCallback(() => {
        if (!existingItem) return;
        Alert.alert(
            t('clothingDetail.removeFromWardrobe'),
            'This cannot be undone.',
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Remove',
                    style: 'destructive',
                    onPress: async () => {
                        try {
                            const removeItem = useWardrobeStore.getState().removeItem;
                            await removeItem(existingItem.id);
                            navigation.goBack();
                        } catch {
                            Alert.alert('Error', 'Failed to remove item. Please try again.');
                        }
                    },
                },
            ]
        );
    }, [existingItem, navigation]);

    const handleClose = () => {
        if (onCancel) {
            onCancel();
            return;
        }
        // Guarded goBack: if this screen was opened with no history behind
        // it (e.g. a deep link that replaced root state), plain goBack()
        // no-ops and strands the user — fall back to the main tabs.
        if (navigation.canGoBack()) {
            navigation.goBack();
        } else {
            (navigation as any).navigate('Main');
        }
    };


    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" />
            <SafeAreaView style={styles.safeArea}>
                {/* Header — explicit back chevron (left) + close (right).
                    Previously only a right-aligned X that could end up under
                    the status bar/overlay area, leaving gesture-only exit
                    (undiscoverable on Android). Both buttons go back. */}
                <View style={styles.header}>
                    <TouchableOpacity
                        onPress={handleClose}
                        style={styles.backButton}
                        testID="clothingEditorBack"
                        accessibilityLabel={t('common.back')}
                        accessibilityRole="button"
                        hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                    >
                        <Ionicons name="chevron-back" size={26} color="#1C1C1E" />
                    </TouchableOpacity>
                    <View style={styles.headerSpacer} />
                    <TouchableOpacity
                        onPress={handleClose}
                        style={styles.closeButton}
                        accessibilityLabel={t('common.close')}
                        accessibilityRole="button"
                        hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                    >
                        <Ionicons name="close" size={24} color="#1C1C1E" />
                    </TouchableOpacity>
                </View>

                <ScrollView
                    style={styles.scrollView}
                    contentContainerStyle={styles.content}
                    showsVerticalScrollIndicator={false}
                >
                    {/* AI Analysis Banner — hidden when editing existing item */}
                    {!isEditingExisting && isAnalyzing && (
                        <View style={styles.aiBanner}>
                            <ActivityIndicator size="small" color="#007AFF" style={{ marginRight: 8 }} />
                            <ScaledText style={styles.aiBannerText}>{t('clothingEditor.aiAnalyzing')}</ScaledText>
                        </View>
                    )}
                    {!isEditingExisting && !isAnalyzing && typeof aiConfidence === 'number' && aiConfidence > 0 && aiConfidence <= 1 && (
                        <View style={styles.aiBanner}>
                            <Ionicons name="sparkles" size={14} color="#007AFF" style={{ marginRight: 6 }} />
                            <ScaledText style={styles.aiBannerText}>
                                {t('clothingEditor.aiDetected')}
                            </ScaledText>
                            <View style={styles.aiBannerBadge}>
                                <ScaledText style={styles.aiBannerBadgeText}>
                                    {Math.round(aiConfidence * 100)}% {t('clothingEditor.confidence')}
                                </ScaledText>
                            </View>
                            {detectedMaterial ? (
                                <ScaledText style={styles.aiBannerMaterial}> · {detectedMaterial}</ScaledText>
                            ) : null}
                        </View>
                    )}
                    {detectedDescription && !isEditingExisting ? (
                        <ScaledText style={styles.aiDescription}>{detectedDescription}</ScaledText>
                    ) : null}

                    {/* Clothing Image */}
                    <View style={styles.imageContainer}>
                        {itemImageUri ? (
                            <Image source={{ uri: itemImageUri }} style={styles.clothingImage} resizeMode="contain" />
                        ) : (
                            <View style={styles.placeholderImage}>
                                <Ionicons name="shirt-outline" size={64} color="#C7C7CC" />
                            </View>
                        )}
                    </View>

                    {/* Type Section */}
                    <View style={styles.section}>
                        <ScaledText style={styles.sectionLabel}>{t('clothingEditor.type')}</ScaledText>
                        <View style={styles.typeGrid}>
                            {TYPES.map(type => (
                                <TouchableOpacity
                                    key={type.id}
                                    style={[
                                        styles.typeChip,
                                        selectedType === type.id && styles.typeChipSelected,
                                    ]}
                                    onPress={() => handleTypeSelect(type.id)}
                                    activeOpacity={0.7}
                                >
                                    <ScaledText
                                        style={[
                                            styles.typeChipText,
                                            selectedType === type.id && styles.typeChipTextSelected,
                                        ]}
                                    >
                                        {type.label}
                                    </ScaledText>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>

                    {/* Colour Section */}
                    <View style={styles.section}>
                        <View style={styles.sectionHeader}>
                            <ScaledText style={styles.sectionLabel}>{t('clothingEditor.colour')}</ScaledText>
                            <ScaledText style={styles.allLink}>{COLORS.find(c => c.id === selectedColor)?.label || ''}</ScaledText>
                        </View>
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.colorRow}
                            style={styles.colorScrollView}
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
                                    onPress={() => handleColorSelect(color.id)}
                                    activeOpacity={0.7}
                                    accessibilityLabel={color.label}
                                    accessibilityRole="button"
                                    accessibilityState={{ selected: selectedColor === color.id }}
                                >
                                    {selectedColor === color.id && (
                                        <Ionicons
                                            name="checkmark"
                                            size={16}
                                            color={(color.id === 'white' || color.id === 'cream' || color.id === 'beige' || color.id === 'yellow') ? '#1C1C1E' : '#FFFFFF'}
                                        />
                                    )}
                                </TouchableOpacity>
                            ))}
                        </ScrollView>
                    </View>

                    {/* Season Section */}
                    <View style={styles.section}>
                        <ScaledText style={styles.sectionLabel}>{t('clothingEditor.seasonLabel')}</ScaledText>
                        <View style={styles.seasonRow}>
                            {SEASONS.map(season => (
                                <TouchableOpacity
                                    key={season.id}
                                    style={[
                                        styles.seasonChip,
                                        selectedSeason === season.id && styles.seasonChipSelected,
                                    ]}
                                    onPress={() => handleSeasonSelect(season.id)}
                                    activeOpacity={0.7}
                                >
                                    <ScaledText
                                        style={[
                                            styles.seasonChipText,
                                            selectedSeason === season.id && styles.seasonChipTextSelected,
                                        ]}
                                    >
                                        {season.label}
                                    </ScaledText>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>

                    {/* Weather Section */}
                    <View style={styles.section}>
                        <ScaledText style={styles.sectionLabel}>{t('clothingEditor.weatherLabel')}</ScaledText>
                        <View style={styles.weatherRow}>
                            {WEATHER.map(weather => {
                                const isSelected = selectedWeather.includes(weather.id);
                                return (
                                    <TouchableOpacity
                                        key={weather.id}
                                        style={[
                                            styles.weatherCircle,
                                            isSelected && styles.weatherCircleSelected,
                                        ]}
                                        onPress={() => toggleWeather(weather.id)}
                                        activeOpacity={0.7}
                                    >
                                        <Ionicons
                                            name={weather.icon as any}
                                            size={22}
                                            color={isSelected ? '#FFFFFF' : '#8E8E93'}
                                        />
                                    </TouchableOpacity>
                                );
                            })}
                        </View>
                    </View>

                    {/* Delete button when editing existing */}
                    {isEditingExisting && (
                        <TouchableOpacity
                            style={styles.deleteButton}
                            onPress={handleDelete}
                            activeOpacity={0.7}
                        >
                            <Ionicons name="trash-outline" size={18} color="#FF3B30" style={{ marginRight: 6 }} />
                            <ScaledText style={styles.deleteButtonText}>{t('clothingDetail.removeFromWardrobe')}</ScaledText>
                        </TouchableOpacity>
                    )}

                    {/* Bottom padding for save button */}
                    <View style={{ height: 100 }} />
                </ScrollView>

                {/* Save Button */}
                <View style={styles.saveButtonContainer}>
                    <TouchableOpacity
                        testID="clothingEditorSave"
                        accessibilityRole="button"
                        style={styles.saveButton}
                        onPress={handleSave}
                        disabled={isSaving || isAnalyzing}
                        accessibilityState={{ disabled: isSaving || isAnalyzing, busy: isSaving || isAnalyzing }}
                        activeOpacity={0.8}
                    >
                        <ScaledText style={styles.saveButtonText}>
                            {isSaving ? t('clothingEditor.saving') : isEditingExisting ? t('clothingEditor.updateItem') : t('clothingEditor.save')}
                        </ScaledText>
                    </TouchableOpacity>
                </View>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFFFFF',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingTop: 8,
        paddingBottom: 8,
    },
    backButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: '#F2F2F7',
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerSpacer: {
        flex: 1,
    },
    closeButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: '#F2F2F7',
        alignItems: 'center',
        justifyContent: 'center',
    },
    scrollView: {
        flex: 1,
    },
    content: {
        paddingHorizontal: 24,
    },
    imageContainer: {
        alignItems: 'center',
        marginBottom: 32,
        height: 280,
    },
    clothingImage: {
        width: '100%',
        height: '100%',
        borderRadius: 16,
    },
    placeholderImage: {
        width: '100%',
        height: '100%',
        borderRadius: 16,
        backgroundColor: '#F2F2F7',
        alignItems: 'center',
        justifyContent: 'center',
    },
    section: {
        marginBottom: 24,
    },
    sectionHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
    },
    sectionLabel: {
        fontSize: 16,
        fontWeight: '600',
        color: '#1C1C1E',
        marginBottom: 12,
    },
    allLink: {
        fontSize: 14,
        color: '#8E8E93',
    },
    // Type
    typeGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
    },
    typeChip: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: '#F2F2F7',
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    typeChipSelected: {
        backgroundColor: '#E5E5EA',
        borderColor: '#C7C7CC',
    },
    typeChipText: {
        fontSize: 14,
        color: '#1C1C1E',
        fontWeight: '500',
    },
    typeChipTextSelected: {
        fontWeight: '600',
    },
    // Color
    colorScrollView: {
        marginHorizontal: -4,
    },
    colorRow: {
        flexDirection: 'row',
        gap: 10,
        paddingVertical: 4,
        paddingHorizontal: 4,
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
    // Season
    seasonRow: {
        flexDirection: 'row',
        gap: 8,
    },
    seasonChip: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: '#F2F2F7',
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    seasonChipSelected: {
        backgroundColor: '#E5E5EA',
        borderColor: '#C7C7CC',
    },
    seasonChipText: {
        fontSize: 14,
        color: '#1C1C1E',
        fontWeight: '500',
    },
    seasonChipTextSelected: {
        fontWeight: '600',
    },
    // Weather
    weatherRow: {
        flexDirection: 'row',
        gap: 12,
    },
    weatherCircle: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: '#F2F2F7',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    weatherCircleSelected: {
        backgroundColor: '#1C1C1E',
        borderColor: '#1C1C1E',
    },
    // AI Banner
    aiBanner: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#EEF4FF',
        borderRadius: 12,
        paddingHorizontal: 12,
        paddingVertical: 8,
        marginBottom: 12,
        flexWrap: 'wrap',
    },
    aiBannerText: {
        fontSize: 13,
        fontWeight: '600',
        color: '#007AFF',
    },
    aiBannerBadge: {
        backgroundColor: '#007AFF',
        borderRadius: 8,
        paddingHorizontal: 8,
        paddingVertical: 2,
        marginLeft: 8,
    },
    aiBannerBadgeText: {
        fontSize: 11,
        fontWeight: '700',
        color: '#FFFFFF',
    },
    aiBannerMaterial: {
        fontSize: 13,
        color: '#3C3C43',
        marginLeft: 2,
    },
    aiDescription: {
        fontSize: 13,
        color: '#636366',
        marginBottom: 16,
        lineHeight: 18,
        fontStyle: 'italic',
    },
    // Save Button
    saveButtonContainer: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        paddingHorizontal: 24,
        paddingBottom: 34,
        paddingTop: 16,
        backgroundColor: '#FFFFFF',
        borderTopWidth: 1,
        borderTopColor: '#F2F2F7',
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
});

export default ClothingDetailEditor;
