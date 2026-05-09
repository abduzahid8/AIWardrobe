/**
 * ClothingDetailEditor - Post-analyzer edit screen
 * Allows users to verify and edit AI-detected clothing attributes
 */

import React, { useState, useEffect, useRef } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Image,
    StatusBar,
    Alert,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import { useNavigation, useRoute } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import useWardrobeStore from '../store/wardrobeStore';
import useAuthStore from '../store/auth';
import type { Season, Occasion } from '../src/types/domain';
import { ExternalAIService } from '../src/services/externalAIService';
import * as FileSystem from 'expo-file-system';
import { mapCategoryToType, mapColorToId } from '../src/utils/mappingUtils';
interface ClothingDetailEditorProps {
    imageUri?: string;
    initialData?: {
        type?: string;
        color?: string;
        season?: string;
    };
    onSave?: (data: ClothingData) => void;
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

    // Get data from route params or props
    const itemImageUri = route.params?.imageUri || imageUri;
    const detectedType = route.params?.detectedType || initialData?.type || 'tops';
    const detectedColor = route.params?.detectedColor || initialData?.color || 'beige';
    const detectedStyle: string | undefined = route.params?.detectedStyle;
    const detectedMaterial: string | undefined = route.params?.detectedMaterial;
    const aiConfidence: number | undefined = route.params?.aiConfidence;
    const detectedDescription: string | undefined = route.params?.detectedDescription;

    // AI analysis for images opened without pre-detected data
    const [isAnalyzing, setIsAnalyzing] = useState(!route.params?.detectedType && !!itemImageUri);
    const hasRunAnalysis = useRef(false);

    const [selectedType, setSelectedType] = useState(detectedType);
    const [selectedColor, setSelectedColor] = useState(detectedColor);
    const [selectedSeason, setSelectedSeason] = useState(
        route.params?.detectedType
            ? inferSeasonFromType(detectedType, detectedMaterial)
            : 'spring'
    );
    const [selectedWeather, setSelectedWeather] = useState<string[]>(
        route.params?.detectedType
            ? inferWeatherFromType(detectedType, detectedMaterial)
            : ['sun']
    );

    // Live AI analysis when editor opened with just imageUri (no pre-classified data)
    useEffect(() => {
        if (!isAnalyzing || !itemImageUri || hasRunAnalysis.current) return;
        hasRunAnalysis.current = true;

        (async () => {
            try {
                const b64 = await FileSystem.readAsStringAsync(itemImageUri, {
                    encoding: 'base64' as any,
                });
                const aiResult = await ExternalAIService.classifyOnly(b64);
                if (aiResult.success && aiResult.classification) {
                    const cat = aiResult.classification.category.toLowerCase();
                    const sec = aiResult.classification.section.toLowerCase();

                    const newType = mapCategoryToType(cat, sec);
                    const newColor = mapColorToId(aiResult.classification.attributes?.color || '');
                    setSelectedType(newType);
                    setSelectedColor(newColor);
                    setSelectedSeason(inferSeasonFromType(newType, aiResult.classification.attributes?.material));
                    setSelectedWeather(inferWeatherFromType(newType, aiResult.classification.attributes?.material));
                }
            } catch {
                // silently ignore
            } finally {
                setIsAnalyzing(false);
            }
        })();
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

    const handleSave = async () => {
        const data: ClothingData = {
            type: selectedType,
            color: selectedColor,
            season: selectedSeason,
            weather: selectedWeather,
        };

        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

        // If onSave callback provided, use it
        if (onSave) {
            onSave(data);
            return;
        }

        // Otherwise save directly to wardrobe
        const { user } = useAuthStore.getState();
        if (!user) {
            Alert.alert(t('clothingEditor.loginRequired'), t('clothingEditor.pleaseLoginToSave'));
            return;
        }

        const addItem = useWardrobeStore.getState().addItem;
        const selectedColorData = COLORS.find(c => c.id === selectedColor) || COLORS[2];

        try {
            await addItem({
                userId: user.id,
                imageUrl: itemImageUri || '',
                category: selectedType as any,
                subCategory: selectedType,
                primaryColor: selectedColorData.label,
                colorHex: selectedColorData.hex,
                pattern: 'solid',
                material: detectedMaterial || '',
                brand: '',
                name: detectedDescription
                    ? detectedDescription.slice(0, 60)
                    : `${selectedColorData.label} ${selectedType}`,
                seasons: [selectedSeason] as Season[],
                occasions: styleToOccasions(detectedStyle),
            });

            Alert.alert(
                t('clothingEditor.saved'),
                t('clothingEditor.itemAddedToWardrobe'),
                [{ text: t('clothingEditor.ok'), onPress: () => navigation.goBack() }]
            );
        } catch (error: any) {
            if (error?.code === 'WARDROBE_LIMIT_REACHED') {
                Alert.alert(
                    t('clothingEditor.wardrobeFull'),
                    t('clothingEditor.freePlanLimit', { limit: error.limit ?? 20 }),
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

    const handleClose = () => {
        if (onCancel) {
            onCancel();
        } else {
            navigation.goBack();
        }
    };

    const selectedColorData = COLORS.find(c => c.id === selectedColor) || COLORS[2];

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" />
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={handleClose} style={styles.closeButton}>
                        <Ionicons name="close" size={24} color="#1C1C1E" />
                    </TouchableOpacity>
                </View>

                <ScrollView
                    style={styles.scrollView}
                    contentContainerStyle={styles.content}
                    showsVerticalScrollIndicator={false}
                >
                    {/* AI Analysis Banner */}
                    {isAnalyzing && (
                        <View style={styles.aiBanner}>
                            <ActivityIndicator size="small" color="#007AFF" style={{ marginRight: 8 }} />
                            <Text style={styles.aiBannerText}>{t('clothingEditor.aiAnalyzing')}</Text>
                        </View>
                    )}
                    {!isAnalyzing && aiConfidence !== undefined && (
                        <View style={styles.aiBanner}>
                            <Ionicons name="sparkles" size={14} color="#007AFF" style={{ marginRight: 6 }} />
                            <Text style={styles.aiBannerText}>
                                {t('clothingEditor.aiDetected')}
                            </Text>
                            <View style={styles.aiBannerBadge}>
                                <Text style={styles.aiBannerBadgeText}>
                                    {Math.round(aiConfidence * 100)}% {t('clothingEditor.confidence')}
                                </Text>
                            </View>
                            {detectedMaterial ? (
                                <Text style={styles.aiBannerMaterial}> · {detectedMaterial}</Text>
                            ) : null}
                        </View>
                    )}
                    {detectedDescription ? (
                        <Text style={styles.aiDescription}>{detectedDescription}</Text>
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
                        <Text style={styles.sectionLabel}>{t('clothingEditor.type')}</Text>
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
                                    <Text
                                        style={[
                                            styles.typeChipText,
                                            selectedType === type.id && styles.typeChipTextSelected,
                                        ]}
                                    >
                                        {type.label}
                                    </Text>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>

                    {/* Colour Section */}
                    <View style={styles.section}>
                        <View style={styles.sectionHeader}>
                            <Text style={styles.sectionLabel}>{t('clothingEditor.colour')}</Text>
                            <Text style={styles.allLink}>{COLORS.find(c => c.id === selectedColor)?.label || ''}</Text>
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
                        <Text style={styles.sectionLabel}>{t('clothingEditor.seasonLabel')}</Text>
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
                                    <Text
                                        style={[
                                            styles.seasonChipText,
                                            selectedSeason === season.id && styles.seasonChipTextSelected,
                                        ]}
                                    >
                                        {season.label}
                                    </Text>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>

                    {/* Weather Section */}
                    <View style={styles.section}>
                        <Text style={styles.sectionLabel}>{t('clothingEditor.weatherLabel')}</Text>
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

                    {/* Bottom padding for save button */}
                    <View style={{ height: 100 }} />
                </ScrollView>

                {/* Save Button */}
                <View style={styles.saveButtonContainer}>
                    <TouchableOpacity
                        style={styles.saveButton}
                        onPress={handleSave}
                        activeOpacity={0.8}
                    >
                        <Text style={styles.saveButtonText}>{t('clothingEditor.save')}</Text>
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
        justifyContent: 'flex-end',
        paddingHorizontal: 16,
        paddingTop: 8,
        paddingBottom: 8,
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
});

export default ClothingDetailEditor;
