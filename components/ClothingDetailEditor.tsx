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
import { useNavigation, useRoute } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import useWardrobeStore from '../store/wardrobeStore';
import useAuthStore from '../store/auth';
import type { Season, Occasion } from '../src/types/domain';
import { ExternalAIService } from '../src/services/externalAIService';
import * as FileSystem from 'expo-file-system';

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

const TYPES = [
    { id: 'tops', label: 'Tops' },
    { id: 'bottoms', label: 'Bottoms' },
    { id: 'shoes', label: 'Shoes' },
    { id: 'accessories', label: 'Accessories' },
    { id: 'outerwear', label: 'Outerwear' },
    { id: 'sportswear', label: 'Sportswear' },
    { id: 'homewear', label: 'Homewear' },
];

const COLORS = [
    { id: 'black', label: 'Black', hex: '#1C1C1E' },
    { id: 'grey', label: 'Grey', hex: '#8E8E93' },
    { id: 'beige', label: 'Beige', hex: '#C7B299' },
    { id: 'white', label: 'White', hex: '#FFFFFF' },
    { id: 'brown', label: 'Brown', hex: '#8B4513' },
    { id: 'green', label: 'Green', hex: '#34C759' },
    { id: 'red', label: 'Red', hex: '#FF3B30' },
    { id: 'blue', label: 'Blue', hex: '#007AFF' },
];

const SEASONS = [
    { id: 'summer', label: 'Summer' },
    { id: 'winter', label: 'Winter' },
    { id: 'autumn', label: 'Autumn' },
    { id: 'spring', label: 'Spring' },
];

const WEATHER = [
    { id: 'rain', icon: 'rainy-outline', label: 'Rain' },
    { id: 'sun', icon: 'sunny-outline', label: 'Sunny' },
    { id: 'snow', icon: 'snow-outline', label: 'Snow' },
    { id: 'wind', icon: 'flag-outline', label: 'Wind' },
];

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
    if (s.includes('streetwear') || s.includes('urban')) return ['casual', 'outdoor'] as Occasion[];
    return ['casual'] as Occasion[];
};

const ClothingDetailEditor: React.FC<ClothingDetailEditorProps> = ({
    imageUri,
    initialData,
    onSave,
    onCancel,
}) => {
    const navigation = useNavigation();
    const route = useRoute<any>();

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

                    const mapType = (c: string, s: string) => {
                        if (s === 'tops' || c.includes('shirt') || c.includes('blouse') || c.includes('sweater') || c.includes('hoodie') || c === 't-shirt') return 'tops';
                        if (s === 'bottoms' || c.includes('pant') || c.includes('jean') || c.includes('skirt') || c.includes('short')) return 'bottoms';
                        if (s === 'shoes' || c.includes('shoe') || c.includes('sneaker') || c.includes('boot') || c.includes('sandal')) return 'shoes';
                        if (s === 'accessories' || c.includes('bag') || c.includes('hat') || c.includes('scarf') || c.includes('belt')) return 'accessories';
                        if (s === 'outerwear' || c.includes('jacket') || c.includes('coat')) return 'outerwear';
                        return 'tops';
                    };

                    const mapColor = (colorName: string) => {
                        const n = (colorName || '').toLowerCase();
                        if (n.includes('black') || n.includes('charcoal')) return 'black';
                        if (n.includes('grey') || n.includes('gray') || n.includes('silver')) return 'grey';
                        if (n.includes('beige') || n.includes('cream') || n.includes('tan') || n.includes('khaki')) return 'beige';
                        if (n.includes('white') || n.includes('ivory')) return 'white';
                        if (n.includes('brown') || n.includes('camel')) return 'brown';
                        if (n.includes('green') || n.includes('olive')) return 'green';
                        if (n.includes('red') || n.includes('burgundy') || n.includes('pink')) return 'red';
                        if (n.includes('blue') || n.includes('navy') || n.includes('denim')) return 'blue';
                        return 'beige';
                    };

                    const newType = mapType(cat, sec);
                    const newColor = mapColor(aiResult.classification.attributes?.color || '');
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
            Alert.alert('Login Required', 'Please login to save items.');
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
                'Saved!',
                'Item added to your wardrobe.',
                [{ text: 'OK', onPress: () => navigation.goBack() }]
            );
        } catch (error: any) {
            if (error?.code === 'WARDROBE_LIMIT_REACHED') {
                Alert.alert(
                    'Wardrobe is full',
                    `Free plan keeps closets at ${error.limit ?? 20} items. Upgrade to Pro for an unlimited wardrobe.`,
                    [
                        { text: 'Maybe later', style: 'cancel' },
                        {
                            text: 'Upgrade',
                            style: 'default',
                            onPress: () => (navigation as any).navigate('Paywall'),
                        },
                    ]
                );
                return;
            }
            console.error('Failed to save item:', error);
            Alert.alert('Error', 'Failed to save item. Please try again.');
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
                            <Text style={styles.aiBannerText}>AI is analyzing your clothing...</Text>
                        </View>
                    )}
                    {!isAnalyzing && aiConfidence !== undefined && (
                        <View style={styles.aiBanner}>
                            <Ionicons name="sparkles" size={14} color="#007AFF" style={{ marginRight: 6 }} />
                            <Text style={styles.aiBannerText}>
                                AI Detected
                            </Text>
                            <View style={styles.aiBannerBadge}>
                                <Text style={styles.aiBannerBadgeText}>
                                    {Math.round(aiConfidence * 100)}% confidence
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
                        <Text style={styles.sectionLabel}>Type</Text>
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
                            <Text style={styles.sectionLabel}>Colour</Text>
                            <TouchableOpacity>
                                <Text style={styles.allLink}>All {'>'}</Text>
                            </TouchableOpacity>
                        </View>
                        <View style={styles.colorRow}>
                            {COLORS.map(color => (
                                <TouchableOpacity
                                    key={color.id}
                                    style={[
                                        styles.colorCircle,
                                        { backgroundColor: color.hex },
                                        color.id === 'white' && styles.colorCircleWhite,
                                        selectedColor === color.id && styles.colorCircleSelected,
                                    ]}
                                    onPress={() => handleColorSelect(color.id)}
                                    activeOpacity={0.7}
                                >
                                    {selectedColor === color.id && (
                                        <Ionicons
                                            name="checkmark"
                                            size={16}
                                            color={color.id === 'white' || color.id === 'beige' ? '#1C1C1E' : '#FFFFFF'}
                                        />
                                    )}
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>

                    {/* Season Section */}
                    <View style={styles.section}>
                        <Text style={styles.sectionLabel}>Season</Text>
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
                        <Text style={styles.sectionLabel}>Weather</Text>
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
                        <Text style={styles.saveButtonText}>Save</Text>
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
    colorRow: {
        flexDirection: 'row',
        gap: 12,
    },
    colorCircle: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
    },
    colorCircleWhite: {
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    colorCircleSelected: {
        borderWidth: 3,
        borderColor: '#007AFF',
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
