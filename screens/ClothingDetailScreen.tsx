import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    ScrollView,
    Image,
    StatusBar,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import type { NativeStackNavigationProp } from '@react-navigation/native-stack';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';
import useWardrobeStore from '../store/wardrobeStore';
import useAuthStore from '../store/auth';
import type { Season, Occasion } from '../src/types/domain';
import { RootStackParamList } from '../navigation/types';
import { BASIC_CLOTHING_ITEMS } from '../data/basicClothingItems';

type ClothingDetailRouteProp = RouteProp<RootStackParamList, 'ClothingDetail'>;

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
    { id: 'rain', icon: 'rainy-outline' as const, label: 'Rain' },
    { id: 'sun', icon: 'sunny-outline' as const, label: 'Sunny' },
    { id: 'snow', icon: 'snow-outline' as const, label: 'Snow' },
    { id: 'wind', icon: 'flag-outline' as const, label: 'Wind' },
];

const ClothingDetailScreen: React.FC = () => {
    const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
    const route = useRoute<ClothingDetailRouteProp>();

    const { itemId, fullItem } = route.params ?? {};

    const imageUri: string | undefined =
        fullItem?.imageUrl || fullItem?.image || fullItem?.imageUri;
    const detectedType: string = fullItem?.category || fullItem?.type || 'outerwear';
    const detectedColor: string = fullItem?.primaryColor?.toLowerCase() || fullItem?.color || 'beige';
    const detectedSeason: string =
        (Array.isArray(fullItem?.seasons) && fullItem.seasons[0]) ||
        fullItem?.season ||
        'winter';

    const [selectedType, setSelectedType] = useState(detectedType);
    const [selectedColor, setSelectedColor] = useState(detectedColor);
    const [selectedSeason, setSelectedSeason] = useState(detectedSeason);
    const [selectedWeather, setSelectedWeather] = useState<string[]>(['snow', 'wind']);

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

    const handleBuildOutfit = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

        const anchorId = String(fullItem?.id || fullItem?._id || itemId || '');
        if (!anchorId) {
            Alert.alert('Unable to build outfit', 'Please save this item first, then try again.');
            return;
        }

        navigation.navigate('AIOutfit', {
            source: 'wardrobe',
            baseItemId: anchorId,
            baseItem: {
                id: anchorId,
                imageUrl: imageUri,
                name: fullItem?.name || fullItem?.type || selectedType,
                type: fullItem?.type || fullItem?.subCategory || selectedType,
                macroCategory: fullItem?.macroCategory,
                color: fullItem?.primaryColor || fullItem?.color || selectedColor,
            },
        });
    };

    const handleSave = async () => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

        const { user } = useAuthStore.getState();
        if (!user) {
            Alert.alert('Login Required', 'Please login to save items.');
            return;
        }

        const selectedColorData = COLORS.find(c => c.id === selectedColor) || COLORS[2];

        if (fullItem?.id) {
            const updateItem = useWardrobeStore.getState().updateItem;
            if (updateItem) {
                try {
                    await updateItem(fullItem.id, {
                        category: selectedType as any,
                        subCategory: selectedType,
                        primaryColor: selectedColorData.label,
                        colorHex: selectedColorData.hex,
                        seasons: [selectedSeason] as Season[],
                    });
                    Alert.alert('Saved!', 'Item updated successfully.', [
                        { text: 'OK', onPress: () => navigation.goBack() },
                    ]);
                    return;
                } catch (error) {
                    console.error('Failed to update item:', error);
                    Alert.alert('Error', 'Failed to update item. Please try again.');
                    return;
                }
            }
        }

        const addItem = useWardrobeStore.getState().addItem;
        try {
            await addItem({
                userId: user.id,
                imageUrl: imageUri || '',
                category: selectedType as any,
                subCategory: selectedType,
                primaryColor: selectedColorData.label,
                colorHex: selectedColorData.hex,
                pattern: 'solid',
                material: '',
                brand: '',
                name: `${selectedColorData.label} ${selectedType}`,
                seasons: [selectedSeason] as Season[],
                occasions: ['casual'] as Occasion[],
            });
            Alert.alert('Saved!', 'Item added to your wardrobe.', [
                { text: 'OK', onPress: () => navigation.goBack() },
            ]);
        } catch (error) {
            console.error('Failed to save item:', error);
            Alert.alert('Error', 'Failed to save item. Please try again.');
        }
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
                style={StyleSheet.absoluteFill}
                pointerEvents="none"
            />
            <View pointerEvents="none" style={styles.backgroundOrbTop} />
            <View pointerEvents="none" style={styles.backgroundOrbBottom} />
            <StatusBar barStyle="dark-content" />
            <SafeAreaView style={styles.safeArea}>
                {/* Close button */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => navigation.goBack()} style={styles.closeButton}>
                        <Ionicons name="close" size={22} color="#1C1C1E" />
                    </TouchableOpacity>
                </View>

        <ScrollView
            style={styles.scrollView}
            contentContainerStyle={styles.content}
            showsVerticalScrollIndicator={false}
        >
            {/* Clothing Image */}
            <View style={styles.imageContainer}>
                {(() => {
                    let finalSource: { uri: string } | null = imageUri ? { uri: imageUri } : null;
                    if (imageUri && imageUri.startsWith('basic_clothing_')) {
                        const basicId = imageUri.replace('basic_clothing_', '');
                        const basicItem = BASIC_CLOTHING_ITEMS.find(b => b.id === basicId);
                        if (basicItem && basicItem.image) finalSource = { uri: basicItem.image };
                    }
                    return finalSource ? (
                        <Image
                            source={finalSource}
                            style={styles.clothingImage}
                            resizeMode="contain"
                        />
                    ) : (
                        <View style={styles.placeholderImage}>
                            <Ionicons name="shirt-outline" size={72} color="#C7C7CC" />
                        </View>
                    );
                })()}
            </View>

                    {/* Type Section */}
                    <View style={styles.section}>
                        <Text style={styles.sectionLabel}>Type</Text>
                        <View style={styles.chipGrid}>
                            {TYPES.map(type => (
                                <TouchableOpacity
                                    key={type.id}
                                    style={[
                                        styles.chip,
                                        selectedType === type.id && styles.chipSelected,
                                    ]}
                                    onPress={() => handleTypeSelect(type.id)}
                                    activeOpacity={0.7}
                                >
                                    <Text
                                        style={[
                                            styles.chipText,
                                            selectedType === type.id && styles.chipTextSelected,
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
                        <View style={styles.sectionHeaderRow}>
                            <Text style={styles.sectionLabel}>Colour</Text>
                            <TouchableOpacity style={styles.allLinkRow}>
                                <Text style={styles.allLinkText}>All</Text>
                                <Ionicons name="chevron-forward" size={14} color="#8E8E93" />
                            </TouchableOpacity>
                        </View>
                        <View style={styles.colorGrid}>
                            {COLORS.map(color => (
                                <View key={color.id} style={styles.colorItem}>
                                    <TouchableOpacity
                                        style={[
                                            styles.colorCircle,
                                            { backgroundColor: color.hex },
                                            color.id === 'white' && styles.colorCircleBorder,
                                            selectedColor === color.id && styles.colorCircleSelected,
                                        ]}
                                        onPress={() => handleColorSelect(color.id)}
                                        activeOpacity={0.7}
                                    >
                                        {selectedColor === color.id && (
                                            <Ionicons
                                                name="checkmark"
                                                size={14}
                                                color={
                                                    color.id === 'white' || color.id === 'beige'
                                                        ? '#1C1C1E'
                                                        : '#FFFFFF'
                                                }
                                            />
                                        )}
                                    </TouchableOpacity>
                                    <Text style={styles.colorLabel}>{color.label}</Text>
                                </View>
                            ))}
                        </View>
                    </View>

                    {/* Season Section */}
                    <View style={styles.section}>
                        <Text style={styles.sectionLabel}>Season</Text>
                        <View style={styles.chipRow}>
                            {SEASONS.map(season => (
                                <TouchableOpacity
                                    key={season.id}
                                    style={[
                                        styles.chip,
                                        selectedSeason === season.id && styles.chipSelected,
                                    ]}
                                    onPress={() => handleSeasonSelect(season.id)}
                                    activeOpacity={0.7}
                                >
                                    <Text
                                        style={[
                                            styles.chipText,
                                            selectedSeason === season.id && styles.chipTextSelected,
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
                                            name={weather.icon}
                                            size={22}
                                            color={isSelected ? '#FFFFFF' : '#8E8E93'}
                                        />
                                    </TouchableOpacity>
                                );
                            })}
                        </View>
                    </View>

                    <View style={{ height: 110 }} />
                </ScrollView>

                {/* Action Buttons */}
                <View style={styles.saveContainer}>
                    <TouchableOpacity
                        style={styles.buildOutfitButton}
                        onPress={handleBuildOutfit}
                        activeOpacity={0.85}
                        accessibilityRole="button"
                        accessibilityLabel="Build outfit with this item"
                    >
                        <Ionicons name="sparkles" size={18} color="#1C1C1E" />
                        <Text style={styles.buildOutfitButtonText}>Build Outfit with This</Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.saveButton}
                        onPress={handleSave}
                        activeOpacity={0.85}
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
    },
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
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        paddingHorizontal: 20,
        paddingTop: 6,
        paddingBottom: 4,
    },
    closeButton: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 12,
        elevation: 3,
    },
    scrollView: {
        flex: 1,
    },
    content: {
        paddingHorizontal: 20,
        paddingTop: 4,
    },
    imageContainer: {
        alignItems: 'center',
        height: 260,
        marginBottom: 28,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderRadius: 30,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.08,
        shadowRadius: 18,
        elevation: 4,
        paddingVertical: 12,
    },
    clothingImage: {
        width: '80%',
        height: '100%',
    },
    placeholderImage: {
        width: '80%',
        height: '100%',
        borderRadius: 16,
        backgroundColor: '#F2F2F7',
        alignItems: 'center',
        justifyContent: 'center',
    },
    section: {
        marginBottom: 22,
    },
    sectionLabel: {
        fontSize: 15,
        fontWeight: '500',
        color: '#8E8E93',
        marginBottom: 12,
    },
    sectionHeaderRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 12,
    },
    allLinkRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 2,
    },
    allLinkText: {
        fontSize: 14,
        color: '#8E8E93',
    },
    // Chips (Type & Season)
    chipGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
    },
    chipRow: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 8,
    },
    chip: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
    },
    chipSelected: {
        backgroundColor: '#173A65',
        borderColor: '#173A65',
    },
    chipText: {
        fontSize: 14,
        color: '#1C1C1E',
        fontWeight: '400',
    },
    chipTextSelected: {
        fontWeight: '600',
        color: '#FFFFFF',
    },
    // Colors — 4 per row with label
    colorGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 0,
        rowGap: 12,
    },
    colorItem: {
        width: '25%',
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        marginBottom: 4,
    },
    colorCircle: {
        width: 28,
        height: 28,
        borderRadius: 14,
        alignItems: 'center',
        justifyContent: 'center',
    },
    colorCircleBorder: {
        borderWidth: 1,
        borderColor: '#E5E5EA',
    },
    colorCircleSelected: {
        borderWidth: 2.5,
        borderColor: '#1C1C1E',
    },
    colorLabel: {
        fontSize: 13,
        color: '#1C1C1E',
        fontWeight: '400',
    },
    // Weather
    weatherRow: {
        flexDirection: 'row',
        gap: 12,
    },
    weatherCircle: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: 'rgba(255,255,255,0.88)',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
    },
    weatherCircleSelected: {
        backgroundColor: '#173A65',
        borderColor: '#173A65',
    },
    // Save Button
    saveContainer: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        paddingHorizontal: 20,
        paddingBottom: 32,
        paddingTop: 12,
        backgroundColor: 'rgba(255,255,255,0.92)',
        gap: 10,
    },
    buildOutfitButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        backgroundColor: 'rgba(255,255,255,0.88)',
        paddingVertical: 16,
        borderRadius: 24,
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
    },
    buildOutfitButtonText: {
        color: '#1C1C1E',
        fontSize: 16,
        fontWeight: '600',
        letterSpacing: 0.2,
    },
    saveButton: {
        backgroundColor: '#173A65',
        paddingVertical: 17,
        borderRadius: 24,
        alignItems: 'center',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.12,
        shadowRadius: 16,
        elevation: 4,
    },
    saveButtonText: {
        color: '#FFFFFF',
        fontSize: 17,
        fontWeight: '600',
        letterSpacing: 0.2,
    },
});

export default ClothingDetailScreen;
