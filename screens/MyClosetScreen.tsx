import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
    ActivityIndicator,
} from 'react-native';
import { Video, ResizeMode } from 'expo-av';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { supabase } from '../lib/supabase';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';

const { width } = Dimensions.get('window');
const { colors, spacing, radius, typography } = LiquidGlass2026Theme;

// Updated Category filters to match design
const CATEGORIES = [
    { id: 'favorite', label: 'Favorite', icon: 'heart' }, // Filled heart will be handled in render
    { id: 'all', label: 'All', icon: 'grid' },
    { id: 'tops', label: 'Tops', icon: 'shirt' },
    { id: 'bottoms', label: 'Bottoms', icon: 'bookmark' }, // Placeholder icon
    { id: 'shoes', label: 'Shoes', icon: 'footsteps' },
    { id: 'accessories', label: 'Accessories', icon: 'glasses' },
];

interface ClothingItem {
    _id: string;
    id?: string;
    type?: string;
    itemType?: string;
    color?: string;
    colorHex?: string;
    style?: string;
    description?: string;
    imageUrl?: string;
    image?: string;
    category?: string;
    wearCount?: number;
    lastWorn?: string;
    createdAt?: string;
    isFavorite?: boolean; // Added for favorite filter
}

// Category Chip from Design
const FilterChip = ({
    label,
    isSelected,
    onPress
}: {
    label: string;
    isSelected: boolean;
    onPress: () => void;
}) => {
    return (
        <TouchableOpacity
            style={[
                styles.filterChip,
                isSelected ? styles.filterChipSelected : styles.filterChipUnselected
            ]}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={0.8}
        >
            <Text style={[
                styles.filterChipText,
                isSelected ? styles.filterChipTextSelected : styles.filterChipTextUnselected
            ]}>
                {label}
            </Text>
        </TouchableOpacity>
    );
};

// Premium Clothing Grid Item
const ClothingGridItem = ({
    item,
    onPress
}: {
    item: ClothingItem;
    onPress: () => void;
}) => {
    const imageUrl = item.imageUrl || item.image;

    return (
        <TouchableOpacity
            style={styles.gridItemContainer}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={0.9}
        >
            <Animated.View style={styles.gridItem}>
                <View style={styles.imageContainer}>
                    {imageUrl ? (
                        <Image
                            source={{ uri: imageUrl }}
                            style={styles.itemImage}
                            resizeMode="cover"
                        />
                    ) : (
                        <View style={styles.itemPlaceholder}>
                            <Ionicons name="shirt-outline" size={32} color={colors.text.tertiary} />
                        </View>
                    )}
                </View>
            </Animated.View>
        </TouchableOpacity>
    );
};

const MyClosetScreen = () => {
    const navigation = useNavigation();
    const [items, setItems] = useState<ClothingItem[]>([]);
    const [filteredItems, setFilteredItems] = useState<ClothingItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [selectedCategory, setSelectedCategory] = useState('tops'); // Default from screenshot
    const [viewMode, setViewMode] = useState<'clothes' | 'collections'>('clothes');

    // Load wardrobe items
    const loadItems = useCallback(async () => {
        try {
            setLoading(true);

            // Fetch from Supabase
            const { data, error } = await supabase
                .from('clothing_items')
                .select('*')
                .order('created_at', { ascending: false });

            if (data) {
                // Map snake_case to camelCase
                const mappedItems: ClothingItem[] = data.map(item => ({
                    _id: item.id,
                    id: item.id,
                    type: item.type,
                    itemType: item.type,
                    color: item.color && item.color.length > 0 ? item.color[0] : 'various',
                    imageUrl: item.image_url,
                    image: item.image_url,
                    category: item.category,
                    wearCount: item.wear_count,
                    createdAt: item.created_at,
                    isFavorite: false, // Default for now
                }));

                setItems(mappedItems);
            }
        } catch (error) {
            console.error('Failed to load wardrobe:', error);
            // local storage
            const localItems = await AsyncStorage.getItem('myWardrobeItems');
            if (localItems) {
                setItems(JSON.parse(localItems));
            }
        } finally {
            setLoading(false);
        }
    }, []);

    useFocusEffect(
        useCallback(() => {
            loadItems();
        }, [loadItems])
    );

    // Filter items
    useEffect(() => {
        let result = [...items];

        if (viewMode === 'clothes') {
            if (selectedCategory !== 'all') {
                if (selectedCategory === 'favorite') {
                    result = result.filter(item => item.isFavorite);
                } else {
                    result = result.filter(item => {
                        const category = (item.category || item.type || '').toLowerCase();
                        // Simple inclusion check for now, can be robustified
                        return category.includes(selectedCategory.replace('s', '')) || // remove plural s for matching
                            (selectedCategory === 'tops' && (category.includes('shirt') || category.includes('blouse'))) ||
                            (selectedCategory === 'bottoms' && (category.includes('pant') || category.includes('skirt') || category.includes('jean')));
                    });
                }
            }
        }

        setFilteredItems(result);
    }, [items, selectedCategory, viewMode]);


    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea} edges={['top']}>
                {/* Header */}
                <View style={[styles.header, items.length === 0 && { justifyContent: 'center' }]}>
                    <View style={[StyleSheet.absoluteFillObject, { alignItems: 'center', justifyContent: 'center' }]} pointerEvents="none">
                        <Text style={styles.headerTitle}>My Closet</Text>
                    </View>
                    {items.length > 0 && (
                        <TouchableOpacity style={styles.headerButtonLeft}>
                            <Ionicons name="search" size={20} color={colors.text.secondary} />
                            <Text style={styles.headerButtonText}>Search</Text>
                        </TouchableOpacity>
                    )}


                    {items.length > 0 && (
                        <TouchableOpacity
                            style={styles.headerButtonRight}
                            onPress={() => (navigation as any).navigate('Camera')} // Navigate to Camera/Upload
                        >
                            <Ionicons name="add" size={22} color={colors.text.secondary} />
                            <Text style={styles.headerButtonText}>Upload</Text>
                        </TouchableOpacity>
                    )}
                </View>

                {/* Segmented Control */}
                <View style={styles.segmentContainer}>
                    <View style={styles.segmentBackground}>
                        <TouchableOpacity
                            style={[styles.segmentButton, viewMode === 'clothes' && styles.segmentButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setViewMode('clothes');
                            }}
                        >
                            <Text style={[styles.segmentText, viewMode === 'clothes' && styles.segmentTextActive]}>Clothes</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.segmentButton, viewMode === 'collections' && styles.segmentButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setViewMode('collections');
                            }}
                        >
                            <Text style={[styles.segmentText, viewMode === 'collections' && styles.segmentTextActive]}>Collections</Text>
                        </TouchableOpacity>
                    </View>
                </View>

                {/* Filter Chips */}
                {viewMode === 'clothes' && (
                    <View style={styles.filterContainer}>
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.filterContentRaw}
                        >
                            {CATEGORIES.map((cat) => (
                                <FilterChip
                                    key={cat.id}
                                    label={cat.label}
                                    isSelected={selectedCategory === cat.id}
                                    onPress={() => setSelectedCategory(cat.id)}
                                />
                            ))}
                        </ScrollView>
                    </View>
                )}

                {/* Content */}
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {loading ? (
                        <View style={styles.loadingContainer}>
                            <ActivityIndicator size="small" color={colors.text.primary} />
                        </View>
                    ) : items.length === 0 ? (
                        <View style={styles.emptyStateContainer}>
                            <View style={styles.videoContainer}>
                                <Video
                                    source={require('../assets/videos/closet.mov')}
                                    style={styles.video}
                                    resizeMode={ResizeMode.CONTAIN}
                                    shouldPlay={true}
                                    isLooping
                                    isMuted
                                />
                            </View>
                            <Text style={styles.emptyTitle}>Your closet is empty</Text>
                            <Text style={styles.emptySubtitle}>Start adding items to build your digital wardrobe.</Text>

                            <TouchableOpacity
                                style={styles.emptyButton}
                                onPress={() => (navigation as any).navigate('Camera')}
                            >
                                <Text style={styles.emptyButtonText}>Scan Wardrobe</Text>
                            </TouchableOpacity>
                        </View>
                    ) : (
                        <View style={styles.grid}>
                            {filteredItems.map((item, index) => (
                                <ClothingGridItem
                                    key={item._id || item.id || index}
                                    item={item}
                                    onPress={() => {
                                        (navigation as any).navigate('OutfitDetail', {
                                            image: item.imageUrl || item.image,
                                            outfit: { id: item._id || item.id, items: [item] }
                                        });
                                    }}
                                />
                            ))}
                        </View>
                    )}
                    <View style={{ height: 100 }} />
                </ScrollView>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#FFFFFF', // iOS System Gray 6 (light mode typical background)
    },
    safeArea: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 12,
        backgroundColor: '#FFFFFF',
    },
    headerTitle: {
        ...typography.scale.titleLarge,
        fontWeight: '700',
        color: '#0A1931',
        letterSpacing: 0.3,
    },
    headerButtonLeft: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FFFFFF',
        paddingVertical: 8,
        paddingHorizontal: 16,
        borderRadius: 20, // Pill shape
        // Subtle shadow
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 1,
    },
    headerButtonRight: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FFFFFF',
        paddingVertical: 8,
        paddingHorizontal: 16,
        borderRadius: 20, // Pill shape
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 1,

    },
    headerButtonText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#3C3C43', // iOS gray
        marginLeft: 6,
    },

    // Segmented Control
    segmentContainer: {
        alignItems: 'center',
        paddingTop: 15,
        paddingBottom: 20,
        backgroundColor: '#FFFFFF',
    },
    segmentBackground: {
        flexDirection: 'row',
        backgroundColor: '#E5E5EA', // iOS System Gray 5
        borderRadius: 24, // Rounded
        padding: 4,
        width: 300,
    },
    segmentButton: {
        flex: 1,
        paddingVertical: 10,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 20,
    },
    segmentButtonActive: {
        backgroundColor: '#FFFFFF',
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
        elevation: 2,
    },
    segmentText: {
        fontSize: 15,
        fontWeight: '500',
        color: '#8E8E93',
    },
    segmentTextActive: {
        color: '#0A1931',
        fontWeight: '600',
    },

    // Filters
    filterContainer: {
        marginBottom: 16,
    },
    filterContentRaw: {
        paddingHorizontal: 16,
        paddingRight: 8,
    },
    filterChip: {
        paddingVertical: 8,
        paddingHorizontal: 20,
        borderRadius: 20,
        marginRight: 8,
        backgroundColor: '#E5E5EA', // Default gray
    },
    filterChipSelected: {
        backgroundColor: '#303030', // Dark grey/black active state
    },
    filterChipUnselected: {
        backgroundColor: '#E5E5EA',
    },
    filterChipText: {
        fontSize: 15,
        fontWeight: '500',
    },
    filterChipTextSelected: {
        color: '#FFFFFF',
    },
    filterChipTextUnselected: {
        color: '#3C3C43',
    },

    // Grid
    scrollContent: {
        paddingHorizontal: 16,
        paddingBottom: 100,
        flexGrow: 1,
    },
    loadingContainer: {
        paddingTop: 50,
        alignItems: 'center',
    },
    grid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginHorizontal: -6, // Account for item padding
    },
    gridItemContainer: {
        width: '50%', // 2 columns
        padding: 6,
    },
    gridItem: {
        backgroundColor: '#FFFFFF',
        borderRadius: 16,
        overflow: 'hidden',
        // Minimal shadow for clean look
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 1 },
        shadowOpacity: 0.05,
        shadowRadius: 2,
        elevation: 1,
    },
    imageContainer: {
        aspectRatio: 3 / 4, // Portrait ratio for clothes
        backgroundColor: '#FFFFFF',
        alignItems: 'center',
        justifyContent: 'center',
    },
    itemImage: {
        width: '100%',
        height: '100%',
    },
    itemPlaceholder: {
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Empty State
    emptyStateContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingTop: 40,
    },
    videoContainer: {
        width: 250,
        height: 250,
        marginBottom: 20,
    },
    video: {
        width: '100%',
        height: '100%',
    },
    emptyTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: '#0A1931',
        marginBottom: 8,
    },
    emptySubtitle: {
        fontSize: 15,
        color: '#8E8E93',
        textAlign: 'center',
        maxWidth: 260,
        marginBottom: 24,
    },
    emptyButton: {
        backgroundColor: '#0A1931',
        paddingVertical: 14,
        paddingHorizontal: 32,
        borderRadius: 30,
    },
    emptyButtonText: {
        color: '#FFF',
        fontSize: 16,
        fontWeight: '600',
    },
});

export default MyClosetScreen;
