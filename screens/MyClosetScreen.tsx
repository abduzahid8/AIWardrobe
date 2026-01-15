import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
    TextInput,
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
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import axios from 'axios';
import { TahoeIconButton } from '../components/TahoeButton';
import AppColors from '../constants/AppColors';
// @ts-ignore
import { API_URL } from '../api/config';

const { width } = Dimensions.get('window');

// Use unified AppColors
const COLORS = {
    background: AppColors.background,
    surface: AppColors.surface,
    surfaceLight: AppColors.surfaceSecondary,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    textMuted: AppColors.textMuted,
    border: AppColors.border,
    success: '#34C759',
    warning: '#FF9500',
    error: '#FF3B30',
};

// Category filters
const CATEGORIES = [
    { id: 'all', label: 'All', icon: 'grid-outline' },
    { id: 'tops', label: 'Tops', icon: 'shirt-outline' },
    { id: 'bottoms', label: 'Bottoms', icon: 'bookmark-outline' },
    { id: 'dresses', label: 'Dresses', icon: 'woman-outline' },
    { id: 'outerwear', label: 'Outerwear', icon: 'snow-outline' },
    { id: 'shoes', label: 'Shoes', icon: 'footsteps-outline' },
    { id: 'accessories', label: 'Accessories', icon: 'glasses-outline' },
];

// Color filters
const COLOR_FILTERS = [
    { id: 'all', name: 'All', hex: null },
    { id: 'black', name: 'Black', hex: '#000000' },
    { id: 'white', name: 'White', hex: '#FFFFFF' },
    { id: 'navy', name: 'Navy', hex: '#1B3A57' },
    { id: 'gray', name: 'Gray', hex: '#808080' },
    { id: 'brown', name: 'Brown', hex: '#8B4513' },
    { id: 'red', name: 'Red', hex: '#DC143C' },
    { id: 'blue', name: 'Blue', hex: '#4169E1' },
    { id: 'green', name: 'Green', hex: '#228B22' },
    { id: 'pink', name: 'Pink', hex: '#FF69B4' },
];

// Type for category item
interface CategoryType {
    id: string;
    label: string;
    icon: string;
}

// Type for color filter
interface ColorFilterType {
    id: string;
    name: string;
    hex: string | null;
}

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
}

// Category Chip Component
const CategoryChip = ({
    category,
    isSelected,
    onPress
}: {
    category: CategoryType;
    isSelected: boolean;
    onPress: () => void;
}) => {
    return (
        <TouchableOpacity
            style={[styles.categoryChip, isSelected && styles.categoryChipSelected]}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={0.7}
        >
            <Ionicons
                name={category.icon as any}
                size={16}
                color={isSelected ? COLORS.background : COLORS.textSecondary}
            />
            <Text style={[
                styles.categoryChipText,
                isSelected && styles.categoryChipTextSelected
            ]}>
                {category.label}
            </Text>
        </TouchableOpacity>
    );
};

// Color Chip Component
const ColorChip = ({
    color,
    isSelected,
    onPress
}: {
    color: ColorFilterType;
    isSelected: boolean;
    onPress: () => void;
}) => {
    return (
        <TouchableOpacity
            style={[styles.colorChip, isSelected && styles.colorChipSelected]}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={0.7}
        >
            {color.hex ? (
                <View style={[styles.colorDot, { backgroundColor: color.hex }]} />
            ) : (
                <Ionicons name="color-palette-outline" size={14} color={COLORS.textSecondary} />
            )}
            <Text style={[
                styles.colorChipText,
                isSelected && styles.colorChipTextSelected
            ]}>
                {color.name}
            </Text>
        </TouchableOpacity>
    );
};

// Bento Grid Item Component
const BentoItem = ({
    title,
    value,
    icon,
    color,
    style,
    size = 'normal'
}: {
    title: string;
    value: string | number;
    icon: string;
    color: string;
    style?: Record<string, string | number>;
    size?: 'normal' | 'large';
}) => {
    return (
        <View style={[
            styles.bentoItem,
            { backgroundColor: color + '10', borderColor: color + '20' },
            style
        ]}>
            <View style={[styles.bentoIcon, { backgroundColor: color + '20' }]}>
                <Ionicons name={icon as any} size={size === 'large' ? 24 : 18} color={color} />
            </View>
            <View>
                <Text style={[styles.bentoValue, size === 'large' && styles.bentoValueLarge]}>
                    {value}
                </Text>
                <Text style={styles.bentoTitle}>{title}</Text>
            </View>
        </View>
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
    const itemType = item.type || item.itemType || 'Item';

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
                            <Ionicons name="shirt-outline" size={32} color={COLORS.textMuted} />
                        </View>
                    )}
                    {item.colorHex && (
                        <View style={[styles.colorBadge, { backgroundColor: item.colorHex }]} />
                    )}
                </View>

                <View style={styles.itemInfo}>
                    <Text style={styles.itemType} numberOfLines={1}>{itemType}</Text>
                    <Text style={styles.itemCategory} numberOfLines={1}>
                        {item.category || 'Uncategorized'}
                    </Text>
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
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedCategory, setSelectedCategory] = useState('all');
    const [selectedColor, setSelectedColor] = useState('all');
    const [showFilters, setShowFilters] = useState(false);

    // Load wardrobe items
    const loadItems = useCallback(async () => {
        try {
            setLoading(true);
            const token = await AsyncStorage.getItem('userToken');

            if (!token) {
                // Load from AsyncStorage as fallback
                const localItems = await AsyncStorage.getItem('myWardrobeItems');
                if (localItems) {
                    const parsed = JSON.parse(localItems);
                    setItems(parsed);
                    setFilteredItems(parsed);
                }
                setLoading(false);
                return;
            }

            const response = await axios.get(`${API_URL}/clothing-items`, {
                headers: { Authorization: `Bearer ${token}` }
            });

            if (response.data) {
                const itemsArray = Array.isArray(response.data) ? response.data : response.data.items || [];
                setItems(itemsArray);
                setFilteredItems(itemsArray);
            }
        } catch (error) {
            console.error('Failed to load wardrobe:', error);
            // Try local storage
            const localItems = await AsyncStorage.getItem('myWardrobeItems');
            if (localItems) {
                const parsed = JSON.parse(localItems);
                setItems(parsed);
                setFilteredItems(parsed);
            }
        } finally {
            setLoading(false);
        }
    }, []);

    // Refresh on focus
    useFocusEffect(
        useCallback(() => {
            loadItems();
        }, [loadItems])
    );

    // Filter items when filters change
    useEffect(() => {
        let result = [...items];

        // Category filter
        if (selectedCategory !== 'all') {
            result = result.filter(item => {
                const category = (item.category || item.type || '').toLowerCase();
                switch (selectedCategory) {
                    case 'tops':
                        return category.includes('shirt') || category.includes('top') ||
                            category.includes('blouse') || category.includes('sweater') ||
                            category.includes('upper');
                    case 'bottoms':
                        return category.includes('pants') || category.includes('jeans') ||
                            category.includes('shorts') || category.includes('skirt');
                    case 'dresses':
                        return category.includes('dress');
                    case 'outerwear':
                        return category.includes('jacket') || category.includes('coat') ||
                            category.includes('blazer');
                    case 'shoes':
                        return category.includes('shoe') || category.includes('sneaker') ||
                            category.includes('boot') || category.includes('footwear');
                    case 'accessories':
                        return category.includes('bag') || category.includes('hat') ||
                            category.includes('scarf') || category.includes('accessory') ||
                            category.includes('belt') || category.includes('glasses');
                    default:
                        return true;
                }
            });
        }

        // Color filter
        if (selectedColor !== 'all') {
            result = result.filter(item => {
                const color = (item.color || '').toLowerCase();
                return color.includes(selectedColor);
            });
        }

        // Search filter
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            result = result.filter(item => {
                const searchFields = [
                    item.type, item.itemType, item.color,
                    item.style, item.description, item.category
                ].filter(Boolean).join(' ').toLowerCase();
                return searchFields.includes(query);
            });
        }

        setFilteredItems(result);
    }, [items, selectedCategory, selectedColor, searchQuery]);

    // Calculate analytics
    const analytics = {
        total: items.length,
        mostWorn: items.reduce((max, item) =>
            (item.wearCount || 0) > (max.wearCount || 0) ? item : max,
            items[0]
        ),
        neverWorn: items.filter(item => !item.wearCount || item.wearCount === 0).length,
        categories: CATEGORIES.slice(1).map(cat => ({
            ...cat,
            count: items.filter(item => {
                const category = (item.category || item.type || '').toLowerCase();
                return category.includes(cat.id) ||
                    (cat.id === 'tops' && (category.includes('shirt') || category.includes('upper'))) ||
                    (cat.id === 'shoes' && (category.includes('sneaker') || category.includes('boot')));
            }).length
        }))
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <Animated.View
                    entering={FadeInDown.delay(50).springify()}
                    style={styles.header}
                >
                    <View>
                        <Text style={styles.headerTitle}>My Closet</Text>
                        <Text style={styles.headerSubtitle}>{items.length} items collected</Text>
                    </View>

                    <TahoeIconButton
                        icon="notifications-outline"
                        onPress={() => { }}
                        color={COLORS.text}
                    />
                </Animated.View>

                {/* Search Bar */}
                <Animated.View
                    entering={FadeInUp.delay(100).springify()}
                    style={styles.searchSection}
                >
                    <View style={styles.searchContainer}>
                        <Ionicons name="search-outline" size={20} color={COLORS.textMuted} />
                        <TextInput
                            style={styles.searchInput}
                            placeholder="Search your wardrobe..."
                            placeholderTextColor={COLORS.textMuted}
                            value={searchQuery}
                            onChangeText={setSearchQuery}
                        />
                        {searchQuery ? (
                            <TouchableOpacity onPress={() => setSearchQuery('')}>
                                <Ionicons name="close-circle" size={20} color={COLORS.textMuted} />
                            </TouchableOpacity>
                        ) : null}
                    </View>
                    <TouchableOpacity
                        style={[styles.filterButton, showFilters && styles.filterButtonActive]}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            setShowFilters(!showFilters);
                        }}
                    >
                        <Ionicons
                            name="options-outline"
                            size={20}
                            color={showFilters ? COLORS.background : COLORS.text}
                        />
                    </TouchableOpacity>
                </Animated.View>

                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {/* Category Filters */}
                    <Animated.View entering={FadeInUp.delay(150).springify()}>
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.categoriesContainer}
                        >
                            {CATEGORIES.map((category) => (
                                <CategoryChip
                                    key={category.id}
                                    category={category}
                                    isSelected={selectedCategory === category.id}
                                    onPress={() => setSelectedCategory(category.id)}
                                />
                            ))}
                        </ScrollView>
                    </Animated.View>

                    {/* Color Filters (expanded) */}
                    {showFilters && (
                        <Animated.View
                            entering={FadeInUp.springify()}
                            style={styles.colorFiltersSection}
                        >
                            <Text style={styles.filterLabel}>Filter by color</Text>
                            <ScrollView
                                horizontal
                                showsHorizontalScrollIndicator={false}
                                contentContainerStyle={styles.colorsContainer}
                            >
                                {COLOR_FILTERS.map((color) => (
                                    <ColorChip
                                        key={color.id}
                                        color={color}
                                        isSelected={selectedColor === color.id}
                                        onPress={() => setSelectedColor(color.id)}
                                    />
                                ))}
                            </ScrollView>
                        </Animated.View>
                    )}

                    {/* Analytics Section - Bento Grid */}
                    <Animated.View
                        entering={FadeInUp.delay(200).springify()}
                        style={styles.analyticsSection}
                    >
                        {/* Closet Video */}
                        <View style={styles.closetVideoContainer}>
                            <Video
                                source={require('./closet.mov')}
                                style={styles.closetVideo}
                                resizeMode={ResizeMode.COVER}
                                shouldPlay={true}
                                isLooping
                                isMuted
                            />
                        </View>

                        {/* Rediscover prompt */}
                        {analytics.neverWorn > 0 && (
                            <TouchableOpacity
                                style={styles.rediscoverCard}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                    // Filter to show never worn items
                                    setFilteredItems(items.filter(item => !item.wearCount || item.wearCount === 0));
                                }}
                            >
                                <View style={styles.rediscoverIcon}>
                                    <Ionicons name="sparkles" size={24} color={COLORS.primary} />
                                </View>
                                <View style={styles.rediscoverContent}>
                                    <Text style={styles.rediscoverTitle}>Rediscover your wardrobe</Text>
                                    <Text style={styles.rediscoverSubtitle}>
                                        You have {analytics.neverWorn} items waiting to be worn!
                                    </Text>
                                </View>
                                <Ionicons name="arrow-forward" size={20} color={COLORS.textMuted} />
                            </TouchableOpacity>
                        )}
                    </Animated.View>

                    {/* Items Grid */}
                    <Animated.View
                        entering={FadeInUp.delay(250).springify()}
                        style={styles.gridSection}
                    >
                        <View style={styles.gridHeader}>
                            <Text style={styles.sectionTitle}>
                                {selectedCategory === 'all' ? 'All Items' :
                                    CATEGORIES.find(c => c.id === selectedCategory)?.label}
                            </Text>
                            <Text style={styles.itemCount}>{filteredItems.length} items</Text>
                        </View>

                        {loading ? (
                            <View style={styles.loadingContainer}>
                                <ActivityIndicator size="large" color={COLORS.primary} />
                                <Text style={styles.loadingText}>Loading your closet...</Text>
                            </View>
                        ) : filteredItems.length === 0 ? (
                            <View style={styles.emptyContainer}>
                                <Ionicons name="shirt-outline" size={64} color={COLORS.textMuted} />
                                <Text style={styles.emptyTitle}>No items found</Text>
                                <Text style={styles.emptySubtitle}>
                                    {searchQuery || selectedCategory !== 'all' || selectedColor !== 'all'
                                        ? 'Try adjusting your filters'
                                        : 'Scan your wardrobe to add items'}
                                </Text>
                                <TouchableOpacity
                                    style={styles.addButton}
                                    onPress={() => (navigation as any).navigate('WardrobeVideo')}
                                >
                                    <Ionicons name="camera-outline" size={20} color={COLORS.background} />
                                    <Text style={styles.addButtonText}>Scan Wardrobe</Text>
                                </TouchableOpacity>
                            </View>
                        ) : (
                            <View style={styles.grid}>
                                {filteredItems.map((item, index) => (
                                    <ClothingGridItem
                                        key={item._id || item.id || index}
                                        item={item}
                                        onPress={() => {
                                            // Navigate to item detail
                                            (navigation as any).navigate('OutfitDetail', {
                                                image: item.imageUrl || item.image,
                                                outfit: { id: item._id || item.id, items: [item] }
                                            });
                                        }}
                                    />
                                ))}
                            </View>
                        )}
                    </Animated.View>

                    <View style={{ height: 100 }} />
                </ScrollView>

                {/* Floating Action Button */}
                <Animated.View entering={FadeInUp.delay(500).springify()} style={styles.fabContainer}>
                    <TouchableOpacity
                        style={styles.fab}
                        onPress={() => (navigation as any).navigate('Camera')}
                        activeOpacity={0.8}
                    >
                        <Ionicons name="camera" size={28} color="#fff" />
                    </TouchableOpacity>
                </Animated.View>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
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
        paddingTop: 10,
        paddingBottom: 16,
    },
    headerTitle: {
        fontSize: 32,
        fontWeight: '800',
        color: COLORS.text,
        letterSpacing: -0.5,
    },
    headerSubtitle: {
        fontSize: 14,
        color: COLORS.textSecondary,
        marginTop: 4,
        fontWeight: '500',
    },

    // Search
    searchSection: {
        flexDirection: 'row',
        paddingHorizontal: 16,
        paddingVertical: 12,
        gap: 10,
    },
    searchContainer: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surfaceLight,
        borderRadius: 12,
        paddingHorizontal: 12,
        paddingVertical: 10,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    searchInput: {
        flex: 1,
        marginLeft: 8,
        fontSize: 15,
        color: COLORS.text,
    },
    filterButton: {
        width: 44,
        height: 44,
        borderRadius: 12,
        backgroundColor: COLORS.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    filterButtonActive: {
        backgroundColor: COLORS.primary,
        borderColor: COLORS.primary,
    },

    scrollContent: {
        paddingTop: 10,
    },

    // Categories
    categoriesContainer: {
        paddingHorizontal: 16,
        gap: 8,
    },
    categoryChip: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 14,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: COLORS.surfaceLight,
        borderWidth: 1,
        borderColor: COLORS.border,
        marginRight: 8,
    },
    categoryChipSelected: {
        backgroundColor: COLORS.primary,
        borderColor: COLORS.primary,
    },
    categoryChipText: {
        marginLeft: 6,
        fontSize: 13,
        fontWeight: '500',
        color: COLORS.textSecondary,
    },
    categoryChipTextSelected: {
        color: COLORS.background,
    },

    // Color Filters
    colorFiltersSection: {
        paddingTop: 16,
        paddingLeft: 16,
    },
    filterLabel: {
        fontSize: 12,
        fontWeight: '600',
        color: COLORS.textSecondary,
        marginBottom: 10,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    colorsContainer: {
        paddingRight: 16,
    },
    colorChip: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 16,
        backgroundColor: COLORS.surfaceLight,
        borderWidth: 1,
        borderColor: COLORS.border,
        marginRight: 8,
    },
    colorChipSelected: {
        borderColor: COLORS.primary,
        backgroundColor: COLORS.primary + '10',
    },
    colorDot: {
        width: 14,
        height: 14,
        borderRadius: 7,
        marginRight: 6,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    colorChipText: {
        fontSize: 12,
        color: COLORS.textSecondary,
    },
    colorChipTextSelected: {
        color: COLORS.primary,
        fontWeight: '600',
    },

    // Analytics
    analyticsSection: {
        paddingHorizontal: 16,
        paddingTop: 24,
        paddingBottom: 16,
    },
    closetVideoContainer: {
        width: '35%',
        height: 100,
        borderRadius: 1,
        overflow: 'hidden',
        alignSelf: 'center',
        marginVertical: 1,
    },
    closetVideo: {
        width: '100%',
        height: '100%',
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 16,
    },
    statsRow: {
        flexDirection: 'row',
        gap: 12,
    },
    statCard: {
        flex: 1,
        backgroundColor: COLORS.surfaceLight,
        borderRadius: 16,
        padding: 16,
        alignItems: 'center',
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    statIconBg: {
        width: 40,
        height: 40,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
    },
    statValue: {
        fontSize: 18,
        fontWeight: '700',
        color: COLORS.text,
    },
    statTitle: {
        fontSize: 11,
        color: COLORS.textSecondary,
        marginTop: 4,
        textAlign: 'center',
    },

    // Rediscover Card
    rediscoverCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.primary + '10',
        borderRadius: 16,
        padding: 16,
        marginTop: 16,
        borderWidth: 1,
        borderColor: COLORS.primary + '30',
    },
    rediscoverIcon: {
        width: 48,
        height: 48,
        borderRadius: 12,
        backgroundColor: COLORS.background,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 12,
    },
    rediscoverContent: {
        flex: 1,
    },
    rediscoverTitle: {
        fontSize: 15,
        fontWeight: '600',
        color: COLORS.text,
    },
    rediscoverSubtitle: {
        fontSize: 13,
        color: COLORS.textSecondary,
        marginTop: 2,
    },

    // Grid Section
    gridSection: {
        paddingHorizontal: 16,
        paddingTop: 8,
    },
    gridHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 16,
    },
    itemCount: {
        fontSize: 14,
        color: COLORS.textSecondary,
        fontWeight: '500',
    },

    // Bento Grid
    bentoGrid: {
        flexDirection: 'row',
        height: 140,
        marginBottom: 20,
    },
    bentoItem: {
        borderRadius: 20,
        padding: 16,
        justifyContent: 'space-between',
        borderWidth: 1,
    },
    bentoIcon: {
        width: 36,
        height: 36,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
    },
    bentoValue: {
        fontSize: 20,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 2,
    },
    bentoValueLarge: {
        fontSize: 32,
        marginBottom: 4,
    },
    bentoTitle: {
        fontSize: 12,
        fontWeight: '600',
        color: COLORS.textSecondary,
    },

    // Grid Items
    grid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 12,
        paddingBottom: 20,
    },
    gridItemContainer: {
        width: (width - 44) / 2,
        marginBottom: 12,
    },
    gridItem: {
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: COLORS.border,
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 8,
        elevation: 2,
    },
    imageContainer: {
        height: 160,
        width: '100%',
        backgroundColor: COLORS.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
    },
    itemImage: {
        width: '100%',
        height: '100%',
    },
    itemPlaceholder: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    colorBadge: {
        position: 'absolute',
        top: 10,
        right: 10,
        width: 20,
        height: 20,
        borderRadius: 10,
        borderWidth: 2,
        borderColor: '#fff',
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.2,
        shadowRadius: 2,
    },
    itemInfo: {
        padding: 12,
    },
    itemType: {
        fontSize: 14,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 2,
    },
    itemCategory: {
        fontSize: 12,
        color: COLORS.textSecondary,
    },

    // Loading & Empty
    loadingContainer: {
        alignItems: 'center',
        paddingVertical: 60,
    },
    loadingText: {
        marginTop: 12,
        fontSize: 14,
        color: COLORS.textSecondary,
    },
    emptyContainer: {
        alignItems: 'center',
        paddingVertical: 60,
    },
    emptyTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: COLORS.text,
        marginTop: 16,
    },
    emptySubtitle: {
        fontSize: 14,
        color: COLORS.textSecondary,
        marginTop: 4,
        textAlign: 'center',
    },
    addButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.primary,
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: 24,
        marginTop: 20,
        gap: 8,
    },
    addButtonText: {
        fontSize: 15,
        fontWeight: '600',
        color: COLORS.background,
    },

    // FAB
    fabContainer: {
        position: 'absolute',
        bottom: 24,
        alignSelf: 'center',
    },
    fab: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: COLORS.text,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 10,
        elevation: 10,
    },
});

export default MyClosetScreen;
