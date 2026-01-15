import React, { useState, useCallback, useMemo } from "react";
import {
    View,
    Text,
    Image,
    ScrollView,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    RefreshControl,
    Modal,
    TextInput,
    Alert,
    FlatList,
    ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    SlideInUp,
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
} from "react-native-reanimated";
import * as Haptics from "expo-haptics";
import { BlurView } from "expo-blur";
import AsyncStorage from "@react-native-async-storage/async-storage";
import { useNavigation, useFocusEffect } from "@react-navigation/native";
import AppColors from "../../../constants/AppColors";

const { width } = Dimensions.get("window");

const ALTA = {
    background: AppColors.background,
    surface: AppColors.surface,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    textMuted: AppColors.textMuted,
    border: AppColors.border,
    accent: AppColors.warning,
    primary: AppColors.primary,
    danger: '#FF453A',
    white: '#FFFFFF',
};

// Clothing type options for editing
const CLOTHING_TYPES = [
    'T-Shirt', 'Shirt', 'Blouse', 'Sweater', 'Hoodie', 'Jacket',
    'Blazer', 'Coat', 'Pants', 'Jeans', 'Shorts', 'Skirt',
    'Dress', 'Suit', 'Hat', 'Cap', 'Scarf', 'Shoes', 'Sneakers',
    'Boots', 'Bag', 'Belt', 'Sunglasses', 'Watch', 'Other'
];

interface WardrobeItem {
    id: string;
    type: string;
    specificType: string;
    color: string;
    colorHex: string;
    image: string;
    dateAdded: string;
    confidence: number;
    outfitId?: number;  // 🎬 Outfit grouping
}

// Main Screen Component
const MyWardrobeScreen = () => {
    const navigation = useNavigation();
    const [items, setItems] = useState<WardrobeItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [selectedItem, setSelectedItem] = useState<WardrobeItem | null>(null);
    const [showEditModal, setShowEditModal] = useState(false);
    const [editType, setEditType] = useState('');
    const [filterCategory, setFilterCategory] = useState<string | null>(null);
    const [filterOutfit, setFilterOutfit] = useState<number | null>(null);  // 🎬 Outfit filter

    // Load items from AsyncStorage
    const loadItems = async () => {
        try {
            const stored = await AsyncStorage.getItem('myWardrobeItems');
            if (stored) {
                const parsed = JSON.parse(stored);
                setItems(parsed);
            }
        } catch (error) {
            console.log('Error loading wardrobe:', error);
        } finally {
            setLoading(false);
        }
    };

    // Save items to AsyncStorage
    const saveItems = async (newItems: WardrobeItem[]) => {
        try {
            await AsyncStorage.setItem('myWardrobeItems', JSON.stringify(newItems));
            setItems(newItems);
        } catch (error) {
            console.log('Error saving wardrobe:', error);
        }
    };

    // Load on focus
    useFocusEffect(
        useCallback(() => {
            loadItems();
        }, [])
    );

    // Refresh handler
    const onRefresh = useCallback(() => {
        setRefreshing(true);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        loadItems().then(() => setRefreshing(false));
    }, []);

    // Delete item
    const handleDelete = (item: WardrobeItem) => {
        Alert.alert(
            'Delete Item',
            `Are you sure you want to delete this ${item.specificType}?`,
            [
                { text: 'Cancel', style: 'cancel' },
                {
                    text: 'Delete',
                    style: 'destructive',
                    onPress: () => {
                        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                        const newItems = items.filter(i => i.id !== item.id);
                        saveItems(newItems);
                    }
                }
            ]
        );
    };

    // Edit item type
    const handleEditType = (newType: string) => {
        if (selectedItem) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
            const newItems = items.map(i =>
                i.id === selectedItem.id
                    ? { ...i, type: newType, specificType: newType }
                    : i
            );
            saveItems(newItems);
            setShowEditModal(false);
            setSelectedItem(null);
        }
    };

    // Filter items by search query
    const filteredItems = useMemo(() => {
        let result = items;

        // Apply search filter
        if (searchQuery.trim()) {
            const query = searchQuery.toLowerCase();
            result = result.filter(item =>
                item.type.toLowerCase().includes(query) ||
                item.specificType.toLowerCase().includes(query) ||
                item.color.toLowerCase().includes(query)
            );
        }

        // Apply category filter
        if (filterCategory) {
            result = result.filter(item =>
                item.type.toLowerCase() === filterCategory.toLowerCase()
            );
        }

        // 🎬 Apply outfit filter
        if (filterOutfit !== null) {
            result = result.filter(item => item.outfitId === filterOutfit);
        }

        return result;
    }, [items, searchQuery, filterCategory, filterOutfit]);

    // Get unique categories for filter pills
    const categories = useMemo(() => {
        const cats = new Set(items.map(i => i.type));
        return Array.from(cats);
    }, [items]);

    // 🎬 Get unique outfit IDs
    const outfits = useMemo(() => {
        const ids = new Set(items.filter(i => i.outfitId).map(i => i.outfitId!));
        return Array.from(ids).sort((a, b) => a - b);
    }, [items]);

    // Item card component
    const ItemCard = ({ item, index }: { item: WardrobeItem; index: number }) => (
        <Animated.View
            entering={FadeInUp.delay(index * 50).springify()}
            style={styles.itemCard}
        >
            <TouchableOpacity
                style={styles.itemImageContainer}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    setSelectedItem(item);
                }}
            >
                <Image source={{ uri: item.image }} style={styles.itemImage} />

                {/* Color indicator */}
                <View style={[styles.colorDot, { backgroundColor: item.colorHex || '#000' }]} />
            </TouchableOpacity>

            <View style={styles.itemInfo}>
                <Text style={styles.itemType} numberOfLines={1}>{item.specificType}</Text>
                <Text style={styles.itemColor}>{item.color}</Text>
            </View>

            {/* Action buttons */}
            <View style={styles.itemActions}>
                <TouchableOpacity
                    style={styles.editBtn}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setSelectedItem(item);
                        setEditType(item.specificType);
                        setShowEditModal(true);
                    }}
                >
                    <Ionicons name="pencil" size={16} color={ALTA.primary} />
                </TouchableOpacity>

                <TouchableOpacity
                    style={styles.deleteBtn}
                    onPress={() => handleDelete(item)}
                >
                    <Ionicons name="trash-outline" size={16} color={ALTA.danger} />
                </TouchableOpacity>
            </View>
        </Animated.View>
    );

    if (loading) {
        return (
            <View style={[styles.container, styles.centered]}>
                <ActivityIndicator size="large" color={ALTA.primary} />
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <View style={styles.header}>
                    <Text style={styles.headerTitle}>My Wardrobe</Text>
                    <Text style={styles.headerSubtitle}>{items.length} items</Text>
                </View>

                {/* Search Bar */}
                <Animated.View entering={FadeInDown.delay(100).springify()} style={styles.searchContainer}>
                    <Ionicons name="search" size={20} color={ALTA.textMuted} />
                    <TextInput
                        style={styles.searchInput}
                        placeholder="Search by type, color..."
                        placeholderTextColor={ALTA.textMuted}
                        value={searchQuery}
                        onChangeText={setSearchQuery}
                    />
                    {searchQuery.length > 0 && (
                        <TouchableOpacity onPress={() => setSearchQuery('')}>
                            <Ionicons name="close-circle" size={20} color={ALTA.textMuted} />
                        </TouchableOpacity>
                    )}
                </Animated.View>

                {/* Category Filter Pills */}
                {categories.length > 0 && (
                    <ScrollView
                        horizontal
                        showsHorizontalScrollIndicator={false}
                        style={styles.filterScroll}
                        contentContainerStyle={styles.filterContent}
                    >
                        <TouchableOpacity
                            style={[styles.filterPill, !filterCategory && styles.filterPillActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setFilterCategory(null);
                            }}
                        >
                            <Text style={[styles.filterPillText, !filterCategory && styles.filterPillTextActive]}>
                                All
                            </Text>
                        </TouchableOpacity>
                        {categories.map(cat => (
                            <TouchableOpacity
                                key={cat}
                                style={[styles.filterPill, filterCategory === cat && styles.filterPillActive]}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                    setFilterCategory(filterCategory === cat ? null : cat);
                                }}
                            >
                                <Text style={[styles.filterPillText, filterCategory === cat && styles.filterPillTextActive]}>
                                    {cat}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                )}

                {/* 🎬 Outfit Filter Pills */}
                {outfits.length > 1 && (
                    <ScrollView
                        horizontal
                        showsHorizontalScrollIndicator={false}
                        style={styles.filterScroll}
                        contentContainerStyle={styles.filterContent}
                    >
                        <TouchableOpacity
                            style={[styles.filterPill, filterOutfit === null && styles.outfitPillActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setFilterOutfit(null);
                            }}
                        >
                            <Text style={[styles.filterPillText, filterOutfit === null && styles.outfitPillTextActive]}>
                                All Outfits
                            </Text>
                        </TouchableOpacity>
                        {outfits.map(outfitId => (
                            <TouchableOpacity
                                key={outfitId}
                                style={[styles.filterPill, filterOutfit === outfitId && styles.outfitPillActive]}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                    setFilterOutfit(filterOutfit === outfitId ? null : outfitId);
                                }}
                            >
                                <Text style={[styles.filterPillText, filterOutfit === outfitId && styles.outfitPillTextActive]}>
                                    Outfit {outfitId}
                                </Text>
                            </TouchableOpacity>
                        ))}
                    </ScrollView>
                )}

                {/* Items Grid */}
                {filteredItems.length > 0 ? (
                    <FlatList
                        data={filteredItems}
                        numColumns={2}
                        keyExtractor={item => item.id}
                        renderItem={({ item, index }) => <ItemCard item={item} index={index} />}
                        contentContainerStyle={styles.gridContent}
                        refreshControl={
                            <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={ALTA.text} />
                        }
                        showsVerticalScrollIndicator={false}
                    />
                ) : (
                    <View style={styles.emptyState}>
                        <Ionicons name="shirt-outline" size={64} color={ALTA.textMuted} />
                        <Text style={styles.emptyTitle}>
                            {searchQuery ? 'No matching items' : 'Your wardrobe is empty'}
                        </Text>
                        <Text style={styles.emptySubtitle}>
                            {searchQuery
                                ? 'Try a different search term'
                                : 'Scan clothing with AI to add items'
                            }
                        </Text>
                        {!searchQuery && (
                            <TouchableOpacity
                                style={styles.scanButton}
                                onPress={() => (navigation as any).navigate('WardrobeVideo')}
                            >
                                <Ionicons name="camera" size={20} color={ALTA.white} />
                                <Text style={styles.scanButtonText}>Scan Clothing</Text>
                            </TouchableOpacity>
                        )}
                    </View>
                )}
            </SafeAreaView>

            {/* Edit Type Modal */}
            <Modal visible={showEditModal} animationType="slide" transparent>
                <View style={styles.modalOverlay}>
                    <View style={styles.editModal}>
                        <View style={styles.editModalHeader}>
                            <Text style={styles.editModalTitle}>Edit Clothing Type</Text>
                            <TouchableOpacity onPress={() => setShowEditModal(false)}>
                                <Ionicons name="close" size={24} color={ALTA.text} />
                            </TouchableOpacity>
                        </View>

                        <Text style={styles.editModalSubtitle}>
                            Current: {selectedItem?.specificType}
                        </Text>

                        <ScrollView style={styles.typeList} showsVerticalScrollIndicator={false}>
                            {CLOTHING_TYPES.map(type => (
                                <TouchableOpacity
                                    key={type}
                                    style={[
                                        styles.typeOption,
                                        editType === type && styles.typeOptionSelected
                                    ]}
                                    onPress={() => setEditType(type)}
                                >
                                    <Text style={[
                                        styles.typeOptionText,
                                        editType === type && styles.typeOptionTextSelected
                                    ]}>
                                        {type}
                                    </Text>
                                    {editType === type && (
                                        <Ionicons name="checkmark" size={20} color={ALTA.primary} />
                                    )}
                                </TouchableOpacity>
                            ))}
                        </ScrollView>

                        <TouchableOpacity
                            style={styles.saveButton}
                            onPress={() => handleEditType(editType)}
                        >
                            <Text style={styles.saveButtonText}>Save Changes</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            </Modal>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: ALTA.background,
    },
    safeArea: {
        flex: 1,
    },
    centered: {
        justifyContent: 'center',
        alignItems: 'center',
    },

    // Header
    header: {
        paddingHorizontal: 20,
        paddingTop: 8,
        paddingBottom: 16,
    },
    headerTitle: {
        fontSize: 28,
        fontWeight: '700',
        color: ALTA.text,
    },
    headerSubtitle: {
        fontSize: 14,
        color: ALTA.textSecondary,
        marginTop: 4,
    },

    // Search
    searchContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: ALTA.surface,
        marginHorizontal: 20,
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 12,
        marginBottom: 12,
    },
    searchInput: {
        flex: 1,
        fontSize: 16,
        color: ALTA.text,
        marginLeft: 12,
    },

    // Filter Pills
    filterScroll: {
        maxHeight: 44,
        marginBottom: 16,
    },
    filterContent: {
        paddingHorizontal: 20,
        gap: 8,
    },
    filterPill: {
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        backgroundColor: ALTA.surface,
        marginRight: 8,
    },
    filterPillActive: {
        backgroundColor: ALTA.primary,
    },
    filterPillText: {
        fontSize: 14,
        color: ALTA.text,
    },
    filterPillTextActive: {
        color: ALTA.white,
        fontWeight: '600',
    },
    // 🎬 Outfit pill styles
    outfitPillActive: {
        backgroundColor: ALTA.accent,
    },
    outfitPillTextActive: {
        color: ALTA.white,
        fontWeight: '600',
    },

    // Grid
    gridContent: {
        paddingHorizontal: 16,
        paddingBottom: 100,
    },
    itemCard: {
        flex: 1,
        margin: 4,
        backgroundColor: ALTA.surface,
        borderRadius: 12,
        overflow: 'hidden',
    },
    itemImageContainer: {
        aspectRatio: 0.85,
        backgroundColor: '#F5F5F3',
    },
    itemImage: {
        width: '100%',
        height: '100%',
    },
    colorDot: {
        position: 'absolute',
        bottom: 8,
        right: 8,
        width: 16,
        height: 16,
        borderRadius: 8,
        borderWidth: 2,
        borderColor: ALTA.white,
    },
    itemInfo: {
        padding: 12,
    },
    itemType: {
        fontSize: 14,
        fontWeight: '600',
        color: ALTA.text,
    },
    itemColor: {
        fontSize: 12,
        color: ALTA.textSecondary,
        marginTop: 2,
    },
    itemActions: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        paddingHorizontal: 12,
        paddingBottom: 12,
        gap: 8,
    },
    editBtn: {
        padding: 8,
        borderRadius: 8,
        backgroundColor: 'rgba(0,122,255,0.1)',
    },
    deleteBtn: {
        padding: 8,
        borderRadius: 8,
        backgroundColor: 'rgba(255,69,58,0.1)',
    },

    // Empty State
    emptyState: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 20,
    },
    emptyTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: ALTA.text,
        marginTop: 16,
    },
    emptySubtitle: {
        fontSize: 14,
        color: ALTA.textSecondary,
        textAlign: 'center',
        marginTop: 8,
    },
    scanButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: ALTA.primary,
        paddingHorizontal: 24,
        paddingVertical: 14,
        borderRadius: 12,
        marginTop: 24,
        gap: 8,
    },
    scanButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: ALTA.white,
    },

    // Modal
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.5)',
        justifyContent: 'flex-end',
    },
    editModal: {
        backgroundColor: ALTA.background,
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        padding: 24,
        maxHeight: '70%',
    },
    editModalHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 8,
    },
    editModalTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: ALTA.text,
    },
    editModalSubtitle: {
        fontSize: 14,
        color: ALTA.textSecondary,
        marginBottom: 16,
    },
    typeList: {
        maxHeight: 300,
    },
    typeOption: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingVertical: 14,
        paddingHorizontal: 16,
        borderRadius: 10,
        marginBottom: 4,
    },
    typeOptionSelected: {
        backgroundColor: 'rgba(0,122,255,0.1)',
    },
    typeOptionText: {
        fontSize: 16,
        color: ALTA.text,
    },
    typeOptionTextSelected: {
        fontWeight: '600',
        color: ALTA.primary,
    },
    saveButton: {
        backgroundColor: ALTA.primary,
        paddingVertical: 16,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 16,
    },
    saveButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: ALTA.white,
    },
});

export default MyWardrobeScreen;
