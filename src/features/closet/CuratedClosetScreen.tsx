import React, { useState, useCallback, useRef } from "react";
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
    FlatList,
    ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    FadeInLeft,
    SlideInUp,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withTiming,
} from "react-native-reanimated";
import { useNavigation } from "@react-navigation/native";
import * as Haptics from "expo-haptics";
import { BlurView } from "expo-blur";
import AppColors from "../../../constants/AppColors";

const { width, height } = Dimensions.get("window");

// Use unified AppColors
const ALTA = {
    background: AppColors.background,
    surface: AppColors.surface,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    textMuted: AppColors.textMuted,
    border: AppColors.border,
    accent: AppColors.warning,
    black: AppColors.primary,
    white: AppColors.background,
    glass: 'rgba(0, 0, 0, 0.5)',
};

// Trending Outfits Data (masonry grid)
const TRENDING_OUTFITS = [
    {
        id: '1',
        image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600',
        saves: 234,
        user: 'styleking',
        items: [
            { id: '1', type: 'Cap', brand: 'New Era', name: 'Baseball Cap', price: '$35.00', image: 'https://images.unsplash.com/photo-1588850561407-ed78c282e89b?w=200' },
            { id: '2', type: 'Sunglasses', brand: 'Ray-Ban', name: 'Wayfarer', price: '$161.00', image: 'https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=200' },
            { id: '3', type: 'T-Shirt', brand: 'Fear of God', name: 'Oversized Tee', price: '$95.00', image: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=200' },
            { id: '4', type: 'Shorts', brand: 'Rick Owens', name: 'Cargo Shorts', price: '$580.00', image: 'https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=200' },
            { id: '5', type: 'Sneakers', brand: 'Converse', name: 'Chuck Taylor', price: '$65.00', image: 'https://images.unsplash.com/photo-1600269452121-4f2416e55c28?w=200' },
        ],
    },
    {
        id: '2',
        image: 'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=600',
        saves: 189,
        user: 'dapper',
        items: [
            { id: '1', type: 'Blazer', brand: 'Hugo Boss', name: 'Slim Fit', price: '$495.00', image: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=200' },
            { id: '2', type: 'Shirt', brand: 'Brooks Brothers', name: 'Oxford', price: '$120.00', image: 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=200' },
            { id: '3', type: 'Pants', brand: 'Theory', name: 'Chinos', price: '$225.00', image: 'https://images.unsplash.com/photo-1473966968600-fa801b869a1a?w=200' },
        ],
    },
    {
        id: '3',
        image: 'https://images.unsplash.com/photo-1552374196-1ab2a1c593e8?w=600',
        saves: 156,
        user: 'streetwear',
        items: [
            { id: '1', type: 'Hoodie', brand: 'Supreme', name: 'Box Logo', price: '$168.00', image: 'https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=200' },
            { id: '2', type: 'Joggers', brand: 'Nike', name: 'Tech Fleece', price: '$110.00', image: 'https://images.unsplash.com/photo-1552902865-b72c031ac5ea?w=200' },
        ],
    },
    {
        id: '4',
        image: 'https://images.unsplash.com/photo-1506634572416-48cdfe530110?w=600',
        saves: 98,
        user: 'casual',
        items: [
            { id: '1', type: 'Cardigan', brand: 'COS', name: 'Wool Blend', price: '$175.00', image: 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=200' },
            { id: '2', type: 'Jeans', brand: "Levi's", name: '501 Original', price: '$98.00', image: 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=200' },
        ],
    },
];

// Curated Collections
const COLLECTIONS = [
    {
        id: '1',
        title: 'Holiday Dressing Gift Guide',
        subtitle: 'Our picks for the holiday season',
        linkText: 'Shop curated items',
        image: 'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=800',
    },
    {
        id: '2',
        title: 'The Harvest Edit',
        subtitle: 'Cozy layers, rich textures, golden-hour tones.',
        credit: 'BY ALTA EDITORIAL TEAM',
        linkText: 'Explore this capsule',
        image: 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=800',
    },
];

// iOS 26 Tahoe Press Hook
const useTahoePress = () => {
    const scale = useSharedValue(1);
    const opacity = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 20, stiffness: 400 }) }],
        opacity: opacity.value,
    }));

    const onPressIn = () => {
        scale.value = 0.97;
        opacity.value = withTiming(0.9, { duration: 60 });
    };

    const onPressOut = () => {
        scale.value = 1;
        opacity.value = withTiming(1, { duration: 100 });
    };

    return { animatedStyle, onPressIn, onPressOut };
};

// Tab Button
const TabButton = ({ title, isActive, onPress }: { title: string; isActive: boolean; onPress: () => void }) => {
    const { animatedStyle, onPressIn, onPressOut } = useTahoePress();

    return (
        <TouchableOpacity
            onPressIn={onPressIn}
            onPressOut={onPressOut}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={1}
        >
            <Animated.View style={animatedStyle}>
                <Text style={[styles.tabText, isActive && styles.tabTextActive]}>{title}</Text>
            </Animated.View>
        </TouchableOpacity>
    );
};

// Outfit Grid Item
const OutfitGridItem = ({ outfit, index, onPress }: { outfit: any; index: number; onPress: () => void }) => {
    const { animatedStyle, onPressIn, onPressOut } = useTahoePress();
    const isLarge = index % 3 === 0;

    return (
        <Animated.View
            entering={FadeInUp.delay(index * 60).springify()}
            style={[styles.gridItem, isLarge && styles.gridItemLarge]}
        >
            <TouchableOpacity
                onPressIn={onPressIn}
                onPressOut={onPressOut}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    onPress();
                }}
                activeOpacity={1}
                style={styles.gridItemTouchable}
            >
                <Animated.View style={[styles.gridItemInner, animatedStyle]}>
                    <Image source={{ uri: outfit.image }} style={styles.gridItemImage} resizeMode="cover" />
                    {/* Hanger/Save Count */}
                    <View style={styles.saveCountBadge}>
                        <Ionicons name="bookmark" size={12} color={ALTA.white} />
                        <Text style={styles.saveCountText}>{outfit.saves}</Text>
                    </View>
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// Full Screen Outfit Modal (Like Alta Daily)
const OutfitModal = ({
    visible,
    outfit,
    allOutfits,
    currentIndex,
    onClose,
    onNavigate,
    onAvatar
}: {
    visible: boolean;
    outfit: any;
    allOutfits: any[];
    currentIndex: number;
    onClose: () => void;
    onNavigate: (direction: 'up' | 'down') => void;
    onAvatar: () => void;
}) => {
    const [selectedItem, setSelectedItem] = useState<any>(null);
    const [isBookmarked, setIsBookmarked] = useState(false);

    if (!outfit) return null;

    return (
        <Modal visible={visible} animationType="fade" transparent onRequestClose={onClose}>
            <View style={styles.modalContainer}>
                {/* Background Image */}
                <Image source={{ uri: outfit.image }} style={styles.modalImage} resizeMode="cover" />

                {/* Overlay */}
                <View style={styles.modalOverlay} />

                {/* Header - Navigation Arrows & Close */}
                <SafeAreaView style={styles.modalHeader}>
                    <View style={styles.modalNavArrows}>
                        <TouchableOpacity
                            style={styles.modalNavArrow}
                            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                        >
                            <Ionicons name="chevron-back" size={20} color={ALTA.white} />
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={styles.modalNavArrow}
                            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                        >
                            <Ionicons name="chevron-forward" size={20} color={ALTA.white} />
                        </TouchableOpacity>
                    </View>

                    <TouchableOpacity
                        style={styles.modalCloseBtn}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            onClose();
                        }}
                    >
                        <Ionicons name="close" size={28} color={ALTA.white} />
                    </TouchableOpacity>
                </SafeAreaView>

                {/* Left Side - Detected Items (Shop the Look) */}
                <View style={styles.shopTheLook}>
                    <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.shopTheLookScroll}>
                        {outfit.items?.map((item: any, idx: number) => (
                            <Animated.View
                                key={item.id}
                                entering={FadeInLeft.delay(idx * 80).springify()}
                            >
                                <TouchableOpacity
                                    style={[
                                        styles.itemThumb,
                                        selectedItem?.id === item.id && styles.itemThumbSelected
                                    ]}
                                    onPress={() => {
                                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                        setSelectedItem(item);
                                    }}
                                >
                                    <Image source={{ uri: item.image }} style={styles.itemThumbImage} />
                                    <View style={styles.itemShopBadge}>
                                        <Ionicons name="bag-outline" size={10} color={ALTA.white} />
                                    </View>
                                </TouchableOpacity>
                            </Animated.View>
                        ))}
                    </ScrollView>
                </View>

                {/* Right Side - Vertical Navigation */}
                <View style={styles.modalVerticalNav}>
                    <TouchableOpacity
                        style={styles.modalVerticalNavBtn}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            onNavigate('up');
                        }}
                    >
                        <Ionicons name="chevron-up" size={24} color={ALTA.white} />
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.modalVerticalNavBtn}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            onNavigate('down');
                        }}
                    >
                        <Ionicons name="chevron-down" size={24} color={ALTA.white} />
                    </TouchableOpacity>
                </View>

                {/* Bottom Bar */}
                <View style={styles.modalBottomBar}>
                    {/* Hanger/Bookmark Count */}
                    <TouchableOpacity
                        style={styles.modalBookmarkBtn}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                            setIsBookmarked(!isBookmarked);
                        }}
                    >
                        <Ionicons
                            name={isBookmarked ? "bookmark" : "bookmark-outline"}
                            size={22}
                            color={ALTA.white}
                        />
                        <Text style={styles.modalBookmarkCount}>{outfit.saves}</Text>
                    </TouchableOpacity>

                    <View style={{ flex: 1 }} />

                    {/* Avatar Button */}
                    <TouchableOpacity
                        style={styles.modalAvatarBtn}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                            onAvatar();
                        }}
                    >
                        <Text style={styles.modalAvatarText}>Avatar</Text>
                        <View style={styles.modalAvatarIcon}>
                            <Ionicons name="person" size={14} color={ALTA.white} />
                        </View>
                        <View style={styles.modalAvatarBadge}>
                            <Text style={styles.modalAvatarBadgeText}>+22</Text>
                        </View>
                    </TouchableOpacity>
                </View>

                {/* Selected Item Details Panel */}
                {selectedItem && (
                    <Animated.View
                        entering={SlideInUp.springify()}
                        style={styles.itemDetailPanel}
                    >
                        <BlurView intensity={80} tint="dark" style={styles.itemDetailBlur}>
                            <TouchableOpacity
                                style={styles.itemDetailClose}
                                onPress={() => setSelectedItem(null)}
                            >
                                <Ionicons name="close" size={20} color={ALTA.white} />
                            </TouchableOpacity>

                            <Image source={{ uri: selectedItem.image }} style={styles.itemDetailImage} />

                            <View style={styles.itemDetailInfo}>
                                <Text style={styles.itemDetailBrand}>{selectedItem.brand}</Text>
                                <Text style={styles.itemDetailName}>{selectedItem.name}</Text>
                                <Text style={styles.itemDetailPrice}>{selectedItem.price}</Text>
                            </View>

                            <View style={styles.itemDetailActions}>
                                <TouchableOpacity style={styles.itemDetailSaveBtn}>
                                    <Ionicons name="bookmark-outline" size={18} color={ALTA.white} />
                                    <Text style={styles.itemDetailSaveText}>Save</Text>
                                </TouchableOpacity>
                                <TouchableOpacity style={styles.itemDetailShopBtn}>
                                    <Text style={styles.itemDetailShopText}>Shop Now</Text>
                                    <Ionicons name="arrow-forward" size={16} color={ALTA.black} />
                                </TouchableOpacity>
                            </View>
                        </BlurView>
                    </Animated.View>
                )}
            </View>
        </Modal>
    );
};

// Collection Card
const CollectionCard = ({ collection, index, onPress }: { collection: any; index: number; onPress: () => void }) => {
    const { animatedStyle, onPressIn, onPressOut } = useTahoePress();

    return (
        <Animated.View entering={FadeInUp.delay(index * 100).springify()} style={styles.collectionCard}>
            <TouchableOpacity
                onPressIn={onPressIn}
                onPressOut={onPressOut}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    onPress();
                }}
                activeOpacity={1}
            >
                <Animated.View style={animatedStyle}>
                    <Image source={{ uri: collection.image }} style={styles.collectionImage} resizeMode="cover" />
                </Animated.View>
            </TouchableOpacity>

            <Text style={styles.collectionTitle}>{collection.title}</Text>
            <Text style={styles.collectionSubtitle}>{collection.subtitle}</Text>
            {collection.credit && <Text style={styles.collectionCredit}>{collection.credit}</Text>}

            <View style={styles.collectionLinkContainer}>
                <TouchableOpacity
                    style={styles.collectionLink}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        onPress();
                    }}
                >
                    <Text style={styles.collectionLinkText}>{collection.linkText}</Text>
                    <Ionicons name="arrow-forward" size={14} color={ALTA.text} />
                </TouchableOpacity>
            </View>

            <View style={styles.collectionDivider} />
        </Animated.View>
    );
};

// Main Screen
const CuratedClosetScreen = () => {
    const navigation = useNavigation();
    const [activeTab, setActiveTab] = useState<'community' | 'saved'>('community');
    const [refreshing, setRefreshing] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const [currentOutfitIndex, setCurrentOutfitIndex] = useState(0);

    const currentOutfit = TRENDING_OUTFITS[currentOutfitIndex];

    const onRefresh = useCallback(() => {
        setRefreshing(true);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setTimeout(() => setRefreshing(false), 1500);
    }, []);

    const handleOutfitPress = (outfit: any, index: number) => {
        setCurrentOutfitIndex(index);
        setShowModal(true);
    };

    const handleNavigate = (direction: 'up' | 'down') => {
        if (direction === 'up' && currentOutfitIndex > 0) {
            setCurrentOutfitIndex(currentOutfitIndex - 1);
        } else if (direction === 'down' && currentOutfitIndex < TRENDING_OUTFITS.length - 1) {
            setCurrentOutfitIndex(currentOutfitIndex + 1);
        }
    };

    const handleAvatar = () => {
        setShowModal(false);
        (navigation as any).navigate('AITryOn', { outfit: currentOutfit });
    };

    // Split into columns for masonry
    const leftColumn = TRENDING_OUTFITS.filter((_, i) => i % 2 === 0);
    const rightColumn = TRENDING_OUTFITS.filter((_, i) => i % 2 === 1);

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header Tabs */}
                <Animated.View entering={FadeInDown.delay(50).springify()} style={styles.header}>
                    <TabButton
                        title="Community"
                        isActive={activeTab === 'community'}
                        onPress={() => setActiveTab('community')}
                    />
                    <TabButton
                        title="Saved"
                        isActive={activeTab === 'saved'}
                        onPress={() => setActiveTab('saved')}
                    />
                </Animated.View>

                <ScrollView
                    showsVerticalScrollIndicator={false}
                    contentContainerStyle={styles.scrollContent}
                    refreshControl={
                        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={ALTA.text} />
                    }
                >
                    {activeTab === 'community' ? (
                        <>
                            {/* Trending Section */}
                            <Animated.View entering={FadeInUp.delay(100).springify()} style={styles.trendingSection}>
                                <View style={styles.trendingBadge}>
                                    <Text style={styles.trendingBadgeText}>UPDATED TODAY</Text>
                                    <View style={styles.trendingDot} />
                                </View>
                                <Text style={styles.trendingTitle}>Trending</Text>
                                <Text style={styles.trendingSubtitle}>See what's popular in the Alta community</Text>

                                <View style={styles.discoverLinkContainer}>
                                    <TouchableOpacity
                                        style={styles.discoverLink}
                                        onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                                    >
                                        <Text style={styles.discoverLinkText}>Discover trends</Text>
                                        <Ionicons name="arrow-forward" size={14} color={ALTA.text} />
                                    </TouchableOpacity>
                                </View>
                            </Animated.View>

                            {/* Masonry Grid */}
                            <View style={styles.masonryGrid}>
                                <View style={styles.masonryColumn}>
                                    {leftColumn.map((outfit, idx) => (
                                        <OutfitGridItem
                                            key={outfit.id}
                                            outfit={outfit}
                                            index={idx * 2}
                                            onPress={() => handleOutfitPress(outfit, TRENDING_OUTFITS.indexOf(outfit))}
                                        />
                                    ))}
                                </View>
                                <View style={styles.masonryColumn}>
                                    {rightColumn.map((outfit, idx) => (
                                        <OutfitGridItem
                                            key={outfit.id}
                                            outfit={outfit}
                                            index={idx * 2 + 1}
                                            onPress={() => handleOutfitPress(outfit, TRENDING_OUTFITS.indexOf(outfit))}
                                        />
                                    ))}
                                </View>
                            </View>

                            {/* Collections */}
                            <View style={styles.collectionsSection}>
                                {COLLECTIONS.map((collection, index) => (
                                    <CollectionCard
                                        key={collection.id}
                                        collection={collection}
                                        index={index}
                                        onPress={() => handleOutfitPress(collection, 0)}
                                    />
                                ))}
                            </View>
                        </>
                    ) : (
                        <Animated.View entering={FadeIn} style={styles.savedSection}>
                            <Text style={styles.savedEmptyTitle}>Your saved looks</Text>
                            <Text style={styles.savedEmptySubtitle}>
                                Bookmark outfits you love to find them here
                            </Text>
                        </Animated.View>
                    )}

                    <View style={{ height: 100 }} />
                </ScrollView>
            </SafeAreaView>

            {/* Full Screen Outfit Modal */}
            <OutfitModal
                visible={showModal}
                outfit={currentOutfit}
                allOutfits={TRENDING_OUTFITS}
                currentIndex={currentOutfitIndex}
                onClose={() => setShowModal(false)}
                onNavigate={handleNavigate}
                onAvatar={handleAvatar}
            />
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

    // Header
    header: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 40,
        paddingVertical: 14,
    },
    tabText: {
        fontSize: 16,
        fontWeight: '400',
        color: ALTA.textMuted,
    },
    tabTextActive: {
        fontWeight: '600',
        color: ALTA.text,
    },

    scrollContent: {
        paddingHorizontal: 16,
    },

    // Trending Section
    trendingSection: {
        marginBottom: 20,
    },
    trendingBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        marginBottom: 8,
    },
    trendingBadgeText: {
        fontSize: 11,
        fontWeight: '500',
        color: ALTA.textSecondary,
        letterSpacing: 0.5,
    },
    trendingDot: {
        width: 6,
        height: 6,
        borderRadius: 3,
        backgroundColor: ALTA.accent,
    },
    trendingTitle: {
        fontSize: 28,
        fontWeight: '700',
        color: ALTA.text,
        marginBottom: 4,
    },
    trendingSubtitle: {
        fontSize: 15,
        color: ALTA.textSecondary,
        marginBottom: 12,
    },
    discoverLinkContainer: {
        alignItems: 'flex-end',
    },
    discoverLink: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    discoverLinkText: {
        fontSize: 14,
        fontWeight: '500',
        color: ALTA.text,
    },

    // Masonry Grid
    masonryGrid: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: 24,
    },
    masonryColumn: {
        flex: 1,
        gap: 8,
    },
    gridItem: {
        borderRadius: 12,
        overflow: 'hidden',
    },
    gridItemLarge: {},
    gridItemTouchable: {
        flex: 1,
    },
    gridItemInner: {
        borderRadius: 12,
        overflow: 'hidden',
    },
    gridItemImage: {
        width: '100%',
        height: 240,
        borderRadius: 12,
    },
    saveCountBadge: {
        position: 'absolute',
        bottom: 8,
        right: 8,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        backgroundColor: 'rgba(0,0,0,0.6)',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 12,
    },
    saveCountText: {
        fontSize: 12,
        fontWeight: '600',
        color: ALTA.white,
    },

    // Collections
    collectionsSection: {
        marginTop: 16,
    },
    collectionCard: {
        marginBottom: 8,
    },
    collectionImage: {
        width: '100%',
        height: 280,
        borderRadius: 0,
        backgroundColor: ALTA.surface,
    },
    collectionTitle: {
        fontSize: 22,
        fontWeight: '600',
        color: ALTA.text,
        marginTop: 16,
        marginBottom: 4,
    },
    collectionSubtitle: {
        fontSize: 14,
        color: ALTA.textSecondary,
        fontStyle: 'italic',
    },
    collectionCredit: {
        fontSize: 11,
        color: ALTA.textMuted,
        letterSpacing: 0.5,
        marginTop: 8,
    },
    collectionLinkContainer: {
        alignItems: 'flex-end',
        marginTop: 12,
    },
    collectionLink: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    collectionLinkText: {
        fontSize: 14,
        fontWeight: '400',
        color: ALTA.text,
    },
    collectionDivider: {
        height: 0.5,
        backgroundColor: ALTA.border,
        marginTop: 20,
        marginBottom: 24,
    },

    // Saved Section
    savedSection: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        paddingTop: 100,
    },
    savedEmptyTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: ALTA.text,
        marginBottom: 8,
    },
    savedEmptySubtitle: {
        fontSize: 14,
        color: ALTA.textSecondary,
        textAlign: 'center',
    },

    // Modal Styles
    modalContainer: {
        flex: 1,
        backgroundColor: ALTA.black,
    },
    modalImage: {
        ...StyleSheet.absoluteFillObject,
        width: width,
        height: height,
    },
    modalOverlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0,0,0,0.2)',
    },
    modalHeader: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingTop: 8,
        zIndex: 10,
    },
    modalNavArrows: {
        flexDirection: 'row',
        gap: 8,
    },
    modalNavArrow: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: 'rgba(255,255,255,0.15)',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.2)',
    },
    modalCloseBtn: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: 'rgba(255,255,255,0.15)',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.2)',
    },

    // Shop the Look
    shopTheLook: {
        position: 'absolute',
        left: 12,
        top: height * 0.15,
        bottom: height * 0.15,
        width: 60,
        zIndex: 5,
    },
    shopTheLookScroll: {
        gap: 12,
        paddingVertical: 8,
    },
    itemThumb: {
        width: 50,
        height: 50,
        borderRadius: 8,
        backgroundColor: ALTA.surface,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    itemThumbSelected: {
        borderColor: ALTA.white,
    },
    itemThumbImage: {
        width: '100%',
        height: '100%',
    },
    itemShopBadge: {
        position: 'absolute',
        bottom: -4,
        right: -4,
        width: 18,
        height: 18,
        borderRadius: 9,
        backgroundColor: '#007AFF',
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Vertical Nav
    modalVerticalNav: {
        position: 'absolute',
        right: 16,
        top: '45%',
        gap: 8,
        zIndex: 5,
    },
    modalVerticalNavBtn: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: 'rgba(255,255,255,0.15)',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.2)',
    },

    // Bottom Bar
    modalBottomBar: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 20,
        paddingBottom: 40,
        paddingTop: 16,
    },
    modalBookmarkBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: 'rgba(255,255,255,0.15)',
        paddingHorizontal: 14,
        paddingVertical: 10,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.2)',
    },
    modalBookmarkCount: {
        fontSize: 14,
        fontWeight: '600',
        color: ALTA.white,
    },
    modalAvatarBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        backgroundColor: 'rgba(255,255,255,0.15)',
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.2)',
    },
    modalAvatarText: {
        fontSize: 14,
        fontWeight: '600',
        color: ALTA.white,
    },
    modalAvatarIcon: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: 'rgba(255,255,255,0.3)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    modalAvatarBadge: {
        backgroundColor: '#FF3B30',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 10,
    },
    modalAvatarBadgeText: {
        fontSize: 10,
        fontWeight: '700',
        color: ALTA.white,
    },

    // Item Detail Panel
    itemDetailPanel: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: height * 0.45,
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        overflow: 'hidden',
    },
    itemDetailBlur: {
        flex: 1,
        padding: 20,
    },
    itemDetailClose: {
        alignSelf: 'flex-end',
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
    },
    itemDetailImage: {
        width: '100%',
        height: 150,
        borderRadius: 12,
        backgroundColor: '#333',
        marginBottom: 16,
    },
    itemDetailInfo: {
        marginBottom: 20,
    },
    itemDetailBrand: {
        fontSize: 12,
        fontWeight: '600',
        color: ALTA.textMuted,
        marginBottom: 4,
    },
    itemDetailName: {
        fontSize: 18,
        fontWeight: '700',
        color: ALTA.white,
        marginBottom: 6,
    },
    itemDetailPrice: {
        fontSize: 16,
        fontWeight: '600',
        color: '#007AFF',
    },
    itemDetailActions: {
        flexDirection: 'row',
        gap: 12,
    },
    itemDetailSaveBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: 'rgba(255,255,255,0.2)',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderRadius: 20,
    },
    itemDetailSaveText: {
        fontSize: 14,
        fontWeight: '600',
        color: ALTA.white,
    },
    itemDetailShopBtn: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        backgroundColor: ALTA.white,
        paddingVertical: 12,
        borderRadius: 20,
    },
    itemDetailShopText: {
        fontSize: 14,
        fontWeight: '600',
        color: ALTA.black,
    },
});

export default CuratedClosetScreen;
