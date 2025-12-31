import React, { useState, useEffect } from "react";
import {
    View,
    Text,
    Image,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Modal,
    ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import Animated, {
    FadeIn,
    FadeInLeft,
    FadeInUp,
    FadeOut,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withTiming,
    SlideInRight,
} from "react-native-reanimated";
import { useNavigation, useRoute } from "@react-navigation/native";
import * as Haptics from "expo-haptics";
import { BlurView } from "expo-blur";

const { width, height } = Dimensions.get("window");

// Colors
const COLORS = {
    background: '#000000',
    surface: '#1A1A1A',
    white: '#FFFFFF',
    text: '#FFFFFF',
    textSecondary: '#999999',
    accent: '#007AFF',
    glass: 'rgba(255, 255, 255, 0.15)',
    glassBorder: 'rgba(255, 255, 255, 0.2)',
};

// Detected clothing items (simulated AI detection)
const DETECTED_ITEMS = [
    {
        id: '1',
        type: 'Cap',
        brand: 'New Era',
        name: 'Baseball Cap',
        price: '$35.00',
        image: 'https://images.unsplash.com/photo-1588850561407-ed78c282e89b?w=200',
        position: { top: 8, left: 0 },
    },
    {
        id: '2',
        type: 'Headphones',
        brand: 'Sony',
        name: 'WH-1000XM5 Wireless',
        price: '$349.00',
        image: 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=200',
        position: { top: 15, left: 0 },
    },
    {
        id: '3',
        type: 'Sunglasses',
        brand: 'Ray-Ban',
        name: 'Wayfarer Classic',
        price: '$161.00',
        image: 'https://images.unsplash.com/photo-1511499767150-a48a237f0083?w=200',
        position: { top: 22, left: 0 },
    },
    {
        id: '4',
        type: 'T-Shirt',
        brand: 'Fear of God',
        name: 'Essentials Oversized Tee',
        price: '$95.00',
        image: 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=200',
        position: { top: 35, left: 0 },
    },
    {
        id: '5',
        type: 'Bag',
        brand: 'IKEA',
        name: 'FRAKTA Shopping Bag',
        price: '$1.99',
        image: 'https://images.unsplash.com/photo-1553062407-98eeb64c6a62?w=200',
        position: { top: 50, left: 0 },
    },
    {
        id: '6',
        type: 'Shorts',
        brand: 'Rick Owens',
        name: 'Cargo Shorts',
        price: '$580.00',
        image: 'https://images.unsplash.com/photo-1591195853828-11db59a44f6b?w=200',
        position: { top: 58, left: 0 },
    },
    {
        id: '7',
        type: 'Sneakers',
        brand: 'Converse',
        name: 'Chuck Taylor All Star',
        price: '$65.00',
        image: 'https://images.unsplash.com/photo-1600269452121-4f2416e55c28?w=200',
        position: { top: 75, left: 0 },
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
        scale.value = 0.95;
        opacity.value = withTiming(0.85, { duration: 60 });
    };

    const onPressOut = () => {
        scale.value = 1;
        opacity.value = withTiming(1, { duration: 100 });
    };

    return { animatedStyle, onPressIn, onPressOut };
};

// Clothing Item Thumbnail Component
const ClothingItemThumbnail = ({
    item,
    index,
    isSelected,
    onPress,
}: {
    item: any;
    index: number;
    isSelected: boolean;
    onPress: () => void;
}) => {
    const { animatedStyle, onPressIn, onPressOut } = useTahoePress();

    return (
        <Animated.View
            entering={FadeInLeft.delay(index * 80).springify()}
            style={styles.itemThumbnailContainer}
        >
            <TouchableOpacity
                onPressIn={onPressIn}
                onPressOut={onPressOut}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    onPress();
                }}
                activeOpacity={1}
            >
                <Animated.View style={[styles.itemThumbnail, isSelected && styles.itemThumbnailSelected, animatedStyle]}>
                    <Image
                        source={{ uri: item.image }}
                        style={styles.itemThumbnailImage}
                        resizeMode="cover"
                    />
                </Animated.View>
            </TouchableOpacity>

            {/* Shopping Icon */}
            <View style={styles.shoppingIcon}>
                <Ionicons name="bag-outline" size={10} color={COLORS.white} />
            </View>
        </Animated.View>
    );
};

// Product Detail Modal
const ProductDetailModal = ({
    visible,
    item,
    onClose,
}: {
    visible: boolean;
    item: any;
    onClose: () => void;
}) => {
    if (!item) return null;

    return (
        <Modal
            visible={visible}
            animationType="slide"
            transparent
            onRequestClose={onClose}
        >
            <View style={styles.modalOverlay}>
                <Animated.View
                    entering={SlideInRight.springify()}
                    style={styles.modalContent}
                >
                    <BlurView intensity={80} tint="dark" style={styles.modalBlur}>
                        {/* Close Button */}
                        <TouchableOpacity
                            style={styles.modalCloseButton}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                onClose();
                            }}
                        >
                            <Ionicons name="close" size={24} color={COLORS.white} />
                        </TouchableOpacity>

                        {/* Product Image */}
                        <Image
                            source={{ uri: item.image }}
                            style={styles.modalProductImage}
                            resizeMode="cover"
                        />

                        {/* Product Info */}
                        <View style={styles.modalProductInfo}>
                            <Text style={styles.modalBrand}>{item.brand}</Text>
                            <Text style={styles.modalName}>{item.name}</Text>
                            <Text style={styles.modalPrice}>{item.price}</Text>
                        </View>

                        {/* Action Buttons */}
                        <View style={styles.modalActions}>
                            <TouchableOpacity
                                style={styles.modalSaveButton}
                                onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium)}
                            >
                                <Ionicons name="bookmark-outline" size={20} color={COLORS.white} />
                                <Text style={styles.modalSaveText}>Save</Text>
                            </TouchableOpacity>

                            <TouchableOpacity
                                style={styles.modalShopButton}
                                onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium)}
                            >
                                <Text style={styles.modalShopText}>Shop Now</Text>
                                <Ionicons name="arrow-forward" size={16} color={COLORS.background} />
                            </TouchableOpacity>
                        </View>
                    </BlurView>
                </Animated.View>
            </View>
        </Modal>
    );
};

// Main Screen
const OutfitDetailScreen = () => {
    const navigation = useNavigation();
    const route = useRoute();
    const [selectedItem, setSelectedItem] = useState<any>(null);
    const [showModal, setShowModal] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [bookmarkCount] = useState(144);
    const [isBookmarked, setIsBookmarked] = useState(false);

    // Get outfit data from route params or use default
    const outfitImage = (route.params as any)?.image ||
        'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=800';

    useEffect(() => {
        // Simulate AI detection loading
        const timer = setTimeout(() => setIsLoading(false), 1000);
        return () => clearTimeout(timer);
    }, []);

    const handleItemPress = (item: any) => {
        setSelectedItem(item);
        setShowModal(true);
    };

    const handleBookmark = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setIsBookmarked(!isBookmarked);
    };

    return (
        <View style={styles.container}>
            {/* Background Image */}
            <Image
                source={{ uri: outfitImage }}
                style={styles.backgroundImage}
                resizeMode="cover"
            />

            {/* Close Button */}
            <SafeAreaView style={styles.safeArea}>
                <Animated.View entering={FadeIn.delay(100)} style={styles.header}>
                    <View style={styles.headerSpacer} />

                    {/* Navigation Arrows */}
                    <View style={styles.navArrows}>
                        <TouchableOpacity
                            style={styles.navArrow}
                            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                        >
                            <Ionicons name="chevron-back" size={20} color={COLORS.white} />
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={styles.navArrow}
                            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                        >
                            <Ionicons name="chevron-forward" size={20} color={COLORS.white} />
                        </TouchableOpacity>
                    </View>

                    <TouchableOpacity
                        style={styles.closeButton}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            navigation.goBack();
                        }}
                    >
                        <Ionicons name="close" size={28} color={COLORS.white} />
                    </TouchableOpacity>
                </Animated.View>
            </SafeAreaView>

            {/* Left Side - Detected Items */}
            <View style={styles.itemsContainer}>
                {isLoading ? (
                    <Animated.View entering={FadeIn} style={styles.loadingContainer}>
                        <ActivityIndicator size="small" color={COLORS.white} />
                        <Text style={styles.loadingText}>Detecting items...</Text>
                    </Animated.View>
                ) : (
                    <ScrollView
                        showsVerticalScrollIndicator={false}
                        contentContainerStyle={styles.itemsList}
                    >
                        {DETECTED_ITEMS.map((item, index) => (
                            <ClothingItemThumbnail
                                key={item.id}
                                item={item}
                                index={index}
                                isSelected={selectedItem?.id === item.id}
                                onPress={() => handleItemPress(item)}
                            />
                        ))}
                    </ScrollView>
                )}
            </View>

            {/* Right Side - Up/Down Navigation */}
            <View style={styles.rightNav}>
                <TouchableOpacity
                    style={styles.navButton}
                    onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                >
                    <Ionicons name="chevron-up" size={24} color={COLORS.white} />
                </TouchableOpacity>
                <TouchableOpacity
                    style={styles.navButton}
                    onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                >
                    <Ionicons name="chevron-down" size={24} color={COLORS.white} />
                </TouchableOpacity>
            </View>

            {/* Bottom Bar */}
            <Animated.View
                entering={FadeInUp.delay(300).springify()}
                style={styles.bottomBar}
            >
                {/* Bookmark Count */}
                <TouchableOpacity
                    style={styles.bookmarkButton}
                    onPress={handleBookmark}
                >
                    <Ionicons
                        name={isBookmarked ? "bookmark" : "bookmark-outline"}
                        size={22}
                        color={COLORS.white}
                    />
                    <Text style={styles.bookmarkCount}>{bookmarkCount}</Text>
                </TouchableOpacity>

                <View style={styles.bottomSpacer} />

                {/* Avatar Button */}
                <TouchableOpacity
                    style={styles.avatarButton}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                        (navigation as any).navigate('AITryOn');
                    }}
                >
                    <Text style={styles.avatarText}>Avatar</Text>
                    <View style={styles.avatarIcon}>
                        <Ionicons name="person" size={14} color={COLORS.white} />
                    </View>
                    <View style={styles.avatarBadge}>
                        <Text style={styles.avatarBadgeText}>+22</Text>
                    </View>
                </TouchableOpacity>
            </Animated.View>

            {/* Product Detail Modal */}
            <ProductDetailModal
                visible={showModal}
                item={selectedItem}
                onClose={() => setShowModal(false)}
            />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    backgroundImage: {
        ...StyleSheet.absoluteFillObject,
        width: width,
        height: height,
    },
    safeArea: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 10,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingTop: 8,
    },
    headerSpacer: {
        width: 40,
    },
    navArrows: {
        flexDirection: 'row',
        gap: 8,
    },
    navArrow: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: COLORS.glass,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: COLORS.glassBorder,
    },
    closeButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: COLORS.glass,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: COLORS.glassBorder,
    },

    // Left Items Container
    itemsContainer: {
        position: 'absolute',
        left: 12,
        top: height * 0.15,
        bottom: height * 0.18,
        width: 60,
        zIndex: 5,
    },
    loadingContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    loadingText: {
        fontSize: 10,
        color: COLORS.white,
        marginTop: 8,
        textAlign: 'center',
    },
    itemsList: {
        gap: 12,
        paddingVertical: 8,
    },
    itemThumbnailContainer: {
        position: 'relative',
    },
    itemThumbnail: {
        width: 50,
        height: 50,
        borderRadius: 8,
        backgroundColor: COLORS.surface,
        overflow: 'hidden',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    itemThumbnailSelected: {
        borderColor: COLORS.accent,
    },
    itemThumbnailImage: {
        width: '100%',
        height: '100%',
    },
    shoppingIcon: {
        position: 'absolute',
        bottom: -4,
        right: -4,
        width: 18,
        height: 18,
        borderRadius: 9,
        backgroundColor: COLORS.accent,
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Right Navigation
    rightNav: {
        position: 'absolute',
        right: 16,
        top: '45%',
        gap: 8,
        zIndex: 5,
    },
    navButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: COLORS.glass,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: COLORS.glassBorder,
    },

    // Bottom Bar
    bottomBar: {
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
    bookmarkButton: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: COLORS.glass,
        paddingHorizontal: 14,
        paddingVertical: 10,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: COLORS.glassBorder,
    },
    bookmarkCount: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.white,
    },
    bottomSpacer: {
        flex: 1,
    },
    avatarButton: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        backgroundColor: COLORS.glass,
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: COLORS.glassBorder,
    },
    avatarText: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.white,
    },
    avatarIcon: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: 'rgba(255,255,255,0.3)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    avatarBadge: {
        backgroundColor: '#FF3B30',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 10,
    },
    avatarBadgeText: {
        fontSize: 10,
        fontWeight: '700',
        color: COLORS.white,
    },

    // Modal Styles
    modalOverlay: {
        flex: 1,
        backgroundColor: 'rgba(0,0,0,0.6)',
        justifyContent: 'flex-end',
    },
    modalContent: {
        height: height * 0.55,
        borderTopLeftRadius: 24,
        borderTopRightRadius: 24,
        overflow: 'hidden',
    },
    modalBlur: {
        flex: 1,
        padding: 20,
    },
    modalCloseButton: {
        alignSelf: 'flex-end',
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 16,
    },
    modalProductImage: {
        width: '100%',
        height: 200,
        borderRadius: 16,
        backgroundColor: '#333',
        marginBottom: 20,
    },
    modalProductInfo: {
        marginBottom: 24,
    },
    modalBrand: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.textSecondary,
        marginBottom: 4,
    },
    modalName: {
        fontSize: 20,
        fontWeight: '700',
        color: COLORS.white,
        marginBottom: 8,
    },
    modalPrice: {
        fontSize: 18,
        fontWeight: '600',
        color: COLORS.accent,
    },
    modalActions: {
        flexDirection: 'row',
        gap: 12,
    },
    modalSaveButton: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        backgroundColor: 'rgba(255,255,255,0.2)',
        paddingHorizontal: 20,
        paddingVertical: 14,
        borderRadius: 24,
    },
    modalSaveText: {
        fontSize: 15,
        fontWeight: '600',
        color: COLORS.white,
    },
    modalShopButton: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        backgroundColor: COLORS.white,
        paddingVertical: 14,
        borderRadius: 24,
    },
    modalShopText: {
        fontSize: 15,
        fontWeight: '600',
        color: COLORS.background,
    },
});

export default OutfitDetailScreen;
