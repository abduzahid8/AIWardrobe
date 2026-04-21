import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    ScrollView,
    TouchableOpacity,
    Image,
    StyleSheet,
    Dimensions,
    RefreshControl,
    ActivityIndicator,
    Linking,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withRepeat,
    withSequence,
    withTiming,
    FadeInDown,
    FadeInUp,
    SlideInRight,
} from 'react-native-reanimated';

import AppColors from '../constants/AppColors';
import flashSalesService from '../src/services/flashSalesService';
import { FlashSaleEvent, FlashSaleProduct } from '../src/types/flashSales';
import { RootStackParamList } from '../navigation/types';
import shoppingService from '../src/services/shoppingService';

const { width } = Dimensions.get('window');

// ============================================
// COUNTDOWN TIMER
// ============================================
const CountdownTimer = ({ event }: { event: FlashSaleEvent }) => {
    const [timeRemaining, setTimeRemaining] = useState(
        flashSalesService.getTimeRemaining(event)
    );

    useEffect(() => {
        const interval = setInterval(() => {
            setTimeRemaining(flashSalesService.getTimeRemaining(event));
        }, 1000);

        return () => clearInterval(interval);
    }, [event]);

    const { hours, minutes, seconds, isEnding } = timeRemaining;
    const label = event.status === 'upcoming' ? 'Starts in' : 'Ends in';

    return (
        <View style={styles.countdownRow}>
            <Text style={styles.countdownLabel}>{label}</Text>
            <View style={[styles.countdownPill, isEnding && styles.countdownPillEnding]}>
                <Ionicons
                    name="time-outline"
                    size={14}
                    color={isEnding ? '#FF3B30' : AppColors.text}
                />
                <Text style={[styles.countdownText, isEnding && styles.countdownTextEnding]}>
                    {hours > 0
                        ? `${hours}h ${minutes}m ${seconds}s`
                        : `${minutes}m ${seconds}s`
                    }
                </Text>
            </View>
        </View>
    );
};

// ============================================
// STOCK BADGE
// ============================================
const StockBadge = ({ status, count }: { status: string; count?: number }) => {
    if (status === 'sold_out') {
        return (
            <View style={[styles.stockBadge, styles.stockSoldOut]}>
                <Text style={styles.stockBadgeText}>Sold Out</Text>
            </View>
        );
    }

    if (status === 'low_stock') {
        return (
            <View style={[styles.stockBadge, styles.stockLow]}>
                <Text style={styles.stockBadgeText}>
                    {count ? `Only ${count} left` : 'Low Stock'}
                </Text>
            </View>
        );
    }

    return null;
};

// ============================================
// PRODUCT CARD
// ============================================
const ProductCard = ({
    product,
    index,
    onPress,
    onAddToWishlist,
}: {
    product: FlashSaleProduct;
    index: number;
    onPress: () => void;
    onAddToWishlist: () => void;
}) => {
    const [isWishlisted, setIsWishlisted] = useState(false);
    const heartScale = useSharedValue(1);

    const discount = Math.round(
        ((product.originalPrice - product.salePrice) / product.originalPrice) * 100
    );

    const handleWishlist = () => {
        heartScale.value = withSequence(
            withSpring(1.3),
            withSpring(1)
        );
        setIsWishlisted(!isWishlisted);
        onAddToWishlist();
    };

    const heartStyle = useAnimatedStyle(() => ({
        transform: [{ scale: heartScale.value }],
    }));

    const isSoldOut = product.stockStatus === 'sold_out';

    return (
        <Animated.View entering={FadeInUp.delay(index * 80).duration(400)}>
            <TouchableOpacity
                style={[styles.productCard, isSoldOut && styles.productCardSoldOut]}
                onPress={onPress}
                activeOpacity={0.9}
                disabled={isSoldOut}
            >
                {/* Image */}
                <View style={styles.productImageContainer}>
                    <Image
                        source={{ uri: product.imageUrl }}
                        style={styles.productImage}
                    />

                    {/* Discount badge */}
                    <View style={styles.productDiscountBadge}>
                        <Text style={styles.productDiscountText}>-{discount}%</Text>
                    </View>

                    {/* Stock badge */}
                    <View style={styles.productStockContainer}>
                        <StockBadge
                            status={product.stockStatus}
                            count={product.stockCount}
                        />
                    </View>

                    {/* Wishlist button */}
                    <Animated.View style={[styles.wishlistButton, heartStyle]}>
                        <TouchableOpacity onPress={handleWishlist}>
                            <Ionicons
                                name={isWishlisted ? "heart" : "heart-outline"}
                                size={20}
                                color={isWishlisted ? "#FF3B30" : "#666"}
                            />
                        </TouchableOpacity>
                    </Animated.View>
                </View>

                {/* Details */}
                <View style={styles.productDetails}>
                    <Text style={styles.productBrand}>{product.brand}</Text>
                    <Text style={styles.productName} numberOfLines={2}>
                        {product.name}
                    </Text>

                    <View style={styles.productPriceRow}>
                        <Text style={styles.productSalePrice}>
                            ${product.salePrice}
                        </Text>
                        <Text style={styles.productOriginalPrice}>
                            ${product.originalPrice}
                        </Text>
                    </View>

                    {/* Rating */}
                    {product.rating && (
                        <View style={styles.productRating}>
                            <Ionicons name="star" size={12} color="#FFD700" />
                            <Text style={styles.productRatingText}>
                                {product.rating} ({product.reviewCount})
                            </Text>
                        </View>
                    )}

                    {/* Sizes */}
                    {product.size && product.size.length > 0 && (
                        <Text style={styles.productSizes}>
                            Sizes: {product.size.join(', ')}
                        </Text>
                    )}
                </View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// ============================================
// MAIN SCREEN
// ============================================
type FlashSaleEventRouteProp = RouteProp<RootStackParamList, 'FlashSaleEvent'>;

const FlashSaleEventScreen = () => {
    const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();
    const route = useRoute<FlashSaleEventRouteProp>();
    const { eventId } = route.params;

    const [event, setEvent] = useState<FlashSaleEvent | null>(null);
    const [products, setProducts] = useState<FlashSaleProduct[]>([]);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);
    const [isSubscribed, setIsSubscribed] = useState(false);

    const loadEventData = async () => {
        try {
            const [eventData, productData] = await Promise.all([
                flashSalesService.getEventById(eventId),
                flashSalesService.getEventProducts(eventId),
            ]);

            setEvent(eventData);
            setProducts(productData);
            setIsSubscribed(flashSalesService.isSubscribed(eventId));
        } catch (error) {
            console.error('Failed to load event:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadEventData();
    }, [eventId]);

    const onRefresh = useCallback(async () => {
        setRefreshing(true);
        await loadEventData();
        setRefreshing(false);
    }, [eventId]);

    const handleSubscribe = async () => {
        if (isSubscribed) {
            await flashSalesService.unsubscribeFromEvent(eventId);
        } else {
            await flashSalesService.subscribeToEvent(eventId);
        }
        setIsSubscribed(!isSubscribed);
    };

    const handleProductPress = async (product: FlashSaleProduct) => {
        await flashSalesService.trackProductView(product);

        const affiliateUrl = flashSalesService.getAffiliateLink(product);

        try {
            const supported = await Linking.canOpenURL(affiliateUrl);
            if (supported) {
                await Linking.openURL(affiliateUrl);
            } else {
                Alert.alert('Unable to open link', 'This product link could not be opened.');
            }
        } catch (error) {
            console.error('Error opening product link:', error);
        }
    };

    const handleAddToWishlist = async (product: FlashSaleProduct) => {
        // Convert to standard Product format for shopping service
        const shopProduct = {
            id: product.id,
            name: product.name,
            brand: product.brand,
            price: product.salePrice,
            originalPrice: product.originalPrice,
            currency: product.currency,
            imageUrl: product.imageUrl,
            productUrl: product.productUrl,
            affiliateUrl: product.affiliateUrl,
            category: product.category,
            color: product.color,
            inStock: product.stockStatus !== 'sold_out',
            rating: product.rating,
            reviewCount: product.reviewCount,
            source: 'flash_sale',
        };

        await shoppingService.addToWishlist(shopProduct);
    };

    if (loading) {
        return (
            <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color={AppColors.accent} />
            </View>
        );
    }

    if (!event) {
        return (
            <View style={styles.errorContainer}>
                <Ionicons name="alert-circle-outline" size={48} color={AppColors.textSecondary} />
                <Text style={styles.errorText}>Event not found</Text>
                <TouchableOpacity
                    style={styles.errorButton}
                    onPress={() => navigation.goBack()}
                >
                    <Text style={styles.errorButtonText}>Go Back</Text>
                </TouchableOpacity>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <ScrollView
                style={styles.scrollView}
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
                refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
                }
            >
                {/* Hero Header */}
                <View style={styles.heroContainer}>
                    <Image
                        source={{ uri: event.heroImage }}
                        style={styles.heroImage}
                    />
                    <LinearGradient
                        colors={['transparent', 'rgba(0,0,0,0.7)', 'rgba(0,0,0,0.9)']}
                        style={styles.heroGradient}
                    />

                    {/* Back button */}
                    <SafeAreaView style={styles.heroHeader} edges={['top']}>
                        <TouchableOpacity
                            style={styles.headerButton}
                            onPress={() => navigation.goBack()}
                        >
                            <Ionicons name="chevron-back" size={24} color="#FFF" />
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={styles.headerButton}
                            onPress={() => { /* Share functionality */ }}
                        >
                            <Ionicons name="share-outline" size={22} color="#FFF" />
                        </TouchableOpacity>
                    </SafeAreaView>

                    {/* Event info overlay */}
                    <View style={styles.heroContent}>
                        <View style={styles.heroBadges}>
                            {event.status === 'active' && (
                                <View style={styles.liveBadge}>
                                    <View style={styles.liveDot} />
                                    <Text style={styles.liveText}>LIVE NOW</Text>
                                </View>
                            )}
                            {event.isExclusive && (
                                <View style={styles.exclusiveBadge}>
                                    <Ionicons name="diamond" size={12} color="#FFD700" />
                                    <Text style={styles.exclusiveText}>EXCLUSIVE</Text>
                                </View>
                            )}
                        </View>

                        <Text style={styles.heroTitle}>{event.title}</Text>
                        <Text style={styles.heroDescription}>{event.description}</Text>

                        <View style={styles.heroStats}>
                            <View style={styles.heroStat}>
                                <Text style={styles.heroStatValue}>{event.discountPercentage}%</Text>
                                <Text style={styles.heroStatLabel}>OFF</Text>
                            </View>
                            <View style={styles.heroStatDivider} />
                            <View style={styles.heroStat}>
                                <Text style={styles.heroStatValue}>{products.length}</Text>
                                <Text style={styles.heroStatLabel}>ITEMS</Text>
                            </View>
                            <View style={styles.heroStatDivider} />
                            <View style={styles.heroStat}>
                                <Text style={styles.heroStatValue}>
                                    {event.subscriberCount?.toLocaleString() || '0'}
                                </Text>
                                <Text style={styles.heroStatLabel}>WAITING</Text>
                            </View>
                        </View>
                    </View>
                </View>

                {/* Countdown & Actions */}
                <View style={styles.actionsContainer}>
                    <CountdownTimer event={event} />

                    {event.status === 'upcoming' && (
                        <TouchableOpacity
                            style={[
                                styles.notifyButton,
                                isSubscribed && styles.notifyButtonActive
                            ]}
                            onPress={handleSubscribe}
                        >
                            <Ionicons
                                name={isSubscribed ? "notifications" : "notifications-outline"}
                                size={18}
                                color={isSubscribed ? "#FFD700" : AppColors.text}
                            />
                            <Text style={[
                                styles.notifyButtonText,
                                isSubscribed && styles.notifyButtonTextActive
                            ]}>
                                {isSubscribed ? 'Notified' : 'Notify Me'}
                            </Text>
                        </TouchableOpacity>
                    )}
                </View>

                {/* Products Grid */}
                <View style={styles.productsSection}>
                    <Text style={styles.productsSectionTitle}>
                        {event.status === 'active' ? 'Shop the Sale' : 'What\'s Coming'}
                    </Text>
                    <Text style={styles.productsSectionSubtitle}>
                        {products.length} exclusive items
                    </Text>

                    <View style={styles.productsGrid}>
                        {products.map((product, index) => (
                            <ProductCard
                                key={product.id}
                                product={product}
                                index={index}
                                onPress={() => handleProductPress(product)}
                                onAddToWishlist={() => handleAddToWishlist(product)}
                            />
                        ))}
                    </View>
                </View>

                {/* Bottom spacing */}
                <View style={{ height: 100 }} />
            </ScrollView>
        </View>
    );
};

// ============================================
// STYLES
// ============================================
const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: AppColors.background,
    },
    scrollView: {
        flex: 1,
    },
    scrollContent: {
        paddingBottom: 20,
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: AppColors.background,
    },
    errorContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: AppColors.background,
        padding: 20,
    },
    errorText: {
        fontSize: 16,
        color: AppColors.textSecondary,
        marginTop: 12,
    },
    errorButton: {
        marginTop: 20,
        paddingHorizontal: 24,
        paddingVertical: 12,
        backgroundColor: AppColors.text,
        borderRadius: 24,
    },
    errorButtonText: {
        color: '#FFF',
        fontSize: 14,
        fontWeight: '600',
    },

    // Hero
    heroContainer: {
        height: 380,
        backgroundColor: '#0A1931',
    },
    heroImage: {
        width: '100%',
        height: '100%',
        position: 'absolute',
    },
    heroGradient: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '80%',
    },
    heroHeader: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingTop: 8,
    },
    headerButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: 'rgba(0,0,0,0.4)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    heroContent: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: 20,
    },
    heroBadges: {
        flexDirection: 'row',
        gap: 8,
        marginBottom: 12,
    },
    liveBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FF3B30',
        paddingHorizontal: 10,
        paddingVertical: 5,
        borderRadius: 12,
        gap: 4,
    },
    liveDot: {
        width: 6,
        height: 6,
        borderRadius: 3,
        backgroundColor: '#FFF',
    },
    liveText: {
        color: '#FFF',
        fontSize: 11,
        fontWeight: '700',
        letterSpacing: 1,
    },
    exclusiveBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.2)',
        paddingHorizontal: 10,
        paddingVertical: 5,
        borderRadius: 12,
        gap: 4,
    },
    exclusiveText: {
        color: '#FFD700',
        fontSize: 11,
        fontWeight: '700',
        letterSpacing: 1,
    },
    heroTitle: {
        fontSize: 28,
        fontWeight: '800',
        color: '#FFF',
        marginBottom: 8,
    },
    heroDescription: {
        fontSize: 14,
        color: 'rgba(255,255,255,0.8)',
        lineHeight: 20,
        marginBottom: 20,
    },
    heroStats: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    heroStat: {
        alignItems: 'center',
    },
    heroStatValue: {
        fontSize: 22,
        fontWeight: '700',
        color: '#FFF',
    },
    heroStatLabel: {
        fontSize: 10,
        color: 'rgba(255,255,255,0.6)',
        letterSpacing: 1,
        marginTop: 2,
    },
    heroStatDivider: {
        width: 1,
        height: 30,
        backgroundColor: 'rgba(255,255,255,0.2)',
        marginHorizontal: 24,
    },

    // Actions
    actionsContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
        borderBottomWidth: 1,
        borderBottomColor: AppColors.border,
    },
    countdownRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 10,
    },
    countdownLabel: {
        fontSize: 13,
        color: AppColors.textSecondary,
    },
    countdownPill: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: AppColors.surface,
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 16,
        gap: 6,
    },
    countdownPillEnding: {
        backgroundColor: 'rgba(255,59,48,0.1)',
    },
    countdownText: {
        fontSize: 14,
        fontWeight: '600',
        color: AppColors.text,
    },
    countdownTextEnding: {
        color: '#FF3B30',
    },
    notifyButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: AppColors.surface,
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
        gap: 6,
    },
    notifyButtonActive: {
        backgroundColor: 'rgba(255,215,0,0.15)',
    },
    notifyButtonText: {
        fontSize: 14,
        fontWeight: '600',
        color: AppColors.text,
    },
    notifyButtonTextActive: {
        color: '#B8860B',
    },

    // Products
    productsSection: {
        paddingTop: 20,
        paddingHorizontal: 16,
    },
    productsSectionTitle: {
        fontSize: 22,
        fontWeight: '700',
        color: AppColors.text,
        marginBottom: 4,
    },
    productsSectionSubtitle: {
        fontSize: 13,
        color: AppColors.textSecondary,
        marginBottom: 20,
    },
    productsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
    },

    // Product Card
    productCard: {
        width: (width - 48) / 2,
        marginBottom: 20,
        backgroundColor: '#FFF',
        borderRadius: 16,
        overflow: 'hidden',
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.06,
        shadowRadius: 8,
        elevation: 3,
    },
    productCardSoldOut: {
        opacity: 0.6,
    },
    productImageContainer: {
        aspectRatio: 0.85,
        backgroundColor: AppColors.surface,
    },
    productImage: {
        width: '100%',
        height: '100%',
    },
    productDiscountBadge: {
        position: 'absolute',
        top: 8,
        left: 8,
        backgroundColor: '#FF3B30',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 8,
    },
    productDiscountText: {
        color: '#FFF',
        fontSize: 12,
        fontWeight: '700',
    },
    productStockContainer: {
        position: 'absolute',
        bottom: 8,
        left: 8,
    },
    stockBadge: {
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 8,
    },
    stockSoldOut: {
        backgroundColor: 'rgba(0,0,0,0.7)',
    },
    stockLow: {
        backgroundColor: 'rgba(255,149,0,0.9)',
    },
    stockBadgeText: {
        color: '#FFF',
        fontSize: 10,
        fontWeight: '600',
    },
    wishlistButton: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: 'rgba(255,255,255,0.9)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    productDetails: {
        padding: 12,
    },
    productBrand: {
        fontSize: 10,
        fontWeight: '600',
        color: AppColors.textSecondary,
        letterSpacing: 1,
        textTransform: 'uppercase',
        marginBottom: 4,
    },
    productName: {
        fontSize: 13,
        fontWeight: '600',
        color: AppColors.text,
        lineHeight: 18,
        marginBottom: 8,
        minHeight: 36,
    },
    productPriceRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
        marginBottom: 6,
    },
    productSalePrice: {
        fontSize: 16,
        fontWeight: '700',
        color: AppColors.text,
    },
    productOriginalPrice: {
        fontSize: 13,
        color: AppColors.textMuted,
        textDecorationLine: 'line-through',
    },
    productRating: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        marginBottom: 4,
    },
    productRatingText: {
        fontSize: 11,
        color: AppColors.textSecondary,
    },
    productSizes: {
        fontSize: 11,
        color: AppColors.textSecondary,
    },
});

export default FlashSaleEventScreen;
