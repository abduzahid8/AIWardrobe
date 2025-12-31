import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    TouchableOpacity,
    Image,
    Dimensions,
    Linking,
    ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import AppColors from '../constants/AppColors';
import {
    shoppingService,
    CompleteYourLookSuggestion,
    Product
} from '../src/services/shoppingService';

const { width } = Dimensions.get('window');

const COLORS = {
    background: AppColors.background,
    surface: AppColors.surface,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    border: AppColors.border,
    sale: '#FF5252',
};

interface CompleteYourLookProps {
    outfitItems: any[];
    onProductPress?: (product: Product) => void;
}

export const CompleteYourLook = ({ outfitItems, onProductPress }: CompleteYourLookProps) => {
    const [suggestions, setSuggestions] = useState<CompleteYourLookSuggestion[]>([]);
    const [loading, setLoading] = useState(true);
    const [expandedCategory, setExpandedCategory] = useState<string | null>(null);

    useEffect(() => {
        loadSuggestions();
    }, [outfitItems]);

    const loadSuggestions = async () => {
        setLoading(true);
        try {
            const results = await shoppingService.getCompleteYourLookSuggestions(outfitItems);
            setSuggestions(results);
        } catch (error) {
            console.error('Failed to load suggestions:', error);
        }
        setLoading(false);
    };

    const handleProductPress = async (product: Product) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

        // Track click
        await shoppingService.trackProductClick(product, 'complete_your_look');

        if (onProductPress) {
            onProductPress(product);
        } else {
            // Open product URL
            const url = shoppingService.getAffiliateLink(product);
            Linking.openURL(url).catch(console.error);
        }
    };

    const handleWishlist = async (product: Product) => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        await shoppingService.addToWishlist(product);
    };

    if (loading) {
        return (
            <View style={styles.loadingContainer}>
                <ActivityIndicator size="small" color={COLORS.primary} />
                <Text style={styles.loadingText}>Finding matching items...</Text>
            </View>
        );
    }

    if (suggestions.length === 0) {
        return null; // Don't show section if no suggestions
    }

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <View style={styles.headerLeft}>
                    <Ionicons name="sparkles" size={20} color={COLORS.accent} />
                    <Text style={styles.title}>Complete Your Look</Text>
                </View>
                <TouchableOpacity>
                    <Text style={styles.seeAllText}>See All</Text>
                </TouchableOpacity>
            </View>

            {suggestions.map((suggestion, idx) => (
                <View key={idx} style={styles.suggestionSection}>
                    <TouchableOpacity
                        style={styles.categoryHeader}
                        onPress={() => setExpandedCategory(
                            expandedCategory === suggestion.missingCategory
                                ? null
                                : suggestion.missingCategory
                        )}
                    >
                        <View>
                            <Text style={styles.categoryTitle}>
                                Add {suggestion.missingCategory.charAt(0).toUpperCase() + suggestion.missingCategory.slice(1)}
                            </Text>
                            <Text style={styles.categoryReason}>{suggestion.reason}</Text>
                        </View>
                        <Ionicons
                            name={expandedCategory === suggestion.missingCategory ? "chevron-up" : "chevron-down"}
                            size={20}
                            color={COLORS.textSecondary}
                        />
                    </TouchableOpacity>

                    <ScrollView
                        horizontal
                        showsHorizontalScrollIndicator={false}
                        contentContainerStyle={styles.productsRow}
                    >
                        {suggestion.suggestedProducts.map((product) => (
                            <ProductCard
                                key={product.id}
                                product={product}
                                onPress={() => handleProductPress(product)}
                                onWishlist={() => handleWishlist(product)}
                            />
                        ))}
                    </ScrollView>
                </View>
            ))}
        </View>
    );
};

// ============================================
// PRODUCT CARD
// ============================================

interface ProductCardProps {
    product: Product;
    onPress: () => void;
    onWishlist: () => void;
}

const ProductCard = ({ product, onPress, onWishlist }: ProductCardProps) => {
    const hasDiscount = product.originalPrice && product.originalPrice > product.price;
    const discountPercent = hasDiscount
        ? Math.round((1 - product.price / product.originalPrice!) * 100)
        : 0;

    return (
        <TouchableOpacity
            style={styles.productCard}
            onPress={onPress}
            activeOpacity={0.8}
        >
            {/* Image */}
            <View style={styles.productImageContainer}>
                {product.imageUrl ? (
                    <Image
                        source={{ uri: product.imageUrl }}
                        style={styles.productImage}
                        resizeMode="cover"
                    />
                ) : (
                    <View style={styles.productImagePlaceholder}>
                        <Ionicons name="shirt-outline" size={32} color={COLORS.textSecondary} />
                    </View>
                )}

                {/* Wishlist button */}
                <TouchableOpacity
                    style={styles.wishlistButton}
                    onPress={(e) => {
                        e.stopPropagation();
                        onWishlist();
                    }}
                >
                    <Ionicons name="heart-outline" size={20} color={COLORS.text} />
                </TouchableOpacity>

                {/* Discount badge */}
                {hasDiscount && (
                    <View style={styles.discountBadge}>
                        <Text style={styles.discountText}>-{discountPercent}%</Text>
                    </View>
                )}
            </View>

            {/* Info */}
            <View style={styles.productInfo}>
                <Text style={styles.productBrand}>{product.brand}</Text>
                <Text style={styles.productName} numberOfLines={2}>{product.name}</Text>

                <View style={styles.priceRow}>
                    <Text style={styles.productPrice}>
                        ${product.price.toFixed(0)}
                    </Text>
                    {hasDiscount && (
                        <Text style={styles.originalPrice}>
                            ${product.originalPrice!.toFixed(0)}
                        </Text>
                    )}
                </View>

                {product.rating && (
                    <View style={styles.ratingRow}>
                        <Ionicons name="star" size={12} color="#FFD700" />
                        <Text style={styles.ratingText}>
                            {product.rating} ({product.reviewCount})
                        </Text>
                    </View>
                )}
            </View>

            {/* Shop button */}
            <TouchableOpacity style={styles.shopButton} onPress={onPress}>
                <Text style={styles.shopButtonText}>Shop Now</Text>
                <Ionicons name="open-outline" size={14} color={COLORS.primary} />
            </TouchableOpacity>
        </TouchableOpacity>
    );
};

// ============================================
// SIMILAR PRODUCTS SECTION
// ============================================

interface SimilarProductsProps {
    item: { type: string; color?: string; style?: string };
    title?: string;
}

export const SimilarProducts = ({ item, title = "Shop Similar" }: SimilarProductsProps) => {
    const [products, setProducts] = useState<Product[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        loadProducts();
    }, [item]);

    const loadProducts = async () => {
        setLoading(true);
        try {
            const results = await shoppingService.findSimilarProducts(item);
            setProducts(results);
        } catch (error) {
            console.error('Failed to load similar products:', error);
        }
        setLoading(false);
    };

    if (loading || products.length === 0) {
        return null;
    }

    return (
        <View style={styles.container}>
            <View style={styles.header}>
                <View style={styles.headerLeft}>
                    <Ionicons name="cart-outline" size={20} color={COLORS.primary} />
                    <Text style={styles.title}>{title}</Text>
                </View>
            </View>

            <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={styles.productsRow}
            >
                {products.map((product) => (
                    <ProductCard
                        key={product.id}
                        product={product}
                        onPress={async () => {
                            await shoppingService.trackProductClick(product, 'similar_products');
                            Linking.openURL(shoppingService.getAffiliateLink(product));
                        }}
                        onWishlist={async () => {
                            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                            await shoppingService.addToWishlist(product);
                        }}
                    />
                ))}
            </ScrollView>
        </View>
    );
};

// ============================================
// STYLES
// ============================================

const CARD_WIDTH = width * 0.4;

const styles = StyleSheet.create({
    container: {
        marginVertical: 16,
    },
    loadingContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
        gap: 8,
    },
    loadingText: {
        fontSize: 14,
        color: COLORS.textSecondary,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        marginBottom: 12,
    },
    headerLeft: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    title: {
        fontSize: 18,
        fontWeight: '700',
        color: COLORS.text,
    },
    seeAllText: {
        fontSize: 14,
        color: COLORS.primary,
        fontWeight: '500',
    },

    // Suggestion Section
    suggestionSection: {
        marginBottom: 16,
    },
    categoryHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        marginBottom: 8,
    },
    categoryTitle: {
        fontSize: 15,
        fontWeight: '600',
        color: COLORS.text,
    },
    categoryReason: {
        fontSize: 13,
        color: COLORS.textSecondary,
    },

    // Products Row
    productsRow: {
        paddingHorizontal: 12,
        gap: 12,
    },

    // Product Card
    productCard: {
        width: CARD_WIDTH,
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        overflow: 'hidden',
    },
    productImageContainer: {
        width: '100%',
        height: CARD_WIDTH * 1.2,
        backgroundColor: COLORS.background,
    },
    productImage: {
        width: '100%',
        height: '100%',
    },
    productImagePlaceholder: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
    wishlistButton: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: 'rgba(255,255,255,0.9)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    discountBadge: {
        position: 'absolute',
        top: 8,
        left: 8,
        backgroundColor: COLORS.sale,
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 8,
    },
    discountText: {
        fontSize: 11,
        fontWeight: '700',
        color: '#FFF',
    },

    // Product Info
    productInfo: {
        padding: 12,
    },
    productBrand: {
        fontSize: 11,
        fontWeight: '600',
        color: COLORS.textSecondary,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    productName: {
        fontSize: 13,
        color: COLORS.text,
        marginTop: 4,
        lineHeight: 18,
    },
    priceRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        marginTop: 8,
    },
    productPrice: {
        fontSize: 16,
        fontWeight: '700',
        color: COLORS.text,
    },
    originalPrice: {
        fontSize: 13,
        color: COLORS.textSecondary,
        textDecorationLine: 'line-through',
    },
    ratingRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        marginTop: 6,
    },
    ratingText: {
        fontSize: 11,
        color: COLORS.textSecondary,
    },

    // Shop Button
    shopButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 10,
        borderTopWidth: 1,
        borderTopColor: COLORS.border,
        gap: 6,
    },
    shopButtonText: {
        fontSize: 13,
        fontWeight: '600',
        color: COLORS.primary,
    },
});

export default CompleteYourLook;
