/**
 * ALTA DAILY - PIXEL PERFECT SHOP SCREEN
 * Based on exact design specification
 */

import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
    TouchableOpacity,
    StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeInUp,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

// EXACT ALTA COLORS
const ALTA = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#0A1931',
    textSecondary: '#666666',
    textMuted: '#999999',
    border: '#E5E5E5',
    heart: '#FF3B30',
};

const CATEGORIES = ['All', 'New In', 'Tops', 'Bottoms', 'Shoes', 'Accessories'];

const SAMPLE_PRODUCTS = [
    { id: '1', name: 'Black Blazer', brand: 'Massimo Dutti', price: 189, image: 'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400' },
    { id: '2', name: 'White Sneakers', brand: 'Adidas', price: 120, image: 'https://images.unsplash.com/photo-1549298916-b41d501d3772?w=400' },
    { id: '3', name: 'Denim Jeans', brand: "Levi's", price: 98, image: 'https://images.unsplash.com/photo-1542272604-787c3835535d?w=400' },
    { id: '4', name: 'Cashmere Sweater', brand: 'COS', price: 150, image: 'https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=400' },
];

const GRID_GAP = 12;
const ITEM_WIDTH = (width - 32 - GRID_GAP) / 2;

// Type for product
interface ProductType {
    id: string;
    name: string;
    brand: string;
    price: number;
    image: string;
}

// Type for category pill props
interface CategoryPillProps {
    title: string;
    isActive: boolean;
    onPress: () => void;
}

// Type for product card props
interface ProductCardProps {
    product: ProductType;
    onPress: () => void;
    onLike: (product: ProductType, liked: boolean) => void;
}

// Category Pill with scale animation
const CategoryPill = ({ title, isActive, onPress }: CategoryPillProps) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    return (
        <TouchableOpacity
            activeOpacity={1}
            onPressIn={() => {
                scale.value = withSpring(0.97);
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            }}
            onPressOut={() => scale.value = withSpring(1)}
            onPress={onPress}
        >
            <Animated.View style={[styles.categoryPill, isActive && styles.categoryPillActive, animatedStyle]}>
                <Text style={[styles.categoryText, isActive && styles.categoryTextActive]}>{title}</Text>
            </Animated.View>
        </TouchableOpacity>
    );
};

// Product Card with scale animation
const ProductCard = ({ product, onPress, onLike }: ProductCardProps) => {
    const [liked, setLiked] = useState(false);
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    return (
        <Animated.View entering={FadeInUp.springify()}>
            <TouchableOpacity
                activeOpacity={1}
                onPressIn={() => {
                    scale.value = withSpring(0.97);
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                }}
                onPressOut={() => scale.value = withSpring(1)}
                onPress={onPress}
            >
                <Animated.View style={[styles.productCard, animatedStyle]}>
                    <View style={styles.productImageBox}>
                        <Image source={{ uri: product.image }} style={styles.productImage} resizeMode="cover" />
                        <TouchableOpacity
                            style={styles.likeButton}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                setLiked(!liked);
                                onLike(product, !liked);
                            }}
                        >
                            <Ionicons name={liked ? 'heart' : 'heart-outline'} size={18} color={liked ? ALTA.heart : ALTA.text} />
                        </TouchableOpacity>
                    </View>
                    <Text style={styles.productBrand}>{product.brand}</Text>
                    <Text style={styles.productName} numberOfLines={1}>{product.name}</Text>
                    <Text style={styles.productPrice}>${product.price}</Text>
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

const AltaShopScreen = () => {
    const navigation = useNavigation();
    const [selectedCategory, setSelectedCategory] = useState('All');
    const [products] = useState(SAMPLE_PRODUCTS);

    const handleLike = async (product: ProductType, liked: boolean) => {
        if (liked) {
            const favorites = await AsyncStorage.getItem('favoriteItems');
            const list = favorites ? JSON.parse(favorites) : [];
            list.push(product);
            await AsyncStorage.setItem('favoriteItems', JSON.stringify(list));
        }
    };

    const handleProductPress = (product: ProductType) => {
        (navigation as any).navigate('AITryOn', { selectedItem: product });
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={ALTA.background} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>

                {/* Header */}
                <View style={styles.header}>
                    <Text style={styles.headerTitle}>Shop</Text>
                    <TouchableOpacity style={styles.avatarPill}>
                        <Ionicons name="person-outline" size={14} color={ALTA.text} />
                        <Text style={styles.avatarPillText}>Your avatar</Text>
                    </TouchableOpacity>
                </View>

                <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>

                    {/* Categories */}
                    <ScrollView horizontal showsHorizontalScrollIndicator={false} contentContainerStyle={styles.categoriesRow}>
                        {CATEGORIES.map(cat => (
                            <CategoryPill
                                key={cat}
                                title={cat}
                                isActive={selectedCategory === cat}
                                onPress={() => setSelectedCategory(cat)}
                            />
                        ))}
                    </ScrollView>

                    {/* Products Grid */}
                    <View style={styles.productGrid}>
                        {products.map(product => (
                            <ProductCard
                                key={product.id}
                                product={product}
                                onPress={() => handleProductPress(product)}
                                onLike={handleLike}
                            />
                        ))}
                    </View>

                    <View style={{ height: 100 }} />
                </ScrollView>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: ALTA.background },
    safeArea: { flex: 1 },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
        borderBottomWidth: 0.5,
        borderBottomColor: ALTA.border,
    },
    headerTitle: { fontSize: 28, fontWeight: '700', color: ALTA.text },
    avatarPill: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        backgroundColor: ALTA.surface,
        paddingHorizontal: 12,
        paddingVertical: 8,
        borderRadius: 20,
    },
    avatarPillText: { fontSize: 12, fontWeight: '500', color: ALTA.text },

    // Content
    scrollContent: { paddingTop: 16 },

    // Categories
    categoriesRow: { paddingHorizontal: 16, paddingBottom: 16, gap: 8 },
    categoryPill: {
        paddingHorizontal: 16,
        paddingVertical: 10,
        backgroundColor: ALTA.background,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: ALTA.border,
        marginRight: 8,
    },
    categoryPillActive: { backgroundColor: ALTA.text, borderColor: ALTA.text },
    categoryText: { fontSize: 14, fontWeight: '500', color: ALTA.text },
    categoryTextActive: { color: ALTA.background },

    // Products
    productGrid: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: 16, gap: GRID_GAP },
    productCard: { width: ITEM_WIDTH, marginBottom: 16 },
    productImageBox: {
        width: '100%',
        aspectRatio: 0.75,
        backgroundColor: ALTA.surface,
        borderRadius: 12,
        overflow: 'hidden',
        marginBottom: 10,
    },
    productImage: { width: '100%', height: '100%' },
    likeButton: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: ALTA.background,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 4,
    },
    productBrand: { fontSize: 11, fontWeight: '600', color: ALTA.textMuted, textTransform: 'uppercase', letterSpacing: 0.5, marginBottom: 2 },
    productName: { fontSize: 14, fontWeight: '500', color: ALTA.text, marginBottom: 4 },
    productPrice: { fontSize: 14, fontWeight: '600', color: ALTA.text },
});

export default AltaShopScreen;
