/**
 * InspoScreen - Alta-style Inspiration Page
 * 3 Tabs: Community, Shopping, Saved
 */

import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
    StatusBar,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, { FadeInUp, useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

// Colors
const COLORS = {
    background: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#000000',
    textSecondary: '#666666',
    textMuted: '#999999',
    border: '#E5E5E5',
};

type TabType = 'community' | 'shopping' | 'saved';

// Sample trending data
const TRENDING_ITEMS = [
    { id: '1', image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400', aspectRatio: 1.2 },
    { id: '2', image: 'https://images.unsplash.com/photo-1539571696357-5a69c17a67c6?w=400', aspectRatio: 1.5 },
    { id: '3', image: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400', aspectRatio: 1.3 },
    { id: '4', image: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400', aspectRatio: 1.4 },
];

// Sample shopping data
const SHOPPING_ITEMS = [
    { id: '1', brand: 'LORO PIANA', name: 'Cashmere Sweater', price: 1200, image: 'https://images.unsplash.com/photo-1576566588028-4147f3842f27?w=300' },
    { id: '2', brand: 'Everlane', name: 'The No-Sweat Sweater', price: 98, image: 'https://images.unsplash.com/photo-1620799140408-edc6dcb6d633?w=300' },
    { id: '3', brand: 'Todd Snyder', name: 'Silk-Cashmere Crewneck', price: 298, image: 'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=300' },
    { id: '4', brand: 'Simkhai', name: 'Bennett Sweater', price: 365, image: 'https://images.unsplash.com/photo-1591047139829-d91aecb6caea?w=300' },
    { id: '5', brand: 'Zegna', name: 'Oasi Cashmere Jacket', price: 6090, image: 'https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=300' },
    { id: '6', brand: 'ZEGNA', name: 'Riviera Slim-Fit Wool', price: 2730, image: 'https://images.unsplash.com/photo-1555069519-127aadedf1ee?w=300' },
    { id: '7', brand: 'ZEGNA', name: 'Riviera Linen-Blend Suit', price: 2730, image: 'https://images.unsplash.com/photo-1617127365659-c47fa864d8bc?w=300' },
    { id: '8', brand: 'Dolce & Gabbana', name: 'Glen Plaid Light Jacket', price: 3245, image: 'https://images.unsplash.com/photo-1593030761757-71fae45fa0e7?w=300' },
];

// Tab Button
const TabButton = ({ title, isActive, onPress }: { title: string; isActive: boolean; onPress: () => void }) => (
    <TouchableOpacity
        style={styles.tabButton}
        onPress={() => {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            onPress();
        }}
    >
        <Text style={[styles.tabText, isActive && styles.tabTextActive]}>{title}</Text>
    </TouchableOpacity>
);

// Trending Card
const TrendingCard = ({ item }: { item: typeof TRENDING_ITEMS[0] }) => (
    <Animated.View entering={FadeInUp.springify()} style={[styles.trendingCard, { aspectRatio: 1 / item.aspectRatio }]}>
        <Image source={{ uri: item.image }} style={styles.trendingImage} resizeMode="cover" />
    </Animated.View>
);

// Shopping Product Card
const ProductCard = ({ item, onSave }: { item: typeof SHOPPING_ITEMS[0]; onSave: () => void }) => {
    const [saved, setSaved] = useState(false);

    return (
        <Animated.View entering={FadeInUp.springify()} style={styles.productCard}>
            <View style={styles.productImageBox}>
                <Image source={{ uri: item.image }} style={styles.productImage} resizeMode="contain" />
                <TouchableOpacity
                    style={styles.saveButton}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        setSaved(!saved);
                        if (!saved) onSave();
                    }}
                >
                    <Ionicons name={saved ? "bookmark" : "bookmark-outline"} size={18} color={COLORS.text} />
                </TouchableOpacity>
            </View>
            <Text style={styles.productBrand}>{item.brand}</Text>
            <Text style={styles.productName} numberOfLines={2}>{item.name}</Text>
            <Text style={styles.productPrice}>${item.price.toLocaleString()}</Text>
        </Animated.View>
    );
};

const InspoScreen = () => {
    const navigation = useNavigation();
    const [activeTab, setActiveTab] = useState<TabType>('community');
    const [savedInspo, setSavedInspo] = useState<any[]>([]);

    useFocusEffect(useCallback(() => {
        loadSavedInspo();
    }, []));

    const loadSavedInspo = async () => {
        try {
            const saved = await AsyncStorage.getItem('savedInspo');
            if (saved) setSavedInspo(JSON.parse(saved));
        } catch (e) { }
    };

    const saveInspo = async (item: any) => {
        const updated = [...savedInspo, item];
        setSavedInspo(updated);
        await AsyncStorage.setItem('savedInspo', JSON.stringify(updated));
    };

    const handleAddInspo = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        (navigation as any).navigate('Camera');
    };

    return (
        <View style={styles.container}>
            <StatusBar barStyle="dark-content" backgroundColor={COLORS.background} />
            <SafeAreaView style={styles.safeArea} edges={['top']}>

                {/* Tabs */}
                <View style={styles.tabsContainer}>
                    <TabButton title="Community" isActive={activeTab === 'community'} onPress={() => setActiveTab('community')} />
                    <TabButton title="Shopping" isActive={activeTab === 'shopping'} onPress={() => setActiveTab('shopping')} />
                    <TabButton title="Saved" isActive={activeTab === 'saved'} onPress={() => setActiveTab('saved')} />
                </View>

                {/* Content */}
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {activeTab === 'community' && (
                        <>
                            {/* Trending Section */}
                            <View style={styles.trendingGrid}>
                                <View style={styles.trendingColumn}>
                                    {TRENDING_ITEMS.filter((_, i) => i % 2 === 0).map(item => (
                                        <TrendingCard key={item.id} item={item} />
                                    ))}
                                </View>
                                <View style={styles.trendingColumn}>
                                    {TRENDING_ITEMS.filter((_, i) => i % 2 === 1).map(item => (
                                        <TrendingCard key={item.id} item={item} />
                                    ))}
                                </View>
                            </View>

                            {/* Trending Label */}
                            <View style={styles.trendingLabel}>
                                <View style={styles.updatedBadge}>
                                    <Text style={styles.updatedText}>UPDATED TODAY</Text>
                                    <View style={styles.orangeDot} />
                                </View>
                                <Text style={styles.trendingTitle}>Trending</Text>
                                <Text style={styles.trendingSubtitle}>See what's popular in the Alta community</Text>
                                <TouchableOpacity style={styles.discoverButton}>
                                    <Text style={styles.discoverText}>Discover trends</Text>
                                    <Ionicons name="arrow-forward" size={16} color={COLORS.text} />
                                </TouchableOpacity>
                            </View>
                        </>
                    )}

                    {activeTab === 'shopping' && (
                        <>
                            <Text style={styles.sectionTitle}>Monnaie's Creative Collaboration</Text>
                            <View style={styles.productGrid}>
                                {SHOPPING_ITEMS.map(item => (
                                    <ProductCard key={item.id} item={item} onSave={() => saveInspo(item)} />
                                ))}
                            </View>
                        </>
                    )}

                    {activeTab === 'saved' && (
                        savedInspo.length === 0 ? (
                            <View style={styles.emptyState}>
                                <Image
                                    source={{ uri: 'https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=200' }}
                                    style={styles.emptyImage}
                                    resizeMode="contain"
                                />
                                <Text style={styles.emptyTitle}>No saved inspo yet</Text>
                                <Text style={styles.emptySubtitle}>Get inspired — save something you like or share your own style.</Text>
                                <TouchableOpacity style={styles.addInspoButton} onPress={handleAddInspo}>
                                    <Ionicons name="add" size={20} color={COLORS.background} />
                                    <Text style={styles.addInspoText}>Add inspo</Text>
                                </TouchableOpacity>
                            </View>
                        ) : (
                            <View style={styles.productGrid}>
                                {savedInspo.map((item, index) => (
                                    <ProductCard key={index} item={item} onSave={() => { }} />
                                ))}
                            </View>
                        )
                    )}

                    <View style={{ height: 100 }} />
                </ScrollView>

            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: COLORS.background },
    safeArea: { flex: 1 },

    // Tabs - Exact Alta style
    tabsContainer: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 32,
        paddingVertical: 16,
        borderBottomWidth: 0.5,
        borderBottomColor: COLORS.border,
    },
    tabButton: { paddingVertical: 4, paddingHorizontal: 4 },
    tabText: { fontSize: 15, fontWeight: '400', color: COLORS.textMuted },
    tabTextActive: { color: COLORS.text, fontWeight: '600' },

    // Content
    scrollContent: { paddingHorizontal: 20, paddingTop: 20 },

    // Trending Grid (Masonry-like) - 2 columns
    trendingGrid: { flexDirection: 'row', gap: 10 },
    trendingColumn: { flex: 1, gap: 10 },
    trendingCard: {
        borderRadius: 12,
        overflow: 'hidden',
        backgroundColor: COLORS.surface,
        minHeight: 180,
    },
    trendingImage: { width: '100%', height: '100%' },

    // Trending Label
    trendingLabel: { marginTop: 20, paddingTop: 16 },
    updatedBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, marginBottom: 8 },
    updatedText: { fontSize: 11, fontWeight: '600', color: COLORS.textMuted, letterSpacing: 1, textTransform: 'uppercase' },
    orangeDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: '#FF9500' },
    trendingTitle: { fontSize: 28, fontWeight: '700', color: COLORS.text, marginBottom: 6 },
    trendingSubtitle: { fontSize: 15, color: COLORS.textSecondary, marginBottom: 20, lineHeight: 22 },
    discoverButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'flex-end',
        gap: 6,
        paddingVertical: 16,
        borderTopWidth: 0.5,
        borderTopColor: COLORS.border,
    },
    discoverText: { fontSize: 14, fontWeight: '500', color: COLORS.text },

    // Section Title
    sectionTitle: { fontSize: 20, fontWeight: '600', color: COLORS.text, marginBottom: 20 },

    // Product Grid - 2 columns for better readability
    productGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 12,
    },
    productCard: {
        width: (width - 52) / 2,
        marginBottom: 20,
    },
    productImageBox: {
        width: '100%',
        aspectRatio: 0.85,
        backgroundColor: COLORS.surface,
        borderRadius: 12,
        overflow: 'hidden',
        marginBottom: 12,
    },
    productImage: { width: '100%', height: '100%' },
    saveButton: {
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
    productBrand: {
        fontSize: 12,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 4,
        textTransform: 'uppercase',
        letterSpacing: 0.5,
    },
    productName: {
        fontSize: 12,
        color: COLORS.textSecondary,
        marginBottom: 6,
        lineHeight: 16,
    },
    productPrice: {
        fontSize: 13,
        fontWeight: '600',
        color: COLORS.text,
    },

    // Empty State
    emptyState: {
        alignItems: 'center',
        paddingVertical: 100,
        paddingHorizontal: 40,
    },
    emptyImage: {
        width: 140,
        height: 140,
        marginBottom: 32,
        opacity: 0.8,
    },
    emptyTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: 12,
    },
    emptySubtitle: {
        fontSize: 15,
        color: COLORS.textSecondary,
        textAlign: 'center',
        lineHeight: 22,
        marginBottom: 40,
    },
    addInspoButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        backgroundColor: COLORS.text,
        paddingVertical: 16,
        paddingHorizontal: 80,
        borderRadius: 28,
        width: '100%',
    },
    addInspoText: {
        fontSize: 16,
        fontWeight: '600',
        color: COLORS.background,
    },
});

export default InspoScreen;
