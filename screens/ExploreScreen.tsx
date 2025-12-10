import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    ScrollView,
    Image,
    TouchableOpacity,
    Dimensions,
    RefreshControl,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { useTranslation } from 'react-i18next';
import { Ionicons } from '@expo/vector-icons';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { LinearGradient } from 'expo-linear-gradient';

import { colors, spacing, shadows, borderRadius } from '../src/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Celebrity Outfits - Real celebrity style inspiration
const celebrityOutfits = [
    {
        id: '1',
        celebrity: 'Zendaya',
        event: 'Met Gala 2024',
        image: 'https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=600',
        description: 'Stunning red carpet look with dramatic silhouette',
        tags: ['Formal', 'Red Carpet'],
        likes: 12400,
    },
    {
        id: '2',
        celebrity: 'Timothée Chalamet',
        event: 'Dune Premiere',
        image: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=600',
        description: 'Modern tailoring meets artistic expression',
        tags: ['Avant-Garde', 'Formal'],
        likes: 8900,
    },
    {
        id: '3',
        celebrity: 'Bella Hadid',
        event: 'Street Style',
        image: 'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=600',
        description: 'Off-duty model chic with vintage vibes',
        tags: ['Street', 'Casual'],
        likes: 15600,
    },
    {
        id: '4',
        celebrity: 'Harry Styles',
        event: 'Coachella',
        image: 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=600',
        description: 'Bold patterns and gender-fluid fashion',
        tags: ['Festival', 'Bold'],
        likes: 21000,
    },
];

// Ready-to-wear inspired looks
const readyOutfits = [
    {
        id: '1',
        title: 'Coffee Run Chic',
        items: ['Oversized blazer', 'White tee', 'Wide-leg jeans'],
        image: 'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=400',
        difficulty: 'Easy',
        saves: 2340,
    },
    {
        id: '2',
        title: 'Date Night',
        items: ['Silk blouse', 'Leather pants', 'Heeled boots'],
        image: 'https://images.unsplash.com/photo-1509631179647-0177331693ae?w=400',
        difficulty: 'Medium',
        saves: 1890,
    },
    {
        id: '3',
        title: 'Work From Home',
        items: ['Cashmere sweater', 'Joggers', 'Clean sneakers'],
        image: 'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=400',
        difficulty: 'Easy',
        saves: 4210,
    },
    {
        id: '4',
        title: 'Weekend Brunch',
        items: ['Linen shirt', 'Midi skirt', 'Sandals'],
        image: 'https://images.unsplash.com/photo-1496747611176-843222e1e57c?w=400',
        difficulty: 'Easy',
        saves: 3150,
    },
];

// Style categories/moods
const styleCategories = [
    { id: '1', name: 'Minimal', emoji: '◻️', color: '#F5F5F5' },
    { id: '2', name: 'Street', emoji: '🔥', color: '#1A1A1A' },
    { id: '3', name: 'Elegant', emoji: '✨', color: '#C9A55C' },
    { id: '4', name: 'Casual', emoji: '☕', color: '#8B7355' },
    { id: '5', name: 'Bold', emoji: '🎨', color: '#FF4757' },
];

const ExploreScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const [refreshing, setRefreshing] = useState(false);
    const [likedCelebs, setLikedCelebs] = useState<string[]>([]);
    const [savedOutfits, setSavedOutfits] = useState<string[]>([]);

    const onRefresh = useCallback(() => {
        setRefreshing(true);
        setTimeout(() => setRefreshing(false), 1000);
    }, []);

    const toggleCelebLike = (id: string) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setLikedCelebs(prev =>
            prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
        );
    };

    const toggleSaveOutfit = (id: string) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setSavedOutfits(prev =>
            prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
        );
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                <ScrollView
                    showsVerticalScrollIndicator={false}
                    refreshControl={
                        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
                    }
                >
                    {/* Header */}
                    <Animated.View entering={FadeInDown} style={styles.header}>
                        <Text style={styles.title}>Explore</Text>
                        <TouchableOpacity
                            style={styles.searchButton}
                            onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                        >
                            <Ionicons name="search" size={22} color={colors.text.primary} />
                        </TouchableOpacity>
                    </Animated.View>

                    {/* 🎬 VIDEO WARDROBE BANNER - Most Important Feature */}
                    <Animated.View entering={FadeInDown.delay(100)}>
                        <TouchableOpacity
                            style={styles.videoBanner}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
                                (navigation as any).navigate('WardrobeVideo');
                            }}
                            activeOpacity={0.95}
                        >
                            <LinearGradient
                                colors={['#1A1A1A', '#2D2D2D']}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 1 }}
                                style={styles.videoBannerGradient}
                            >
                                <View style={styles.videoBannerContent}>
                                    <View style={styles.videoBannerIcon}>
                                        <Ionicons name="videocam" size={28} color="#FFF" />
                                    </View>
                                    <View style={styles.videoBannerText}>
                                        <Text style={styles.videoBannerTitle}>
                                            Scan Your Wardrobe
                                        </Text>
                                        <Text style={styles.videoBannerSubtitle}>
                                            One video → Entire closet digitized by AI
                                        </Text>
                                    </View>
                                    <Ionicons name="chevron-forward" size={24} color="#FFF" />
                                </View>
                                <View style={styles.videoBannerBadge}>
                                    <Text style={styles.videoBannerBadgeText}>✨ AI Powered</Text>
                                </View>
                            </LinearGradient>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* Style Categories */}
                    <Animated.View entering={FadeInDown.delay(150)}>
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.categoriesScroll}
                        >
                            {styleCategories.map((cat) => (
                                <TouchableOpacity
                                    key={cat.id}
                                    style={[styles.categoryChip, { backgroundColor: cat.color }]}
                                    onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                                >
                                    <Text style={styles.categoryEmoji}>{cat.emoji}</Text>
                                    <Text style={[
                                        styles.categoryName,
                                        { color: cat.color === '#1A1A1A' ? '#FFF' : '#1A1A1A' }
                                    ]}>
                                        {cat.name}
                                    </Text>
                                </TouchableOpacity>
                            ))}
                        </ScrollView>
                    </Animated.View>

                    {/* ⭐ CELEBRITY LOOKS - Main Social Feature */}
                    <Animated.View entering={FadeInUp.delay(200)} style={styles.section}>
                        <View style={styles.sectionHeader}>
                            <View>
                                <Text style={styles.sectionTitle}>Celebrity Looks</Text>
                                <Text style={styles.sectionSubtitle}>Get inspired by the stars</Text>
                            </View>
                            <TouchableOpacity style={styles.seeAllButton}>
                                <Text style={styles.seeAllText}>See All</Text>
                            </TouchableOpacity>
                        </View>

                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.celebScroll}
                            decelerationRate="fast"
                            snapToInterval={SCREEN_WIDTH * 0.8 + spacing.m}
                        >
                            {celebrityOutfits.map((celeb) => (
                                <TouchableOpacity
                                    key={celeb.id}
                                    style={styles.celebCard}
                                    onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium)}
                                    activeOpacity={0.95}
                                >
                                    <Image
                                        source={{ uri: celeb.image }}
                                        style={styles.celebImage}
                                        resizeMode="cover"
                                    />
                                    <LinearGradient
                                        colors={['transparent', 'rgba(0,0,0,0.85)']}
                                        style={styles.celebGradient}
                                    >
                                        <View style={styles.celebInfo}>
                                            <Text style={styles.celebName}>{celeb.celebrity}</Text>
                                            <Text style={styles.celebEvent}>{celeb.event}</Text>
                                            <Text style={styles.celebDesc}>{celeb.description}</Text>
                                            <View style={styles.celebTags}>
                                                {celeb.tags.map(tag => (
                                                    <View key={tag} style={styles.celebTag}>
                                                        <Text style={styles.celebTagText}>{tag}</Text>
                                                    </View>
                                                ))}
                                            </View>
                                        </View>
                                        <TouchableOpacity
                                            style={styles.celebLikeButton}
                                            onPress={() => toggleCelebLike(celeb.id)}
                                        >
                                            <Ionicons
                                                name={likedCelebs.includes(celeb.id) ? 'heart' : 'heart-outline'}
                                                size={24}
                                                color={likedCelebs.includes(celeb.id) ? '#EF4444' : '#FFF'}
                                            />
                                            <Text style={styles.celebLikeCount}>
                                                {(likedCelebs.includes(celeb.id) ? celeb.likes + 1 : celeb.likes).toLocaleString()}
                                            </Text>
                                        </TouchableOpacity>
                                    </LinearGradient>
                                </TouchableOpacity>
                            ))}
                        </ScrollView>
                    </Animated.View>

                    {/* 👗 READY OUTFITS - Shop the Look / Copy This */}
                    <Animated.View entering={FadeInUp.delay(300)} style={styles.section}>
                        <View style={styles.sectionHeader}>
                            <View>
                                <Text style={styles.sectionTitle}>Ready Outfits</Text>
                                <Text style={styles.sectionSubtitle}>Copy these complete looks</Text>
                            </View>
                        </View>

                        <View style={styles.readyGrid}>
                            {readyOutfits.map((outfit, idx) => (
                                <Animated.View
                                    key={outfit.id}
                                    entering={FadeInUp.delay(350 + idx * 50)}
                                >
                                    <TouchableOpacity
                                        style={styles.readyCard}
                                        onPress={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                                        activeOpacity={0.9}
                                    >
                                        <Image
                                            source={{ uri: outfit.image }}
                                            style={styles.readyImage}
                                            resizeMode="cover"
                                        />
                                        <View style={styles.readyContent}>
                                            <Text style={styles.readyTitle}>{outfit.title}</Text>
                                            <View style={styles.readyItems}>
                                                {outfit.items.slice(0, 2).map((item, i) => (
                                                    <Text key={i} style={styles.readyItem}>• {item}</Text>
                                                ))}
                                                {outfit.items.length > 2 && (
                                                    <Text style={styles.readyMore}>+{outfit.items.length - 2} more</Text>
                                                )}
                                            </View>
                                            <View style={styles.readyFooter}>
                                                <View style={[
                                                    styles.difficultyBadge,
                                                    { backgroundColor: outfit.difficulty === 'Easy' ? '#F0FDF4' : '#FEF3C7' }
                                                ]}>
                                                    <Text style={[
                                                        styles.difficultyText,
                                                        { color: outfit.difficulty === 'Easy' ? '#22C55E' : '#F59E0B' }
                                                    ]}>
                                                        {outfit.difficulty}
                                                    </Text>
                                                </View>
                                                <TouchableOpacity
                                                    style={styles.saveButton}
                                                    onPress={() => toggleSaveOutfit(outfit.id)}
                                                >
                                                    <Ionicons
                                                        name={savedOutfits.includes(outfit.id) ? 'bookmark' : 'bookmark-outline'}
                                                        size={18}
                                                        color={savedOutfits.includes(outfit.id) ? colors.text.accent : colors.text.secondary}
                                                    />
                                                </TouchableOpacity>
                                            </View>
                                        </View>
                                    </TouchableOpacity>
                                </Animated.View>
                            ))}
                        </View>
                    </Animated.View>

                    {/* AI Tools */}
                    <Animated.View entering={FadeInUp.delay(500)} style={styles.section}>
                        <Text style={styles.sectionTitle}>AI Tools</Text>
                        <View style={styles.aiToolsRow}>
                            <TouchableOpacity
                                style={styles.aiTool}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                    (navigation as any).navigate('AIChat');
                                }}
                            >
                                <View style={[styles.aiToolIcon, { backgroundColor: '#EFF6FF' }]}>
                                    <Ionicons name="sparkles" size={24} color="#3B82F6" />
                                </View>
                                <Text style={styles.aiToolTitle}>AI Stylist</Text>
                            </TouchableOpacity>

                            <TouchableOpacity
                                style={styles.aiTool}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                    (navigation as any).navigate('AITryOn');
                                }}
                            >
                                <View style={[styles.aiToolIcon, { backgroundColor: '#FFF1F2' }]}>
                                    <Ionicons name="shirt" size={24} color="#EC4899" />
                                </View>
                                <Text style={styles.aiToolTitle}>Try-On</Text>
                            </TouchableOpacity>

                            <TouchableOpacity
                                style={styles.aiTool}
                                onPress={() => {
                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                    (navigation as any).navigate('DesignRoom');
                                }}
                            >
                                <View style={[styles.aiToolIcon, { backgroundColor: '#F0FDF4' }]}>
                                    <Ionicons name="color-palette" size={24} color="#22C55E" />
                                </View>
                                <Text style={styles.aiToolTitle}>Design</Text>
                            </TouchableOpacity>
                        </View>
                    </Animated.View>

                    <View style={{ height: spacing.xl * 2 }} />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background,
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: spacing.l,
        paddingVertical: spacing.m,
    },
    title: {
        fontSize: 28,
        fontWeight: '800',
        color: colors.text.primary,
    },
    searchButton: {
        padding: spacing.xs,
    },
    // Video Wardrobe Banner
    videoBanner: {
        marginHorizontal: spacing.l,
        marginBottom: spacing.m,
        borderRadius: borderRadius.xl,
        overflow: 'hidden',
    },
    videoBannerGradient: {
        padding: spacing.m,
        position: 'relative',
    },
    videoBannerContent: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.m,
    },
    videoBannerIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: 'rgba(255,255,255,0.15)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    videoBannerText: {
        flex: 1,
    },
    videoBannerTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#FFF',
        marginBottom: 2,
    },
    videoBannerSubtitle: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.7)',
    },
    videoBannerBadge: {
        position: 'absolute',
        top: spacing.s,
        right: spacing.s,
        backgroundColor: 'rgba(255,255,255,0.2)',
        paddingHorizontal: spacing.s,
        paddingVertical: 4,
        borderRadius: 12,
    },
    videoBannerBadgeText: {
        fontSize: 11,
        fontWeight: '600',
        color: '#FFF',
    },
    // Category Chips
    categoriesScroll: {
        paddingHorizontal: spacing.l,
        gap: spacing.s,
        marginBottom: spacing.l,
    },
    categoryChip: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
        borderRadius: borderRadius.full,
        gap: spacing.xs,
    },
    categoryEmoji: {
        fontSize: 16,
    },
    categoryName: {
        fontSize: 14,
        fontWeight: '600',
    },
    // Section
    section: {
        marginBottom: spacing.l,
    },
    sectionHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'flex-start',
        paddingHorizontal: spacing.l,
        marginBottom: spacing.m,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: colors.text.primary,
    },
    sectionSubtitle: {
        fontSize: 13,
        color: colors.text.secondary,
        marginTop: 2,
    },
    seeAllButton: {
        paddingVertical: 4,
    },
    seeAllText: {
        fontSize: 14,
        fontWeight: '600',
        color: colors.text.accent,
    },
    // Celebrity Looks
    celebScroll: {
        paddingHorizontal: spacing.l,
        gap: spacing.m,
    },
    celebCard: {
        width: SCREEN_WIDTH * 0.8,
        height: 320,
        borderRadius: borderRadius.xl,
        overflow: 'hidden',
        ...shadows.medium,
    },
    celebImage: {
        width: '100%',
        height: '100%',
    },
    celebGradient: {
        ...StyleSheet.absoluteFillObject,
        justifyContent: 'flex-end',
        padding: spacing.m,
    },
    celebInfo: {
        flex: 1,
        justifyContent: 'flex-end',
    },
    celebName: {
        fontSize: 24,
        fontWeight: '800',
        color: '#FFF',
        marginBottom: 2,
    },
    celebEvent: {
        fontSize: 14,
        fontWeight: '600',
        color: 'rgba(255,255,255,0.8)',
        marginBottom: spacing.xs,
    },
    celebDesc: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.7)',
        marginBottom: spacing.s,
    },
    celebTags: {
        flexDirection: 'row',
        gap: spacing.xs,
    },
    celebTag: {
        backgroundColor: 'rgba(255,255,255,0.2)',
        paddingHorizontal: spacing.s,
        paddingVertical: 4,
        borderRadius: 12,
    },
    celebTagText: {
        fontSize: 11,
        fontWeight: '600',
        color: '#FFF',
    },
    celebLikeButton: {
        position: 'absolute',
        top: spacing.m,
        right: spacing.m,
        alignItems: 'center',
        backgroundColor: 'rgba(0,0,0,0.4)',
        paddingHorizontal: spacing.m,
        paddingVertical: spacing.s,
        borderRadius: borderRadius.m,
        gap: 4,
    },
    celebLikeCount: {
        fontSize: 12,
        fontWeight: '600',
        color: '#FFF',
    },
    // Ready Outfits
    readyGrid: {
        paddingHorizontal: spacing.l,
        gap: spacing.m,
    },
    readyCard: {
        flexDirection: 'row',
        backgroundColor: colors.surface,
        borderRadius: borderRadius.l,
        overflow: 'hidden',
        ...shadows.soft,
    },
    readyImage: {
        width: 100,
        height: 120,
    },
    readyContent: {
        flex: 1,
        padding: spacing.m,
        justifyContent: 'space-between',
    },
    readyTitle: {
        fontSize: 16,
        fontWeight: '700',
        color: colors.text.primary,
        marginBottom: spacing.xs,
    },
    readyItems: {
        marginBottom: spacing.s,
    },
    readyItem: {
        fontSize: 13,
        color: colors.text.secondary,
        marginBottom: 2,
    },
    readyMore: {
        fontSize: 12,
        color: colors.text.accent,
        fontWeight: '500',
    },
    readyFooter: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    difficultyBadge: {
        paddingHorizontal: spacing.s,
        paddingVertical: 4,
        borderRadius: 8,
    },
    difficultyText: {
        fontSize: 11,
        fontWeight: '600',
    },
    saveButton: {
        padding: 4,
    },
    // AI Tools
    aiToolsRow: {
        flexDirection: 'row',
        paddingHorizontal: spacing.l,
        gap: spacing.s,
    },
    aiTool: {
        flex: 1,
        backgroundColor: colors.surface,
        borderRadius: borderRadius.l,
        padding: spacing.m,
        alignItems: 'center',
        ...shadows.soft,
    },
    aiToolIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: spacing.s,
    },
    aiToolTitle: {
        fontSize: 12,
        fontWeight: '700',
        color: colors.text.primary,
        textAlign: 'center',
    },
});

export default ExploreScreen;

