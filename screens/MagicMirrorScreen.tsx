import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
    View,
    Text,
    Image,
    TouchableOpacity,
    StyleSheet,
    SafeAreaView,
    ActivityIndicator,
    Alert,
    Dimensions,
    StatusBar,
} from 'react-native';
import { BlurView } from 'expo-blur';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Animated, {
    FadeIn,
    FadeInDown,
    FadeInUp,
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
} from 'react-native-reanimated';

import SwipeableClothingCarousel from '../components/ui/SwipeableClothingCarousel';
import { ClosetlyTheme, ClosetlyStyles } from '../constants/ClosetlyTheme';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import { useTranslation } from 'react-i18next';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Types
interface ClothingItem {
    _id: string;
    id?: string;
    type?: string;
    itemType?: string;
    category?: string;
    color?: string;
    colorHex?: string;
    imageUrl?: string;
    image?: string;
    style?: string;
}

interface MatchedBottom {
    item: ClothingItem;
    matchScore: number;
}

/**
 * MagicMirrorScreen - Split-View Virtual Try-On Interface
 * 
 * A rapid-fire, gamified outfit experimentation interface following
 * the Closetly "Invisible UI" aesthetic.
 * 
 * Features:
 * - Vertical split screen (Tops / Bottoms)
 * - Independent horizontal swiping
 * - Central stationary user photo
 * - AI-powered bottom suggestions based on selected top
 * - Match percentage overlay
 * - Glassmorphism action buttons
 */
const MagicMirrorScreen: React.FC = () => {
    const { t } = useTranslation();
    const navigation = useNavigation();
    const { user } = useAuthStore(); // Use AuthStore

    // State
    const [userPhoto, setUserPhoto] = useState<string | null>(null);
    const [tops, setTops] = useState<ClothingItem[]>([]);
    const [bottoms, setBottoms] = useState<ClothingItem[]>([]);
    const [selectedTop, setSelectedTop] = useState<ClothingItem | null>(null);
    const [selectedBottom, setSelectedBottom] = useState<ClothingItem | null>(null);
    const [sortedBottoms, setSortedBottoms] = useState<ClothingItem[]>([]);
    const [matchScores, setMatchScores] = useState<Map<string, number>>(new Map());
    const [overallMatchScore, setOverallMatchScore] = useState<number>(0);

    const [loading, setLoading] = useState(true);
    const [tryingOn, setTryingOn] = useState(false);
    const [resultImage, setResultImage] = useState<string | null>(null);

    // Animations
    const matchScoreScale = useSharedValue(1);
    const tryOnButtonScale = useSharedValue(1);
    const animationTimeoutRef = React.useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        return () => {
            if (animationTimeoutRef.current) clearTimeout(animationTimeoutRef.current);
        };
    }, []);

    // Load wardrobe items
    const loadWardrobeItems = useCallback(async () => {
        try {
            if (!user) return;
            setLoading(true);

            // Fetch from Supabase
            const { data, error } = await supabase
                .from('clothing_items')
                .select('*')
                .eq('user_id', user.id);

            if (error) throw error;

            let items: ClothingItem[] = [];
            if (data) {
                items = data.map(item => ({
                    _id: item.id,
                    id: item.id,
                    imageUrl: item.image_url,
                    image: item.image_url,
                    type: item.type,
                    category: item.category,
                    itemType: item.category, // Map category to itemType for logic
                    color: item.color,
                    style: item.style
                }));
            }

            // Categorize items
            const topCategories = ['shirt', 'top', 'blouse', 'sweater', 'jacket', 'coat', 'hoodie', 'tee', 't-shirt', 'upper'];
            const bottomCategories = ['pants', 'jeans', 'trousers', 'shorts', 'skirt', 'bottom', 'lower'];

            const topsFiltered = items.filter((item) => {
                const category = (item.category || item.type || item.itemType || '').toLowerCase();
                return topCategories.some((cat) => category.includes(cat));
            });

            const bottomsFiltered = items.filter((item) => {
                const category = (item.category || item.type || item.itemType || '').toLowerCase();
                return bottomCategories.some((cat) => category.includes(cat));
            });

            setTops(topsFiltered.length > 0 ? topsFiltered : items.slice(0, Math.ceil(items.length / 2)));
            setBottoms(bottomsFiltered.length > 0 ? bottomsFiltered : items.slice(Math.ceil(items.length / 2)));
            setSortedBottoms(bottomsFiltered.length > 0 ? bottomsFiltered : items.slice(Math.ceil(items.length / 2)));

            // Load saved user photo (Can stick with AsyncStorage for local prefs, or migrate to user metadata)
            const savedPhoto = await AsyncStorage.getItem('userTryOnPhoto');
            if (savedPhoto) {
                setUserPhoto(savedPhoto);
            }

        } catch (error) {
            console.error('Failed to load wardrobe:', error);
            Alert.alert(t('common.error'), t('magicMirror.failedLoadWardrobe'));
        } finally {
            setLoading(false);
        }
    }, [user]);

    useFocusEffect(
        useCallback(() => {
            loadWardrobeItems();
        }, [loadWardrobeItems])
    );

    // Get AI match scores when top changes
    const getMatchingBottoms = useCallback(async (top: ClothingItem) => {
        if (bottoms.length === 0) return;

        try {
            // Generate match scores based on color harmony and style compatibility
            // This is a simplified local implementation - can be replaced with API call
            const scores = new Map<string, number>();

            const topColor = (top.color || '').toLowerCase();
            const topStyle = (top.style || '').toLowerCase();

            // Color harmony rules (simplified)
            const colorHarmony: Record<string, string[]> = {
                '#0A1931': ['white', 'gray', 'blue', 'red', 'beige', 'tan', 'khaki'],
                'white': ['#0A1931', 'blue', 'navy', 'gray', 'beige', 'brown'],
                'blue': ['white', 'beige', 'khaki', 'gray', 'tan', 'brown'],
                'navy': ['white', 'beige', 'khaki', 'tan', 'gray'],
                'gray': ['#0A1931', 'white', 'blue', 'navy', 'burgundy'],
                'beige': ['navy', 'blue', 'brown', 'white', '#0A1931'],
                'brown': ['beige', 'white', 'khaki', 'navy'],
            };

            const harmonicColors = colorHarmony[topColor] || [];

            const scoredBottoms: MatchedBottom[] = bottoms.map((bottom) => {
                const bottomColor = (bottom.color || '').toLowerCase();
                const bottomStyle = (bottom.style || '').toLowerCase();

                let score = 70; // Base score

                // Color harmony bonus
                if (harmonicColors.includes(bottomColor)) {
                    score += 20;
                } else if (bottomColor === topColor) {
                    score += 10; // Monochrome bonus
                }

                // Style match bonus
                if (topStyle && bottomStyle && topStyle === bottomStyle) {
                    score += 10;
                }

                // Add some randomness for variety
                score = Math.min(99, Math.max(60, score + Math.floor(Math.random() * 10) - 5));

                scores.set(bottom._id || bottom.id || '', score);
                return { item: bottom, matchScore: score };
            });

            // Sort by match score descending
            scoredBottoms.sort((a, b) => b.matchScore - a.matchScore);

            setMatchScores(scores);
            setSortedBottoms(scoredBottoms.map((sb) => sb.item));

            // Animate match score badge
            matchScoreScale.value = withSpring(1.2, { damping: 10 });
            if (animationTimeoutRef.current) clearTimeout(animationTimeoutRef.current);
            animationTimeoutRef.current = setTimeout(() => {
                matchScoreScale.value = withSpring(1, { damping: 15 });
            }, 200);

        } catch (error) {
            console.error('Failed to get matching bottoms:', error);
        }
    }, [bottoms, matchScoreScale]);

    // Handle top selection
    const handleTopChange = useCallback((item: ClothingItem, index: number) => {
        if (selectedTop?._id !== item._id) {
            setSelectedTop(item);
            getMatchingBottoms(item);
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        }
    }, [selectedTop, getMatchingBottoms]);

    // Handle bottom selection
    const handleBottomChange = useCallback((item: ClothingItem, index: number) => {
        if (selectedBottom?._id !== item._id) {
            setSelectedBottom(item);

            // Calculate overall match score
            const score = matchScores.get(item._id || item.id || '') || 75;
            setOverallMatchScore(score);

            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        }
    }, [selectedBottom, matchScores]);

    // Pick user photo
    const pickUserPhoto = useCallback(async () => {
        const permissionResult = await ImagePicker.requestMediaLibraryPermissionsAsync();

        if (!permissionResult.granted) {
            Alert.alert(t('magicMirror.permissionRequired'), t('magicMirror.allowAccessPhotos'));
            return;
        }

        const result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ['images'],
            allowsEditing: true,
            aspect: [3, 4],
            quality: 0.7,
            base64: true,
        });

        if (!result.canceled && result.assets?.[0]?.base64) {
            const base64Image = `data:image/jpeg;base64,${result.assets[0].base64}`;
            setUserPhoto(base64Image);
            await AsyncStorage.setItem('userTryOnPhoto', base64Image);
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        }
    }, []);

    // Perform virtual try-on
    const handleTryOn = useCallback(async () => {
        if (!userPhoto) {
            Alert.alert(t('magicMirror.photoRequired'), t('magicMirror.addPhotoFirst'));
            return;
        }

        if (!selectedTop) {
            Alert.alert(t('magicMirror.selectOutfit'), t('magicMirror.swipeSelectTop'));
            return;
        }

        if (!user) {
            Alert.alert(t('magicMirror.loginRequired'), t('magicMirror.loginUseMagicMirror'));
            return;
        }

        setTryingOn(true);
        setResultImage(null);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

        try {
            const garmentImage = selectedTop.imageUrl || selectedTop.image;

            if (!garmentImage) {
                throw new Error(t('common.noGarmentImageAvailable'));
            }

            const { data, error } = await supabase.functions.invoke('try-on', {
                body: {
                    person_image: userPhoto,
                    garment_image: garmentImage,
                    garment_type: 'upper_body',
                }
            });

            if (error) throw error;

            if (data.success && data.resultImage) {
                setResultImage(data.resultImage);
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                if (data.methodUsed === 'mock') {
                    Alert.alert(t('magicMirror.demoMode'), t('magicMirror.aiServiceNotConfigured'));
                }
            } else {
                throw new Error(data.error || 'Try-on failed');
            }
        } catch (error: any) {
            console.error('Try-on error:', error);
            Alert.alert(t('magicMirror.tryOnFailed'), t('magicMirror.tryAgainLater') + ' ' + (error.message || ''));
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        } finally {
            setTryingOn(false);
        }
    }, [userPhoto, selectedTop, user]);

    // Animated styles
    const matchScoreAnimatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: matchScoreScale.value }],
    }));

    // Loading state
    if (loading) {
        return (
            <SafeAreaView style={styles.screen}>
                <StatusBar barStyle="dark-content" />
                <View style={styles.loadingContainer}>
                    <ActivityIndicator size="large" color={ClosetlyTheme.colors.text} />
                    <Text style={styles.loadingText}>{t('magicMirror.loading')}</Text>
                </View>
            </SafeAreaView>
        );
    }

    return (
        <Animated.View
            style={[styles.screen]}
            // @ts-ignore
            sharedTransitionTag="magicMirrorContainer"
        >
            <SafeAreaView style={{ flex: 1 }}>
                <StatusBar barStyle="dark-content" />

                {/* Header */}
                <Animated.View entering={FadeInDown.duration(400)} style={styles.header}>
                    <TouchableOpacity
                        onPress={() => navigation.goBack()}
                        style={styles.backButton}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="chevron-back" size={28} color={ClosetlyTheme.colors.text} />
                    </TouchableOpacity>

                    <Text style={styles.headerTitle}>{t('magicMirror.title')}</Text>

                    {/* Match Score Badge */}
                    {overallMatchScore > 0 && (
                        <Animated.View style={[styles.matchBadge, matchScoreAnimatedStyle]}>
                            <Text style={styles.matchBadgeText}>{overallMatchScore}%</Text>
                        </Animated.View>
                    )}

                    {!overallMatchScore && <View style={{ width: 50 }} />}
                </Animated.View>

                {/* Main Content */}
                <View style={styles.content}>

                    {/* TOPS Section */}
                    <Animated.View entering={FadeInUp.delay(100).duration(400)} style={styles.section}>
                        <Text style={styles.sectionLabel}>{t('magicMirror.tops')}</Text>
                        <SwipeableClothingCarousel
                            items={tops}
                            onItemChange={handleTopChange}
                            category="tops"
                            selectedId={selectedTop?._id}
                            containerHeight={200}
                        />
                    </Animated.View>

                    {/* Center Model View */}
                    <Animated.View entering={FadeIn.delay(200).duration(500)} style={styles.modelSection}>
                        <TouchableOpacity
                            style={styles.modelContainer}
                            onPress={pickUserPhoto}
                            activeOpacity={0.8}
                        >
                            {resultImage ? (
                                <Image source={{ uri: resultImage }} style={styles.modelImage} />
                            ) : userPhoto ? (
                                <Image source={{ uri: userPhoto }} style={styles.modelImage} />
                            ) : (
                                <View style={styles.modelPlaceholder}>
                                    <View style={styles.modelIconCircle}>
                                        <Ionicons name="person" size={40} color={ClosetlyTheme.colors.textMuted} />
                                    </View>
                                    <Text style={styles.modelPlaceholderText}>{t('magicMirror.tapToAddPhoto')}</Text>
                                </View>
                            )}

                            {/* Try-On Loading Overlay */}
                            {tryingOn && (
                                <BlurView intensity={80} style={styles.tryOnOverlay}>
                                    <ActivityIndicator size="large" color={ClosetlyTheme.colors.text} />
                                    <Text style={styles.tryOnText}>{t('magicMirror.creatingMagic')}</Text>
                                </BlurView>
                            )}
                        </TouchableOpacity>
                    </Animated.View>

                    {/* BOTTOMS Section */}
                    <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.section}>
                        <Text style={styles.sectionLabel}>{t('magicMirror.bottoms')}</Text>
                        <SwipeableClothingCarousel
                            items={sortedBottoms}
                            onItemChange={handleBottomChange}
                            category="bottoms"
                            matchScores={matchScores}
                            selectedId={selectedBottom?._id}
                            containerHeight={200}
                        />
                    </Animated.View>
                </View>

                {/* Bottom Action Bar with Glassmorphism */}
                <Animated.View entering={FadeInUp.delay(400).duration(400)} style={styles.actionBar}>
                    <BlurView intensity={90} style={styles.actionBarBlur}>
                        <TouchableOpacity
                            style={[styles.tryOnButton, (!userPhoto || !selectedTop) && styles.tryOnButtonDisabled]}
                            onPress={handleTryOn}
                            disabled={tryingOn || !userPhoto || !selectedTop}
                        >
                            <Ionicons
                                name={tryingOn ? 'hourglass-outline' : 'sparkles'}
                                size={20}
                                color={ClosetlyTheme.colors.background}
                            />
                            <Text style={styles.tryOnButtonText}>
                                {tryingOn ? t('magicMirror.generating') : t('magicMirror.tryOn')}
                            </Text>
                        </TouchableOpacity>

                        {resultImage && (
                            <TouchableOpacity style={styles.saveButton}>
                                <Ionicons name="heart-outline" size={20} color={ClosetlyTheme.colors.text} />
                                <Text style={styles.saveButtonText}>{t('common.save')}</Text>
                            </TouchableOpacity>
                        )}
                    </BlurView>
                </Animated.View>
            </SafeAreaView>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    screen: {
        flex: 1,
        backgroundColor: ClosetlyTheme.colors.background,
    },
    loadingContainer: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        gap: 16,
    },
    loadingText: {
        ...ClosetlyTheme.typography.body,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
    },
    backButton: {
        width: 44,
        height: 44,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerTitle: {
        ...ClosetlyTheme.typography.header,
        fontSize: 28,
    },
    matchBadge: {
        backgroundColor: ClosetlyTheme.colors.card,
        borderRadius: ClosetlyTheme.borderRadius.sm,
        paddingVertical: 6,
        paddingHorizontal: 12,
        ...ClosetlyTheme.shadows.cardSmall,
    },
    matchBadgeText: {
        ...ClosetlyTheme.typography.matchScore,
        fontSize: 16,
    },

    // Content
    content: {
        flex: 1,
        justifyContent: 'space-between',
    },

    // Sections
    section: {
        paddingVertical: 8,
    },
    sectionLabel: {
        ...ClosetlyTheme.typography.label,
        marginLeft: 24,
        marginBottom: 8,
    },

    // Model View
    modelSection: {
        alignItems: 'center',
        paddingVertical: 16,
    },
    modelContainer: {
        width: 180,
        height: 240,
        borderRadius: ClosetlyTheme.borderRadius.xl,
        overflow: 'hidden',
        backgroundColor: ClosetlyTheme.colors.card,
        ...ClosetlyTheme.shadows.floating,
    },
    modelImage: {
        width: '100%',
        height: '100%',
        resizeMode: 'cover',
    },
    modelPlaceholder: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
    },
    modelIconCircle: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: ClosetlyTheme.colors.background,
        alignItems: 'center',
        justifyContent: 'center',
    },
    modelPlaceholderText: {
        ...ClosetlyTheme.typography.caption,
        textAlign: 'center',
        paddingHorizontal: 20,
    },
    tryOnOverlay: {
        ...StyleSheet.absoluteFillObject,
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
    },
    tryOnText: {
        ...ClosetlyTheme.typography.body,
        fontWeight: '600',
    },

    // Action Bar
    actionBar: {
        paddingHorizontal: 20,
        paddingBottom: 24,
    },
    actionBarBlur: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        paddingVertical: 12,
        paddingHorizontal: 20,
        borderRadius: ClosetlyTheme.borderRadius.xl,
        overflow: 'hidden',
        backgroundColor: ClosetlyTheme.colors.glassBg,
    },
    tryOnButton: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        backgroundColor: ClosetlyTheme.colors.text,
        paddingVertical: 16,
        borderRadius: ClosetlyTheme.borderRadius.button,
        ...ClosetlyTheme.shadows.button,
    },
    tryOnButtonDisabled: {
        backgroundColor: ClosetlyTheme.colors.textMuted,
    },
    tryOnButtonText: {
        color: ClosetlyTheme.colors.background,
        fontSize: 16,
        fontWeight: '600',
    },
    saveButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 6,
        backgroundColor: ClosetlyTheme.colors.card,
        paddingVertical: 16,
        paddingHorizontal: 24,
        borderRadius: ClosetlyTheme.borderRadius.button,
        ...ClosetlyTheme.shadows.cardSmall,
    },
    saveButtonText: {
        color: ClosetlyTheme.colors.text,
        fontSize: 16,
        fontWeight: '600',
    },
});

export default MagicMirrorScreen;
