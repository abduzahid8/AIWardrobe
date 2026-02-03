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
import axios from 'axios';
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
// @ts-ignore
import { API_URL } from '../api/config';

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
    const navigation = useNavigation();

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

    // Load wardrobe items
    const loadWardrobeItems = useCallback(async () => {
        try {
            setLoading(true);
            const token = await AsyncStorage.getItem('userToken');

            let items: ClothingItem[] = [];

            if (token) {
                const response = await axios.get(`${API_URL}/clothing-items`, {
                    headers: { Authorization: `Bearer ${token}` },
                });
                items = Array.isArray(response.data) ? response.data : response.data.items || [];
            } else {
                const localItems = await AsyncStorage.getItem('wardrobeItems');
                if (localItems) {
                    items = JSON.parse(localItems);
                }
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

            // Load saved user photo
            const savedPhoto = await AsyncStorage.getItem('userTryOnPhoto');
            if (savedPhoto) {
                setUserPhoto(savedPhoto);
            }

        } catch (error) {
            console.error('Failed to load wardrobe:', error);
            Alert.alert('Error', 'Failed to load your wardrobe. Please try again.');
        } finally {
            setLoading(false);
        }
    }, []);

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
                'black': ['white', 'gray', 'blue', 'red', 'beige', 'tan', 'khaki'],
                'white': ['black', 'blue', 'navy', 'gray', 'beige', 'brown'],
                'blue': ['white', 'beige', 'khaki', 'gray', 'tan', 'brown'],
                'navy': ['white', 'beige', 'khaki', 'tan', 'gray'],
                'gray': ['black', 'white', 'blue', 'navy', 'burgundy'],
                'beige': ['navy', 'blue', 'brown', 'white', 'black'],
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
            setTimeout(() => {
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
            Alert.alert('Permission Required', 'Please allow access to your photos.');
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
            Alert.alert('Photo Required', 'Please add your photo first.');
            return;
        }

        if (!selectedTop) {
            Alert.alert('Select Outfit', 'Please swipe to select a top.');
            return;
        }

        setTryingOn(true);
        setResultImage(null);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

        try {
            const garmentImage = selectedTop.imageUrl || selectedTop.image;

            if (!garmentImage) {
                throw new Error('No garment image available');
            }

            const response = await fetch(`${API_URL}/tryon`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    person_image: userPhoto,
                    garment_image: garmentImage,
                    garment_type: 'upper_body',
                }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            if (data.success && data.resultImage) {
                setResultImage(data.resultImage);
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            } else {
                throw new Error(data.error || 'Try-on failed');
            }
        } catch (error) {
            console.error('Try-on error:', error);
            Alert.alert('Try-On Failed', 'Please try again later.');
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        } finally {
            setTryingOn(false);
        }
    }, [userPhoto, selectedTop]);

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
                    <Text style={styles.loadingText}>Loading your wardrobe...</Text>
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

                    <Text style={styles.headerTitle}>Magic Mirror</Text>

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
                        <Text style={styles.sectionLabel}>TOPS</Text>
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
                                    <Text style={styles.modelPlaceholderText}>Tap to add your photo</Text>
                                </View>
                            )}

                            {/* Try-On Loading Overlay */}
                            {tryingOn && (
                                <BlurView intensity={80} style={styles.tryOnOverlay}>
                                    <ActivityIndicator size="large" color={ClosetlyTheme.colors.text} />
                                    <Text style={styles.tryOnText}>Creating magic...</Text>
                                </BlurView>
                            )}
                        </TouchableOpacity>
                    </Animated.View>

                    {/* BOTTOMS Section */}
                    <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.section}>
                        <Text style={styles.sectionLabel}>BOTTOMS</Text>
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
                                {tryingOn ? 'Generating...' : 'Try On'}
                            </Text>
                        </TouchableOpacity>

                        {resultImage && (
                            <TouchableOpacity style={styles.saveButton}>
                                <Ionicons name="heart-outline" size={20} color={ClosetlyTheme.colors.text} />
                                <Text style={styles.saveButtonText}>Save</Text>
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
