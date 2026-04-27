/**
 * OutfitInspoScreen — Upload a photo of an outfit or clothing item,
 * AI analyzes it, then recommends similar/better items from your
 * wardrobe and the shop catalog.
 */

import React, { useState, useCallback, useRef } from 'react';
import {
    View,
    Text,
    ScrollView,
    Image,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ActivityIndicator,
    Share,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, FadeInUp, FadeInDown, ZoomIn } from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { useTranslation } from 'react-i18next';

import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import { TahoeIconButton } from '../components/TahoeButton';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const { colors, spacing, typography, radius } = LiquidGlass2026Theme;

// ── Types ────────────────────────────────────────────────────────────────────

interface DetectedItem {
    category: string;
    section: string;
    specificType: string;
    primaryColor: string;
    colorHex: string;
    style: string;
    material: string | null;
    pattern: string;
    fit: string;
    description: string;
}

interface WardrobeMatch {
    id: string;
    category: string;
    color?: string;
    primary_color?: string;
    style?: string;
    image_url?: string;
    imageUrl?: string;
    description?: string;
    sub_category?: string;
    matchScore: number;
}

interface ShopMatch {
    id: string;
    name: string;
    brand: string;
    price: number;
    currency: string;
    imageUrl?: string;
    image_url?: string;
    garment_type?: string;
    description?: string;
    isShopItem: boolean;
    matchScore: number;
}

interface RecommendationGroup {
    detectedItem: DetectedItem;
    similarFromWardrobe: WardrobeMatch[];
    similarFromShop: ShopMatch[];
}

type AnalysisMode = 'outfit' | 'single';

// ── Component ────────────────────────────────────────────────────────────────

const OutfitInspoScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const { user } = useAuthStore();

    const [imageUri, setImageUri] = useState<string | null>(null);
    const [imageBase64, setImageBase64] = useState<string | null>(null);
    const [mode, setMode] = useState<AnalysisMode>('outfit');
    const [analyzing, setAnalyzing] = useState(false);
    const [recommendations, setRecommendations] = useState<RecommendationGroup[]>([]);
    const [outfitDescription, setOutfitDescription] = useState<string>('');
    const [error, setError] = useState<string | null>(null);
    const [expandedItem, setExpandedItem] = useState<number | null>(0);
    const scrollViewRef = useRef<ScrollView>(null);

    // ── Image Picking ────────────────────────────────────────────────────────

    const pickImage = useCallback(async (fromCamera: boolean) => {
        try {
            if (fromCamera) {
                const { status } = await ImagePicker.requestCameraPermissionsAsync();
                if (status !== 'granted') {
                    Alert.alert(t('outfitInspo.cameraPermission'), t('outfitInspo.cameraPermissionMsg'));
                    return;
                }
            } else {
                const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
                if (status !== 'granted') {
                    Alert.alert(t('outfitInspo.photoPermission'), t('outfitInspo.photoPermissionMsg'));
                    return;
                }
            }

            const result = fromCamera
                ? await ImagePicker.launchCameraAsync({
                    base64: true,
                    quality: 0.8,
                    allowsEditing: true,
                })
                : await ImagePicker.launchImageLibraryAsync({
                    base64: true,
                    quality: 0.8,
                    allowsEditing: true,
                    mediaTypes: ImagePicker.MediaTypeOptions.Images,
                });

            if (!result.canceled && result.assets?.[0]) {
                const asset = result.assets[0];
                setImageUri(asset.uri);
                setImageBase64(asset.base64 || null);
                setRecommendations([]);
                setOutfitDescription('');
                setError(null);
                setExpandedItem(0);
            }
        } catch (err) {
            console.error('Image pick error:', err);
            setError(t('outfitInspo.imagePickError'));
        }
    }, [t]);

    // ── AI Analysis ──────────────────────────────────────────────────────────

    const analyzeImage = useCallback(async () => {
        if (!imageBase64) return;

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setAnalyzing(true);
        setError(null);

        try {
            const { data, error: fnError } = await supabase.functions.invoke('analyze-outfit', {
                body: {
                    image: `data:image/jpeg;base64,${imageBase64}`,
                    mode,
                    userId: user?.id,
                },
            });

            if (fnError) {
                throw new Error(fnError.message || t('common.edgeFunctionError'));
            }

            if (!data?.success) {
                throw new Error(data?.error || t('outfitInspo.analysisFailed'));
            }

            setRecommendations(data.recommendations || []);
            setOutfitDescription(data.outfitDescription || '');

            // Scroll to results after a short delay
            setTimeout(() => {
                scrollViewRef.current?.scrollTo({ y: 400, animated: true });
            }, 300);
        } catch (err: any) {
            console.error('Analysis error:', err);
            setError(err.message || t('outfitInspo.analysisFailed'));
        } finally {
            setAnalyzing(false);
        }
    }, [imageBase64, mode, user?.id, t]);

    // ── Share ────────────────────────────────────────────────────────────────

    const handleShare = useCallback(async () => {
        try {
            await Share.share({
                message: outfitDescription || t('outfitInspo.shareMessage'),
            });
        } catch {}
    }, [outfitDescription, t]);

    // ── Navigate to item ─────────────────────────────────────────────────────

    const handleItemPress = useCallback((item: any, isShopItem: boolean) => {
        if (isShopItem) {
            // Navigate to shop item detail or external link
            (navigation as any).navigate('ClothingDetail', {
                itemId: item.id,
                fullItem: item,
            });
        } else {
            (navigation as any).navigate('ClothingDetail', {
                itemId: item.id,
                fullItem: item,
            });
        }
    }, [navigation]);

    // ── Render ───────────────────────────────────────────────────────────────

    const renderImageUpload = () => (
        <Animated.View entering={FadeIn.duration(400)} style={styles.uploadSection}>
            <Text style={styles.sectionTitle}>{t('outfitInspo.uploadTitle')}</Text>
            <Text style={styles.sectionSubtitle}>
                {mode === 'outfit'
                    ? t('outfitInspo.uploadSubtitleOutfit')
                    : t('outfitInspo.uploadSubtitleSingle')}
            </Text>

            {imageUri ? (
                <Animated.View entering={ZoomIn.duration(300)} style={styles.imagePreviewContainer}>
                    <Image source={{ uri: imageUri }} style={styles.imagePreview} resizeMode="cover" />
                    <View style={styles.imageOverlay}>
                        <TouchableOpacity
                            style={styles.changeImageBtn}
                            onPress={() => pickImage(false)}
                        >
                            <Ionicons name="camera-outline" size={18} color="#FFF" />
                            <Text style={styles.changeImageText}>{t('outfitInspo.changePhoto')}</Text>
                        </TouchableOpacity>
                    </View>
                </Animated.View>
            ) : (
                <View style={styles.uploadButtons}>
                    <TouchableOpacity
                        style={styles.uploadButton}
                        onPress={() => pickImage(true)}
                        activeOpacity={0.7}
                    >
                        <LinearGradient
                            colors={['#0A1931', '#16213e']}
                            style={styles.uploadButtonGradient}
                        >
                            <Ionicons name="camera-outline" size={24} color="#FFF" />
                            <Text style={styles.uploadButtonText}>{t('outfitInspo.takePhoto')}</Text>
                        </LinearGradient>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={styles.uploadButton}
                        onPress={() => pickImage(false)}
                        activeOpacity={0.7}
                    >
                        <LinearGradient
                            colors={['#1a1a2e', '#0f4c75']}
                            style={styles.uploadButtonGradient}
                        >
                            <Ionicons name="images-outline" size={24} color="#FFF" />
                            <Text style={styles.uploadButtonText}>{t('outfitInspo.chooseGallery')}</Text>
                        </LinearGradient>
                    </TouchableOpacity>
                </View>
            )}

            {/* Mode Toggle */}
            <View style={styles.modeToggle}>
                <TouchableOpacity
                    style={[styles.modeOption, mode === 'outfit' && styles.modeOptionActive]}
                    onPress={() => setMode('outfit')}
                    activeOpacity={0.7}
                >
                    <Ionicons
                        name="shirt-outline"
                        size={18}
                        color={mode === 'outfit' ? '#FFF' : colors.text.secondary}
                    />
                    <Text style={[styles.modeText, mode === 'outfit' && styles.modeTextActive]}>
                        {t('outfitInspo.fullOutfit')}
                    </Text>
                </TouchableOpacity>
                <TouchableOpacity
                    style={[styles.modeOption, mode === 'single' && styles.modeOptionActive]}
                    onPress={() => setMode('single')}
                    activeOpacity={0.7}
                >
                    <Ionicons
                        name="search-outline"
                        size={18}
                        color={mode === 'single' ? '#FFF' : colors.text.secondary}
                    />
                    <Text style={[styles.modeText, mode === 'single' && styles.modeTextActive]}>
                        {t('outfitInspo.findSimilar')}
                    </Text>
                </TouchableOpacity>
            </View>

            {/* Analyze Button */}
            {imageUri && (
                <Animated.View entering={FadeInUp.duration(300)}>
                    <TouchableOpacity
                        style={[styles.analyzeButton, analyzing && styles.analyzeButtonDisabled]}
                        onPress={analyzeImage}
                        disabled={analyzing}
                        activeOpacity={0.8}
                    >
                        <LinearGradient
                            colors={['#0A1931', '#16213e']}
                            style={styles.analyzeButtonGradient}
                        >
                            {analyzing ? (
                                <ActivityIndicator size="small" color="#FFF" />
                            ) : (
                                <>
                                    <Ionicons name="sparkles" size={20} color="#FFF" />
                                    <Text style={styles.analyzeButtonText}>
                                        {mode === 'outfit'
                                            ? t('outfitInspo.analyzeOutfit')
                                            : t('outfitInspo.findSimilarItems')}
                                    </Text>
                                </>
                            )}
                        </LinearGradient>
                    </TouchableOpacity>
                </Animated.View>
            )}
        </Animated.View>
    );

    const renderDetectedItem = (item: DetectedItem, index: number) => {
        const isExpanded = expandedItem === index;
        const rec = recommendations[index];

        return (
            <Animated.View
                key={index}
                entering={FadeInUp.delay(index * 80).duration(400)}
                style={styles.detectedItemCard}
            >
                <TouchableOpacity
                    style={styles.detectedItemHeader}
                    onPress={() => setExpandedItem(isExpanded ? null : index)}
                    activeOpacity={0.7}
                >
                    <View style={styles.detectedItemInfo}>
                        <View style={[styles.colorDot, { backgroundColor: item.colorHex || '#808080' }]} />
                        <View style={styles.detectedItemText}>
                            <Text style={styles.detectedItemName}>{item.specificType}</Text>
                            <Text style={styles.detectedItemMeta}>
                                {item.primaryColor} · {item.style} · {item.pattern}
                            </Text>
                        </View>
                    </View>
                    <Ionicons
                        name={isExpanded ? 'chevron-up' : 'chevron-down'}
                        size={20}
                        color={colors.text.tertiary}
                    />
                </TouchableOpacity>

                {isExpanded && rec && (
                    <Animated.View entering={FadeInDown.duration(250)} style={styles.expandedContent}>
                        {item.description ? (
                            <Text style={styles.itemDescription}>{item.description}</Text>
                        ) : null}

                        {/* Wardrobe Matches */}
                        {rec.similarFromWardrobe.length > 0 && (
                            <View style={styles.matchSection}>
                                <Text style={styles.matchSectionTitle}>
                                    <Ionicons name="shirt-outline" size={14} color={colors.accent.primary} />
                                    {'  '}{t('outfitInspo.fromYourWardrobe')}
                                </Text>
                                <ScrollView
                                    horizontal
                                    showsHorizontalScrollIndicator={false}
                                    contentContainerStyle={styles.matchScroll}
                                >
                                    {rec.similarFromWardrobe.map((match, mIdx) => (
                                        <TouchableOpacity
                                            key={match.id}
                                            style={styles.matchCard}
                                            onPress={() => handleItemPress(match, false)}
                                            activeOpacity={0.7}
                                        >
                                            {match.imageUrl || match.image_url ? (
                                                <Image
                                                    source={{ uri: match.imageUrl || match.image_url }}
                                                    style={styles.matchImage}
                                                    resizeMode="cover"
                                                />
                                            ) : (
                                                <View style={styles.matchImagePlaceholder}>
                                                    <Ionicons name="shirt-outline" size={20} color={colors.text.tertiary} />
                                                </View>
                                            )}
                                            <Text style={styles.matchName} numberOfLines={1}>
                                                {match.sub_category || match.category || t('outfitInspo.item')}
                                            </Text>
                                            <View style={styles.matchScoreBadge}>
                                                <Text style={styles.matchScoreText}>
                                                    {Math.round(match.matchScore)}%
                                                </Text>
                                            </View>
                                        </TouchableOpacity>
                                    ))}
                                </ScrollView>
                            </View>
                        )}

                        {/* Shop Matches */}
                        {rec.similarFromShop.length > 0 && (
                            <View style={styles.matchSection}>
                                <Text style={styles.matchSectionTitle}>
                                    <Ionicons name="bag-outline" size={14} color={colors.accent.primary} />
                                    {'  '}{t('outfitInspo.shopRecommendations')}
                                </Text>
                                <ScrollView
                                    horizontal
                                    showsHorizontalScrollIndicator={false}
                                    contentContainerStyle={styles.matchScroll}
                                >
                                    {rec.similarFromShop.map((match, mIdx) => (
                                        <TouchableOpacity
                                            key={match.id}
                                            style={styles.matchCard}
                                            onPress={() => handleItemPress(match, true)}
                                            activeOpacity={0.7}
                                        >
                                            {match.imageUrl || match.image_url ? (
                                                <Image
                                                    source={{ uri: match.imageUrl || match.image_url }}
                                                    style={styles.matchImage}
                                                    resizeMode="cover"
                                                />
                                            ) : (
                                                <View style={styles.matchImagePlaceholder}>
                                                    <Ionicons name="bag-outline" size={20} color={colors.text.tertiary} />
                                                </View>
                                            )}
                                            <Text style={styles.matchName} numberOfLines={2}>
                                                {match.name}
                                            </Text>
                                            {match.price > 0 && (
                                                <Text style={styles.matchPrice}>
                                                    {match.currency || '$'}{match.price}
                                                </Text>
                                            )}
                                            <View style={styles.matchScoreBadge}>
                                                <Text style={styles.matchScoreText}>
                                                    {Math.round(match.matchScore)}%
                                                </Text>
                                            </View>
                                        </TouchableOpacity>
                                    ))}
                                </ScrollView>
                            </View>
                        )}

                        {/* No matches */}
                        {rec.similarFromWardrobe.length === 0 && rec.similarFromShop.length === 0 && (
                            <View style={styles.noMatches}>
                                <Ionicons name="search-outline" size={24} color={colors.text.tertiary} />
                                <Text style={styles.noMatchesText}>{t('outfitInspo.noMatchesFound')}</Text>
                            </View>
                        )}
                    </Animated.View>
                )}
            </Animated.View>
        );
    };

    const renderResults = () => {
        if (error) {
            return (
                <Animated.View entering={FadeIn.duration(300)} style={styles.errorContainer}>
                    <Ionicons name="alert-circle-outline" size={40} color={colors.accent.primary} />
                    <Text style={styles.errorText}>{error}</Text>
                    <TouchableOpacity style={styles.retryButton} onPress={analyzeImage}>
                        <Text style={styles.retryButtonText}>{t('outfitInspo.tryAgain')}</Text>
                    </TouchableOpacity>
                </Animated.View>
            );
        }

        if (recommendations.length === 0) return null;

        return (
            <Animated.View entering={FadeIn.duration(400)} style={styles.resultsSection}>
                {/* Outfit Summary */}
                {outfitDescription ? (
                    <View style={styles.outfitSummary}>
                        <View style={styles.summaryIconContainer}>
                            <Ionicons name="sparkles" size={20} color={colors.accent.primary} />
                        </View>
                        <Text style={styles.outfitSummaryText}>{outfitDescription}</Text>
                        <TouchableOpacity style={styles.shareButton} onPress={handleShare}>
                            <Ionicons name="share-outline" size={18} color={colors.text.secondary} />
                        </TouchableOpacity>
                    </View>
                ) : null}

                <Text style={styles.sectionTitle}>{t('outfitInspo.detectedItems')}</Text>

                {recommendations.map((rec, index) =>
                    renderDetectedItem(rec.detectedItem, index)
                )}
            </Animated.View>
        );
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea} edges={['top']}>
                {/* Header */}
                <Animated.View entering={FadeIn} style={styles.header}>
                    <TahoeIconButton
                        icon="arrow-back"
                        onPress={() => navigation.goBack()}
                        color={colors.text.primary}
                    />
                    <View style={styles.headerCenter}>
                        <View style={styles.headerBadge}>
                            <Ionicons name="sparkles" size={12} color={colors.accent.primary} />
                        </View>
                        <Text style={styles.headerTitle}>{t('outfitInspo.title')}</Text>
                    </View>
                    <View style={styles.headerPlaceholder} />
                </Animated.View>

                <ScrollView
                    ref={scrollViewRef}
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                    keyboardShouldPersistTaps="handled"
                >
                    {renderImageUpload()}

                    {analyzing && (
                        <Animated.View entering={FadeIn.duration(300)} style={styles.analyzingContainer}>
                            <ActivityIndicator size="large" color={colors.accent.primary} />
                            <Text style={styles.analyzingText}>
                                {mode === 'outfit'
                                    ? t('outfitInspo.analyzingOutfit')
                                    : t('outfitInspo.findingSimilar')}
                            </Text>
                            <Text style={styles.analyzingSubtext}>{t('outfitInspo.takesMoment')}</Text>
                        </Animated.View>
                    )}

                    {!analyzing && renderResults()}

                    <View style={{ height: 120 }} />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

// ── Styles ──────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    safeArea: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.screenPadding,
        paddingVertical: spacing.sm,
    },
    headerCenter: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.xs,
    },
    headerBadge: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: colors.glass.tinted,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerTitle: {
        ...typography.scale.titleMedium,
        color: colors.text.primary,
        fontWeight: '600',
    },
    headerPlaceholder: {
        width: 40,
    },

    // Scroll
    scrollContent: {
        paddingTop: spacing.md,
        paddingHorizontal: spacing.screenPadding,
    },

    // Section titles
    sectionTitle: {
        ...typography.scale.headlineSmall,
        color: colors.text.primary,
        fontWeight: '600',
        marginBottom: spacing.xs,
    },
    sectionSubtitle: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        marginBottom: spacing.lg,
        lineHeight: 20,
    },

    // Upload Section
    uploadSection: {
        marginBottom: spacing.xl,
    },

    // Image Preview
    imagePreviewContainer: {
        borderRadius: radius.xl,
        overflow: 'hidden',
        marginBottom: spacing.md,
        height: SCREEN_WIDTH * 0.75,
    },
    imagePreview: {
        width: '100%',
        height: '100%',
    },
    imageOverlay: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: spacing.md,
    },
    changeImageBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        alignSelf: 'flex-end',
        backgroundColor: 'rgba(0,0,0,0.5)',
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.sm,
        borderRadius: radius.pill,
        gap: spacing.xs,
    },
    changeImageText: {
        color: '#FFF',
        fontSize: 13,
        fontWeight: '600',
    },

    // Upload Buttons
    uploadButtons: {
        flexDirection: 'row',
        gap: spacing.md,
        marginBottom: spacing.lg,
    },
    uploadButton: {
        flex: 1,
        borderRadius: radius.lg,
        overflow: 'hidden',
    },
    uploadButtonGradient: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.xl,
        gap: spacing.sm,
    },
    uploadButtonText: {
        color: '#FFF',
        fontSize: 14,
        fontWeight: '600',
    },

    // Mode Toggle
    modeToggle: {
        flexDirection: 'row',
        backgroundColor: colors.background.secondary,
        borderRadius: radius.pill,
        padding: 4,
        marginBottom: spacing.lg,
    },
    modeOption: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.sm,
        borderRadius: radius.pill,
        gap: spacing.xs,
    },
    modeOptionActive: {
        backgroundColor: '#0A1931',
    },
    modeText: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        fontWeight: '600',
    },
    modeTextActive: {
        color: '#FFF',
    },

    // Analyze Button
    analyzeButton: {
        borderRadius: radius.pill,
        overflow: 'hidden',
        marginBottom: spacing.md,
    },
    analyzeButtonDisabled: {
        opacity: 0.6,
    },
    analyzeButtonGradient: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.md,
        gap: spacing.sm,
    },
    analyzeButtonText: {
        color: '#FFF',
        fontSize: 16,
        fontWeight: '600',
    },

    // Analyzing
    analyzingContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.xxl,
        gap: spacing.md,
    },
    analyzingText: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        fontWeight: '600',
    },
    analyzingSubtext: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
    },

    // Results
    resultsSection: {
        marginBottom: spacing.xl,
    },

    // Outfit Summary
    outfitSummary: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        padding: spacing.md,
        marginBottom: spacing.lg,
        gap: spacing.sm,
    },
    summaryIconContainer: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: colors.glass.tinted,
        alignItems: 'center',
        justifyContent: 'center',
    },
    outfitSummaryText: {
        ...typography.scale.bodySmall,
        color: colors.text.primary,
        flex: 1,
        lineHeight: 18,
    },
    shareButton: {
        padding: spacing.sm,
    },

    // Detected Item Card
    detectedItemCard: {
        backgroundColor: colors.background.secondary,
        borderRadius: radius.lg,
        marginBottom: spacing.md,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: colors.border.subtle,
    },
    detectedItemHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: spacing.md,
    },
    detectedItemInfo: {
        flexDirection: 'row',
        alignItems: 'center',
        flex: 1,
        gap: spacing.sm,
    },
    colorDot: {
        width: 28,
        height: 28,
        borderRadius: 14,
        borderWidth: 2,
        borderColor: '#FFF',
    },
    detectedItemText: {
        flex: 1,
    },
    detectedItemName: {
        ...typography.scale.bodyLarge,
        color: colors.text.primary,
        fontWeight: '600',
        textTransform: 'capitalize',
    },
    detectedItemMeta: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        textTransform: 'capitalize',
        marginTop: 2,
    },

    // Expanded Content
    expandedContent: {
        paddingHorizontal: spacing.md,
        paddingBottom: spacing.md,
        borderTopWidth: 1,
        borderTopColor: colors.border.subtle,
    },
    itemDescription: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        paddingTop: spacing.md,
        paddingBottom: spacing.sm,
        lineHeight: 18,
    },

    // Match Section
    matchSection: {
        marginTop: spacing.sm,
    },
    matchSectionTitle: {
        ...typography.scale.labelMedium,
        color: colors.text.secondary,
        marginBottom: spacing.sm,
        fontWeight: '600',
    },
    matchScroll: {
        gap: spacing.sm,
        paddingRight: spacing.md,
    },
    matchCard: {
        width: 110,
        borderRadius: radius.md,
        backgroundColor: colors.background.tertiary,
        overflow: 'hidden',
        paddingBottom: spacing.sm,
    },
    matchImage: {
        width: '100%',
        height: 110,
    },
    matchImagePlaceholder: {
        width: '100%',
        height: 110,
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: colors.background.tertiary,
    },
    matchName: {
        ...typography.scale.labelSmall,
        color: colors.text.primary,
        paddingHorizontal: spacing.xs,
        paddingTop: spacing.xs,
    },
    matchPrice: {
        ...typography.scale.labelSmall,
        color: colors.accent.primary,
        fontWeight: '600',
        paddingHorizontal: spacing.xs,
    },
    matchScoreBadge: {
        position: 'absolute',
        top: spacing.xs,
        right: spacing.xs,
        backgroundColor: 'rgba(10,25,49,0.8)',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: radius.sm,
    },
    matchScoreText: {
        color: '#FFF',
        fontSize: 10,
        fontWeight: '700',
    },

    // No Matches
    noMatches: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.lg,
        gap: spacing.sm,
    },
    noMatchesText: {
        ...typography.scale.bodySmall,
        color: colors.text.tertiary,
    },

    // Error
    errorContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.xxl,
        gap: spacing.md,
    },
    errorText: {
        ...typography.scale.bodyLarge,
        color: colors.text.secondary,
        textAlign: 'center',
        maxWidth: 280,
    },
    retryButton: {
        backgroundColor: '#0A1931',
        paddingHorizontal: spacing.lg,
        paddingVertical: spacing.sm,
        borderRadius: radius.pill,
    },
    retryButtonText: {
        color: '#FFF',
        fontWeight: '600',
    },
});

export default OutfitInspoScreen;
