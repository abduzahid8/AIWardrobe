/**
 * PaywallScreen — the upgrade surface for Free / Pro.
 *
 * Conversion hooks baked in:
 *   1. 2-tier layout (Free → Pro).
 *   2. "Most Popular" badge on Pro (social proof default).
 *   3. Explicit feature comparison list (Free vs Pro) so users
 *      see exactly what they're missing.
 *   4. Friction-free close — Free is forever, close just goes back.
 *   5. Restore Purchases link (App Store compliance).
 */

import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ScrollView,
    Platform,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import useSubscriptionStore, {
    SUBSCRIPTION_PRICING,
    TIER_DISPLAY_NAMES,
} from '../store/subscriptionStore';
import useDailyUsageStore from '../store/dailyUsageStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import AppColors from '../constants/AppColors';
import { iapService } from '../src/services/iapService';
import analyticsService from '../src/services/analyticsService';

const COLORS = {
    background: '#0A0A0A',
    surface: '#1A1A1A',
    surfaceLight: '#2A2A2A',
    text: '#FFFFFF',
    textSecondary: AppColors.textMuted,
    premium: AppColors.premium,       // Pro accent
    premiumDark: AppColors.premiumDark,
    accent: AppColors.accent,
    success: AppColors.success,
    border: '#333333',
    free: '#6B7280',
};

// ──────────────────────────────────────────────────────────
// Feature comparison matrix — exact copy shown on the paywall.
// Keep this in sync with store/subscriptionStore.ts TIER_FEATURES.
// ──────────────────────────────────────────────────────────
const TIER_CARDS = {
    free: {
        label: 'Free',
        sublabel: '7-day free trial · no credit card',
        price: null as string | null,
        features: [
            '10 AI outfits / day',
            'Up to 20 items in your closet',
            'Basic today-view calendar',
            'Browse inspiration',
        ],
        missing: [
            'No Virtual Try-On',
            'No wardrobe insights',
            'No trip planner',
        ],
    },
    pro: {
        label: 'Pro',
        sublabel: 'The complete AI wardrobe',
        productId: 'com.aiwardrobe.premium.monthly' as const,
        price: null as string | null,
        period: '/month',
        features: [
            'Unlimited AI outfits',
            'Unlimited Virtual Try-On',
            'Unlimited closet size',
            'Wardrobe insights & stats',
            'AI trip / travel planner',
            'Full outfit calendar',
            'Priority AI model · no ads',
            'Priority support',
            'Early access to new features',
        ],
    },
};

// ──────────────────────────────────────────────────────────
// Pressable card with subtle scale-down on press
// ──────────────────────────────────────────────────────────
const PressableCard: React.FC<{
    onPress: () => void;
    children: React.ReactNode;
    style?: any;
}> = ({ onPress, children, style }) => {
    const scale = useSharedValue(1);
    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 15, stiffness: 400 }) }],
    }));
    return (
        <TouchableOpacity
            onPressIn={() => { scale.value = 0.98; }}
            onPressOut={() => { scale.value = 1; }}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                onPress();
            }}
            activeOpacity={1}
        >
            <Animated.View style={[style, animatedStyle]}>{children}</Animated.View>
        </TouchableOpacity>
    );
};

const PaywallScreen = () => {
    const navigation = useNavigation<any>();
    const { verifySubscriptionFromServer } = useSubscriptionStore();
    const { completeOnboarding } = useStylePreferenceStore();
    const [isLoading, setIsLoading] = useState<string | null>(null);
    const [livePrice, setLivePrice] = useState<string | null>(null);
    const canGoBack = navigation.canGoBack();

    useEffect(() => {
        let cancelled = false;
        iapService.getProducts().then((products) => {
            if (cancelled) return;
            const proProduct = products.find(
                (p) => p.id === TIER_CARDS.pro.productId
            );
            if (proProduct?.price) {
                setLivePrice(proProduct.price);
            }
        }).catch(() => {
            // Keep fallback empty; UI will show nothing or a placeholder
        });
        return () => { cancelled = true; };
    }, []);

    const purchase = async (
        productId: 'com.aiwardrobe.premium.monthly',
        analyticsTier: 'premium',
        analyticsPrice: number,
    ) => {
        setIsLoading(productId);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        try {
            const result = await iapService.purchase(productId);
            if (result.success) {
                // Analytics already tracked inside iapService.purchase()
                completeOnboarding();
                // Give the user a fresh daily bucket after upgrading.
                await useDailyUsageStore.getState().resetToday();
                await verifySubscriptionFromServer();
                navigation.reset({
                    index: 0,
                    routes: [{ name: 'Main' as never }],
                });
            } else if (result.error) {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            }
        } catch {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        } finally {
            setIsLoading(null);
        }
    };

    const handleRestore = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setIsLoading('restore');
        try {
            const result = await iapService.restorePurchases();
            if (result.success) {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                await verifySubscriptionFromServer();
                navigation.reset({
                    index: 0,
                    routes: [{ name: 'Main' as never }],
                });
            } else {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            }
        } catch {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        } finally {
            setIsLoading(null);
        }
    };

    const closePaywall = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        if (canGoBack) {
            navigation.goBack();
        } else {
            completeOnboarding();
            navigation.reset({
                index: 0,
                routes: [{ name: 'Main' as never }],
            });
        }
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#0A0A0A', '#1A1A2E', '#16213E']}
                style={styles.gradient}
            >
                <SafeAreaView style={styles.safeArea}>
                    {/* Header */}
                    <View style={styles.header}>
                        <TouchableOpacity style={styles.closeButton} onPress={closePaywall}>
                            <Ionicons name="close" size={26} color={COLORS.text} />
                        </TouchableOpacity>
                    </View>

                    <ScrollView
                        contentContainerStyle={styles.scrollContent}
                        showsVerticalScrollIndicator={false}
                    >
                        {/* Hero */}
                        <Animated.View entering={FadeIn.duration(500)} style={styles.hero}>
                            <View style={styles.iconContainer}>
                                <Ionicons name="sparkles" size={40} color={COLORS.premium} />
                            </View>
                            <Text style={styles.heroTitle}>Your style, unlocked.</Text>
                            <Text style={styles.heroSubtitle}>
                                Keep Free forever — or upgrade when you're ready for more.
                            </Text>
                        </Animated.View>

                        {/* FREE tier (informational — already active) */}
                        <Animated.View entering={FadeInUp.delay(100).springify()}>
                            <View style={[styles.card, styles.cardFree]}>
                                <View style={styles.cardHeaderRow}>
                                    <Text style={styles.tierName}>{TIER_CARDS.free.label}</Text>
                                    <View style={styles.currentBadge}>
                                        <Text style={styles.currentBadgeText}>CURRENT</Text>
                                    </View>
                                </View>
                                <Text style={styles.tierSub}>{TIER_CARDS.free.sublabel}</Text>
                                <View style={styles.featuresBlock}>
                                    {TIER_CARDS.free.features.map((f, i) => (
                                        <View key={i} style={styles.featureRow}>
                                            <Ionicons name="checkmark" size={16} color={COLORS.success} />
                                            <Text style={styles.featureText}>{f}</Text>
                                        </View>
                                    ))}
                                    {TIER_CARDS.free.missing.map((f, i) => (
                                        <View key={`m-${i}`} style={styles.featureRow}>
                                            <Ionicons name="lock-closed" size={14} color={COLORS.textSecondary} />
                                            <Text style={[styles.featureText, styles.featureMuted]}>{f}</Text>
                                        </View>
                                    ))}
                                </View>
                            </View>
                        </Animated.View>

                        {/* PRO tier */}
                        <Animated.View entering={FadeInUp.delay(200).springify()}>
                            <PressableCard
                                style={styles.card}
                                onPress={() =>
                                    purchase(
                                        TIER_CARDS.pro.productId,
                                        'premium',
                                        SUBSCRIPTION_PRICING.premium.price,
                                    )
                                }
                            >
                                <LinearGradient
                                    colors={[COLORS.premium, COLORS.premiumDark]}
                                    start={{ x: 0, y: 0 }}
                                    end={{ x: 1, y: 1 }}
                                    style={styles.cardGradient}
                                >
                                    <View style={styles.popularBadge}>
                                        <Text style={styles.popularBadgeText}>MOST POPULAR</Text>
                                    </View>
                                    <Text style={styles.tierNameLight}>{TIER_CARDS.pro.label}</Text>
                                    <Text style={styles.tierSubLight}>{TIER_CARDS.pro.sublabel}</Text>
                                    <View style={styles.priceRow}>
                                        <Text style={styles.priceBig}>{livePrice ?? TIER_CARDS.pro.price ?? ''}</Text>
                                        <Text style={styles.pricePeriod}>{TIER_CARDS.pro.period}</Text>
                                    </View>
                                    <View style={styles.featuresBlock}>
                                        {TIER_CARDS.pro.features.map((f, i) => (
                                            <View key={i} style={styles.featureRow}>
                                                <Ionicons name="checkmark-circle" size={16} color={COLORS.text} />
                                                <Text style={styles.featureTextLight}>{f}</Text>
                                            </View>
                                        ))}
                                    </View>
                                    <View style={styles.ctaButton}>
                                        {isLoading === TIER_CARDS.pro.productId ? (
                                            <ActivityIndicator color={COLORS.premium} />
                                        ) : (
                                            <Text style={[styles.ctaText, { color: COLORS.premium }]}>
                                                Get Pro
                                            </Text>
                                        )}
                                    </View>
                                </LinearGradient>
                            </PressableCard>
                        </Animated.View>

                        {/* Restore + Terms */}
                        <TouchableOpacity style={styles.restoreButton} onPress={handleRestore}>
                            {isLoading === 'restore' ? (
                                <ActivityIndicator color={COLORS.accent} />
                            ) : (
                                <Text style={styles.restoreText}>Restore Purchases</Text>
                            )}
                        </TouchableOpacity>

                        <Text style={styles.termsText}>
                            7-day free trial, then $0 forever on the Free plan. Paid plans
                            are charged to your Apple ID at confirmation. Subscriptions
                            renew automatically unless canceled at least 24 hours before
                            the end of the current period.
                        </Text>

                        <View style={{ height: 40 }} />
                    </ScrollView>
                </SafeAreaView>
            </LinearGradient>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: COLORS.background },
    gradient: { flex: 1 },
    safeArea: { flex: 1 },

    header: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        paddingHorizontal: 20,
        paddingVertical: 10,
    },
    closeButton: {
        width: 38,
        height: 38,
        borderRadius: 19,
        backgroundColor: COLORS.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
    },

    scrollContent: { paddingHorizontal: 20 },

    hero: { alignItems: 'center', marginBottom: 22 },
    iconContainer: {
        width: 76,
        height: 76,
        borderRadius: 38,
        backgroundColor: 'rgba(255, 215, 0, 0.15)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 14,
    },
    heroTitle: {
        fontSize: 28,
        fontWeight: '800',
        color: COLORS.text,
        marginBottom: 8,
        textAlign: 'center',
    },
    heroSubtitle: {
        fontSize: 14,
        color: COLORS.textSecondary,
        textAlign: 'center',
        lineHeight: 20,
        paddingHorizontal: 20,
    },

    // Cards
    card: {
        borderRadius: 20,
        overflow: 'hidden',
        marginBottom: 14,
        ...Platform.select({
            ios: {
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 8 },
                shadowOpacity: 0.28,
                shadowRadius: 14,
            },
            android: { elevation: 6 },
        }),
    },
    cardFree: {
        backgroundColor: COLORS.surface,
        borderWidth: 1,
        borderColor: COLORS.border,
        padding: 18,
    },
    cardGradient: { padding: 20, position: 'relative' },

    cardHeaderRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
    },
    currentBadge: {
        backgroundColor: COLORS.free,
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 8,
    },
    currentBadgeText: {
        color: COLORS.text,
        fontSize: 10,
        fontWeight: '700',
        letterSpacing: 0.5,
    },
    popularBadge: {
        position: 'absolute',
        top: 12,
        right: 12,
        backgroundColor: 'rgba(0,0,0,0.35)',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 10,
    },
    popularBadgeText: {
        color: COLORS.text,
        fontSize: 10,
        fontWeight: '700',
        letterSpacing: 0.5,
    },
    tierName: {
        fontSize: 22,
        fontWeight: '800',
        color: COLORS.text,
    },
    tierNameLight: {
        fontSize: 26,
        fontWeight: '800',
        color: COLORS.text,
    },
    tierNameDark: { color: '#0A0A0A' },
    tierSub: {
        fontSize: 13,
        color: COLORS.textSecondary,
        marginTop: 4,
        marginBottom: 8,
    },
    tierSubLight: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.85)',
        marginTop: 4,
        marginBottom: 14,
    },
    tierSubDark: { color: 'rgba(0,0,0,0.75)' },

    priceRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        marginBottom: 4,
    },
    priceBig: {
        fontSize: 36,
        fontWeight: '800',
        color: COLORS.text,
    },
    priceDark: { color: '#0A0A0A' },
    pricePeriod: {
        fontSize: 14,
        color: 'rgba(255,255,255,0.8)',
        marginBottom: 6,
        marginLeft: 4,
    },
    priceEquiv: {
        fontSize: 12,
        color: 'rgba(0,0,0,0.7)',
        marginBottom: 12,
        fontWeight: '600',
    },

    // Features
    featuresBlock: { marginTop: 12, marginBottom: 16 },
    featureRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 8,
        gap: 10,
    },
    featureText: {
        fontSize: 14,
        color: COLORS.text,
        fontWeight: '500',
    },
    featureTextLight: {
        fontSize: 14,
        color: COLORS.text,
        fontWeight: '500',
    },
    featureTextDark: { color: '#0A0A0A' },
    featureMuted: { color: COLORS.textSecondary },
    featureLightMuted: { color: 'rgba(255,255,255,0.55)' },

    // CTAs
    ctaButton: {
        backgroundColor: COLORS.text,
        paddingVertical: 14,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 6,
    },
    ctaButtonDark: {
        backgroundColor: '#0A0A0A',
        paddingVertical: 14,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 6,
    },
    ctaText: { fontSize: 15, fontWeight: '800' },

    restoreButton: { alignItems: 'center', paddingVertical: 14 },
    restoreText: {
        fontSize: 13,
        color: COLORS.accent,
        fontWeight: '600',
    },
    termsText: {
        fontSize: 11,
        color: COLORS.textSecondary,
        textAlign: 'center',
        lineHeight: 16,
        paddingHorizontal: 10,
        marginBottom: 20,
    },
});

export default PaywallScreen;
