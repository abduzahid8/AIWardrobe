/**
 * TrialExpiredScreen — non-dismissable paywall gate shown when the
 * 7-day free trial ends and the user has no active subscription.
 *
 * Design goals:
 *  • Premium, cinematic feel — not a generic "locked" screen
 *  • Embedded purchase flow (no navigation needed — purchase callback
 *    updates the store, which reactively removes this screen from the stack)
 *  • App Store compliant: Restore Purchases + terms copy always visible
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
    Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withTiming,
    withSequence,
    Easing,
} from 'react-native-reanimated';
import useSubscriptionStore, {
    SUBSCRIPTION_PRICING,
} from '../store/subscriptionStore';
import useDailyUsageStore from '../store/dailyUsageStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { iapService } from '../src/services/iapService';
import analyticsService from '../src/services/analyticsService';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// ─────────────────────────────────────────────────────────────
// Design tokens
// ─────────────────────────────────────────────────────────────
const C = {
    bg: '#07070F',
    surface: '#12121F',
    border: '#1E1E35',
    text: '#FFFFFF',
    textSub: 'rgba(255,255,255,0.55)',
    textMuted: 'rgba(255,255,255,0.35)',
    pro: '#7C3AED',       // violet — Pro
    proDark: '#4C1D95',
    gold: '#F5A623',      // amber — Max
    goldDark: '#B7730E',
    accent: '#818CF8',
    success: '#34D399',
};

// ─────────────────────────────────────────────────────────────
// Plan cards (same product IDs as PaywallScreen)
// ─────────────────────────────────────────────────────────────
const PLANS = {
    pro: {
        label: 'Pro',
        tagline: 'Everything you need',
        productId: 'com.aiwardrobe.premium.monthly' as const,
        price: `$${SUBSCRIPTION_PRICING.premium.price.toFixed(2)}`,
        period: '/month',
        features: [
            '100 AI outfits / day',
            'Unlimited closet',
            'Analytics & trip planner',
            'Full outfit calendar',
            'Priority AI · no ads',
        ],
        colors: [C.pro, C.proDark] as [string, string],
        cta: 'Get Pro',
        tier: 'premium' as const,
        price_num: SUBSCRIPTION_PRICING.premium.price,
    },
    maxYearly: {
        label: 'Max',
        tagline: `Save ${SUBSCRIPTION_PRICING.vipYearly.savingsPct}% · best value`,
        productId: 'com.aiwardrobe.vip.yearly' as const,
        price: `$${SUBSCRIPTION_PRICING.vipYearly.price.toFixed(2)}`,
        period: '/year',
        monthlyEquiv: `~$${(SUBSCRIPTION_PRICING.vipYearly.price / 12).toFixed(2)}/mo`,
        features: [
            'Everything in Pro',
            'Unlimited Virtual Try-On',
            'Priority support',
            'Early access to features',
        ],
        colors: [C.gold, C.goldDark] as [string, string],
        cta: 'Unlock Max',
        tier: 'vip' as const,
        price_num: SUBSCRIPTION_PRICING.vipYearly.price,
    },
    maxMonthly: {
        label: 'Max',
        tagline: 'The complete AI wardrobe',
        productId: 'com.aiwardrobe.vip.monthly' as const,
        price: `$${SUBSCRIPTION_PRICING.vip.price.toFixed(2)}`,
        period: '/month',
        features: [
            'Everything in Pro',
            'Unlimited Virtual Try-On',
            'Priority support',
            'Early access to features',
        ],
        colors: [C.gold, C.goldDark] as [string, string],
        cta: 'Unlock Max',
        tier: 'vip' as const,
        price_num: SUBSCRIPTION_PRICING.vip.price,
    },
};

// ─────────────────────────────────────────────────────────────
// Animated clock icon in the hero
// ─────────────────────────────────────────────────────────────
const ClockHero: React.FC = () => {
    const rotation = useSharedValue(0);
    useEffect(() => {
        rotation.value = withRepeat(
            withSequence(
                withTiming(15, { duration: 800, easing: Easing.inOut(Easing.quad) }),
                withTiming(-15, { duration: 800, easing: Easing.inOut(Easing.quad) }),
            ),
            -1,
            true,
        );
    }, []);
    const style = useAnimatedStyle(() => ({
        transform: [{ rotate: `${rotation.value}deg` }],
    }));
    return (
        <Animated.View style={[styles.heroIcon, style]}>
            <LinearGradient
                colors={['rgba(245,166,35,0.25)', 'rgba(124,58,237,0.15)']}
                style={styles.heroIconGradient}
            >
                <Ionicons name="time" size={44} color={C.gold} />
            </LinearGradient>
        </Animated.View>
    );
};

// ─────────────────────────────────────────────────────────────
// Main screen
// ─────────────────────────────────────────────────────────────
const TrialExpiredScreen: React.FC = () => {
    const { verifySubscriptionFromServer } = useSubscriptionStore();
    const { completeOnboarding } = useStylePreferenceStore();
    const [loading, setLoading] = useState<string | null>(null);
    const [maxPlan, setMaxPlan] = useState<'yearly' | 'monthly'>('yearly');

    const maxCard = maxPlan === 'yearly' ? PLANS.maxYearly : PLANS.maxMonthly;

    const purchase = async (
        productId:
            | 'com.aiwardrobe.premium.monthly'
            | 'com.aiwardrobe.vip.monthly'
            | 'com.aiwardrobe.vip.yearly',
        tier: 'premium' | 'vip',
        price: number,
    ) => {
        setLoading(productId);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        try {
            const result = await iapService.purchase(productId);
            if (result.success) {
                analyticsService.trackSubscriptionPurchased(tier, price);
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                // This updates the store → navigator reacts → removes this screen
                await verifySubscriptionFromServer();
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            } else {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            }
        } catch {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        } finally {
            setLoading(null);
        }
    };

    const handleRestore = async () => {
        setLoading('restore');
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        try {
            const result = await iapService.restorePurchases();
            if (result.success) {
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                await verifySubscriptionFromServer();
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            } else {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
            }
        } catch {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
        } finally {
            setLoading(null);
        }
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#07070F', '#0E0E22', '#12122A']}
                style={StyleSheet.absoluteFill}
            />

            {/* Ambient orb — visual depth */}
            <View style={styles.orbTop} />
            <View style={styles.orbBottom} />

            <SafeAreaView style={styles.safeArea}>
                <ScrollView
                    contentContainerStyle={styles.scroll}
                    showsVerticalScrollIndicator={false}
                >
                    {/* ── Hero ─────────────────────────────────── */}
                    <Animated.View entering={FadeIn.duration(600)} style={styles.hero}>
                        <ClockHero />

                        <Text style={styles.heroEyebrow}>FREE TRIAL ENDED</Text>
                        <Text style={styles.heroTitle}>
                            Your 7-day trial{'\n'}has ended
                        </Text>
                        <Text style={styles.heroSub}>
                            You've experienced the full power of{'\n'}
                            <Text style={{ color: C.accent, fontWeight: '700' }}>
                                AIWardrobe Pro
                            </Text>
                            {' '}— unlock it to keep going.
                        </Text>

                        {/* Social proof */}
                        <Animated.View
                            entering={FadeInDown.delay(300).duration(500)}
                            style={styles.proof}
                        >
                            <View style={styles.proofAvatars}>
                                {['#7C3AED', '#F5A623', '#34D399', '#818CF8'].map((color, i) => (
                                    <View
                                        key={i}
                                        style={[styles.proofAvatar, { backgroundColor: color, marginLeft: i === 0 ? 0 : -8 }]}
                                    >
                                        <Ionicons name="person" size={11} color="rgba(255,255,255,0.9)" />
                                    </View>
                                ))}
                            </View>
                            <Text style={styles.proofText}>
                                Join <Text style={{ color: C.accent, fontWeight: '700' }}>50,000+</Text> fashion-forward users
                            </Text>
                        </Animated.View>
                    </Animated.View>

                    {/* ── Divider ──────────────────────────────── */}
                    <Animated.View entering={FadeInUp.delay(200).duration(500)} style={styles.dividerRow}>
                        <View style={styles.dividerLine} />
                        <Text style={styles.dividerText}>CHOOSE YOUR PLAN</Text>
                        <View style={styles.dividerLine} />
                    </Animated.View>

                    {/* ── Pro card ─────────────────────────────── */}
                    <Animated.View entering={FadeInUp.delay(300).springify()}>
                        <TouchableOpacity
                            style={styles.card}
                            activeOpacity={0.9}
                            onPress={() => purchase(PLANS.pro.productId, PLANS.pro.tier, PLANS.pro.price_num)}
                        >
                            <LinearGradient
                                colors={PLANS.pro.colors}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 1 }}
                                style={styles.cardGradient}
                            >
                                {/* Most Popular badge */}
                                <View style={styles.badge}>
                                    <Text style={styles.badgeText}>MOST POPULAR</Text>
                                </View>

                                <Text style={styles.planLabel}>{PLANS.pro.label}</Text>
                                <Text style={styles.planTagline}>{PLANS.pro.tagline}</Text>

                                <View style={styles.priceRow}>
                                    <Text style={styles.priceBig}>{PLANS.pro.price}</Text>
                                    <Text style={styles.pricePeriod}>{PLANS.pro.period}</Text>
                                </View>

                                <View style={styles.features}>
                                    {PLANS.pro.features.map((f, i) => (
                                        <View key={i} style={styles.featureRow}>
                                            <Ionicons name="checkmark-circle" size={15} color="rgba(255,255,255,0.9)" />
                                            <Text style={styles.featureText}>{f}</Text>
                                        </View>
                                    ))}
                                </View>

                                <View style={styles.ctaButton}>
                                    {loading === PLANS.pro.productId ? (
                                        <ActivityIndicator color={C.pro} />
                                    ) : (
                                        <Text style={[styles.ctaText, { color: C.pro }]}>
                                            {PLANS.pro.cta}
                                        </Text>
                                    )}
                                </View>
                            </LinearGradient>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* ── Max card ─────────────────────────────── */}
                    <Animated.View entering={FadeInUp.delay(420).springify()}>
                        <View style={[styles.card, styles.cardMaxOuter]}>
                            <LinearGradient
                                colors={maxCard.colors}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 1 }}
                                style={styles.cardGradient}
                            >
                                {/* Best Value badge */}
                                <View style={styles.bestBadge}>
                                    <Ionicons name="diamond" size={9} color={C.bg} />
                                    <Text style={styles.bestBadgeText}>BEST VALUE</Text>
                                </View>

                                <Text style={[styles.planLabel, styles.planLabelDark]}>{maxCard.label}</Text>
                                <Text style={[styles.planTagline, styles.planTaglineDark]}>{maxCard.tagline}</Text>

                                {/* Monthly / Yearly toggle */}
                                <View style={styles.toggle}>
                                    {(['yearly', 'monthly'] as const).map(opt => (
                                        <TouchableOpacity
                                            key={opt}
                                            style={[styles.toggleOption, maxPlan === opt && styles.toggleOptionActive]}
                                            onPress={() => { Haptics.selectionAsync(); setMaxPlan(opt); }}
                                        >
                                            <Text style={[styles.toggleText, maxPlan === opt && styles.toggleTextActive]}>
                                                {opt.charAt(0).toUpperCase() + opt.slice(1)}
                                            </Text>
                                            {opt === 'yearly' && (
                                                <Text style={[styles.toggleSavings, maxPlan === opt && styles.toggleSavingsActive]}>
                                                    Save {SUBSCRIPTION_PRICING.vipYearly.savingsPct}%
                                                </Text>
                                            )}
                                        </TouchableOpacity>
                                    ))}
                                </View>

                                <View style={styles.priceRow}>
                                    <Text style={[styles.priceBig, styles.priceDark]}>{maxCard.price}</Text>
                                    <Text style={[styles.pricePeriod, styles.priceDark]}>{maxCard.period}</Text>
                                </View>
                                {'monthlyEquiv' in maxCard && maxPlan === 'yearly' && (
                                    <Text style={styles.priceEquiv}>{maxCard.monthlyEquiv}</Text>
                                )}

                                <View style={styles.features}>
                                    {maxCard.features.map((f, i) => (
                                        <View key={i} style={styles.featureRow}>
                                            <Ionicons name="checkmark-circle" size={15} color="rgba(0,0,0,0.75)" />
                                            <Text style={[styles.featureText, styles.featureTextDark]}>{f}</Text>
                                        </View>
                                    ))}
                                </View>

                                <TouchableOpacity
                                    style={styles.ctaButtonDark}
                                    activeOpacity={0.85}
                                    onPress={() => purchase(maxCard.productId, maxCard.tier, maxCard.price_num)}
                                >
                                    {loading === maxCard.productId ? (
                                        <ActivityIndicator color={C.gold} />
                                    ) : (
                                        <Text style={[styles.ctaText, { color: C.bg }]}>
                                            {maxCard.cta}
                                        </Text>
                                    )}
                                </TouchableOpacity>
                            </LinearGradient>
                        </View>
                    </Animated.View>

                    {/* ── Restore & Terms ──────────────────────── */}
                    <TouchableOpacity style={styles.restoreBtn} onPress={handleRestore}>
                        {loading === 'restore' ? (
                            <ActivityIndicator color={C.accent} size="small" />
                        ) : (
                            <Text style={styles.restoreText}>Restore Purchases</Text>
                        )}
                    </TouchableOpacity>

                    <Text style={styles.terms}>
                        Cancel anytime. Payment charged to your Apple ID at confirmation.
                        Subscription renews automatically unless canceled at least 24 hours
                        before the end of the current period.
                    </Text>

                    <View style={{ height: 40 }} />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

// ─────────────────────────────────────────────────────────────
// Styles
// ─────────────────────────────────────────────────────────────
const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: C.bg },
    safeArea: { flex: 1 },
    scroll: { paddingHorizontal: 20 },

    // Ambient orbs
    orbTop: {
        position: 'absolute',
        top: -80,
        left: -60,
        width: 280,
        height: 280,
        borderRadius: 140,
        backgroundColor: 'rgba(124,58,237,0.12)',
    },
    orbBottom: {
        position: 'absolute',
        bottom: -60,
        right: -60,
        width: 240,
        height: 240,
        borderRadius: 120,
        backgroundColor: 'rgba(245,166,35,0.10)',
    },

    // Hero
    hero: { alignItems: 'center', paddingTop: 20, paddingBottom: 16 },
    heroIcon: { marginBottom: 18 },
    heroIconGradient: {
        width: 88,
        height: 88,
        borderRadius: 44,
        alignItems: 'center',
        justifyContent: 'center',
    },
    heroEyebrow: {
        fontSize: 11,
        fontWeight: '800',
        letterSpacing: 2.5,
        color: C.gold,
        marginBottom: 10,
    },
    heroTitle: {
        fontSize: 34,
        fontWeight: '800',
        color: C.text,
        textAlign: 'center',
        lineHeight: 40,
        marginBottom: 12,
    },
    heroSub: {
        fontSize: 15,
        color: C.textSub,
        textAlign: 'center',
        lineHeight: 22,
        paddingHorizontal: 10,
        marginBottom: 18,
    },

    // Social proof
    proof: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.06)',
        borderRadius: 20,
        paddingVertical: 8,
        paddingHorizontal: 14,
        gap: 10,
    },
    proofAvatars: { flexDirection: 'row' },
    proofAvatar: {
        width: 24,
        height: 24,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1.5,
        borderColor: C.bg,
    },
    proofText: { fontSize: 12, color: C.textSub, fontWeight: '500' },

    // Divider
    dividerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginVertical: 18,
        gap: 10,
    },
    dividerLine: { flex: 1, height: 1, backgroundColor: C.border },
    dividerText: {
        fontSize: 11,
        fontWeight: '700',
        letterSpacing: 1.5,
        color: C.textMuted,
    },

    // Cards
    card: {
        borderRadius: 22,
        overflow: 'hidden',
        marginBottom: 14,
        ...Platform.select({
            ios: {
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 10 },
                shadowOpacity: 0.4,
                shadowRadius: 18,
            },
            android: { elevation: 8 },
        }),
    },
    cardMaxOuter: {
        borderWidth: 1.5,
        borderColor: C.gold,
    },
    cardGradient: { padding: 22, position: 'relative' },

    // Badges
    badge: {
        position: 'absolute',
        top: 14,
        right: 14,
        backgroundColor: 'rgba(0,0,0,0.35)',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 10,
    },
    badgeText: { color: '#FFF', fontSize: 9, fontWeight: '800', letterSpacing: 0.8 },
    bestBadge: {
        position: 'absolute',
        top: 14,
        right: 14,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        backgroundColor: 'rgba(255,255,255,0.92)',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 10,
    },
    bestBadgeText: { color: C.bg, fontSize: 9, fontWeight: '800', letterSpacing: 0.8 },

    // Plan text
    planLabel: { fontSize: 26, fontWeight: '800', color: '#FFF', marginBottom: 2 },
    planLabelDark: { color: C.bg },
    planTagline: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.8)',
        marginBottom: 14,
        fontWeight: '500',
    },
    planTaglineDark: { color: 'rgba(0,0,0,0.65)' },

    // Price
    priceRow: { flexDirection: 'row', alignItems: 'flex-end', marginBottom: 4 },
    priceBig: { fontSize: 38, fontWeight: '800', color: '#FFF' },
    priceDark: { color: C.bg },
    pricePeriod: {
        fontSize: 14,
        color: 'rgba(255,255,255,0.75)',
        marginBottom: 7,
        marginLeft: 4,
    },
    priceEquiv: {
        fontSize: 12,
        color: 'rgba(0,0,0,0.6)',
        fontWeight: '600',
        marginBottom: 12,
    },

    // Toggle
    toggle: {
        flexDirection: 'row',
        backgroundColor: 'rgba(0,0,0,0.15)',
        borderRadius: 12,
        padding: 4,
        marginBottom: 16,
    },
    toggleOption: { flex: 1, paddingVertical: 10, borderRadius: 9, alignItems: 'center' },
    toggleOptionActive: { backgroundColor: 'rgba(255,255,255,0.9)' },
    toggleText: { fontSize: 13, fontWeight: '700', color: 'rgba(0,0,0,0.5)' },
    toggleTextActive: { color: C.bg },
    toggleSavings: { fontSize: 10, fontWeight: '600', color: 'rgba(0,0,0,0.35)', marginTop: 2 },
    toggleSavingsActive: { color: C.proDark },

    // Features
    features: { marginTop: 10, marginBottom: 16 },
    featureRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 8, gap: 10 },
    featureText: { fontSize: 14, color: '#FFF', fontWeight: '500', flex: 1 },
    featureTextDark: { color: 'rgba(0,0,0,0.8)' },

    // CTAs
    ctaButton: {
        backgroundColor: '#FFF',
        paddingVertical: 14,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 4,
    },
    ctaButtonDark: {
        backgroundColor: C.bg,
        paddingVertical: 14,
        borderRadius: 12,
        alignItems: 'center',
        marginTop: 4,
    },
    ctaText: { fontSize: 15, fontWeight: '800', letterSpacing: 0.3 },

    // Footer
    restoreBtn: { alignItems: 'center', paddingVertical: 16 },
    restoreText: { fontSize: 13, color: C.accent, fontWeight: '600' },
    terms: {
        fontSize: 11,
        color: C.textMuted,
        textAlign: 'center',
        lineHeight: 16,
        paddingHorizontal: 10,
        marginBottom: 16,
    },
});

export default TrialExpiredScreen;
