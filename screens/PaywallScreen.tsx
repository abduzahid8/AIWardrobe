/**
 * PaywallScreen — Liquid Glass upgrade surface.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Platform,
    ActivityIndicator,
    Alert,
    StatusBar,
    ViewStyle,
    Linking,
} from 'react-native';

// Apple Standard EULA — required by Guideline 3.1.2(c)
const APPLE_EULA_URL = 'https://www.apple.com/legal/internet-services/itunes/dev/stdeula/';
const PRIVACY_POLICY_URL = 'https://aiwardrobe.app/privacy'; // Update to your hosted URL
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import { useNavigation } from '@react-navigation/native';
import { navigationRef } from '../navigation/navigationRef';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    interpolate,
    Extrapolation,
    useAnimatedScrollHandler,
} from 'react-native-reanimated';
import useSubscriptionStore, { SUBSCRIPTION_PRICING } from '../store/subscriptionStore';
import useDailyUsageStore from '../store/dailyUsageStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { iapService } from '../src/services/iapService';
import { useTranslation } from 'react-i18next';
import { useTheme } from '../src/theme/ThemeContext';

type BlurTint = 'light' | 'dark';
type IconName = keyof typeof Ionicons.glyphMap;

function useDesignTokens() {
    const { colors, isDark } = useTheme();
    return {
        isDark,
        tint: (isDark ? 'dark' : 'light') as BlurTint,
        bg: colors.background,
        glass: isDark ? 'rgba(17, 20, 30, 0.58)' : 'rgba(255, 255, 255, 0.56)',
        glassStrong: isDark ? 'rgba(14, 16, 26, 0.76)' : 'rgba(255, 255, 255, 0.76)',
        glassMuted: isDark ? 'rgba(255, 255, 255, 0.05)' : 'rgba(255, 255, 255, 0.30)',
        glassBorder: isDark ? 'rgba(255, 255, 255, 0.14)' : 'rgba(255, 255, 255, 0.68)',
        glassBorderSoft: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(10, 25, 49, 0.06)',
        text: colors.text.primary,
        textSub: colors.text.secondary,
        textMute: colors.text.muted,
        accent: isDark ? '#A8C0DA' : '#12385F',
        accentStart: isDark ? '#446B95' : '#2A537F',
        accentEnd: isDark ? '#1C3654' : '#0D2743',
        accentSoft: isDark ? 'rgba(126, 162, 201, 0.18)' : 'rgba(18, 56, 95, 0.12)',
        accentSoftStrong: isDark ? 'rgba(126, 162, 201, 0.28)' : 'rgba(42, 83, 127, 0.20)',
        success: colors.success,
        white: '#FFFFFF',
        heroGradient: (isDark
            ? ['#142338', '#0D1828', '#070C15']
            : ['#F2F7FC', '#ECF3FA', '#FAFCFF']) as readonly [string, string, string],
        panelHighlight: (isDark
            ? ['rgba(255,255,255,0.12)', 'rgba(255,255,255,0.02)']
            : ['rgba(255,255,255,0.82)', 'rgba(255,255,255,0.16)']) as readonly [string, string],
        buttonHighlight: (isDark
            ? ['rgba(255,255,255,0.08)', 'rgba(255,255,255,0.02)']
            : ['rgba(255,255,255,0.70)', 'rgba(255,255,255,0.12)']) as readonly [string, string],
        orbPrimary: isDark ? 'rgba(68, 107, 149, 0.30)' : 'rgba(42, 83, 127, 0.18)',
        orbSecondary: isDark ? 'rgba(122, 161, 203, 0.18)' : 'rgba(121, 158, 198, 0.14)',
        orbWarm: isDark ? 'rgba(166, 191, 219, 0.12)' : 'rgba(188, 210, 231, 0.22)',
        shadow: isDark ? '#000000' : '#4C6076',
        overlay: isDark ? 'rgba(7, 9, 16, 0.74)' : 'rgba(16, 18, 28, 0.34)',
    };
}

type DTokens = ReturnType<typeof useDesignTokens>;

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
    const D = useDesignTokens();
    const styles = useMemo(() => createStyles(D), [D]);
    const insets = useSafeAreaInsets();
    const navigation = useNavigation<any>();
    const { completeOnboarding } = useStylePreferenceStore();
    const { t } = useTranslation();
    const [isLoading, setIsLoading] = useState<string | null>(null);
    const [livePrice, setLivePrice] = useState<string | null>(null);
    const [liveProductId, setLiveProductId] = useState<string | null>(null);
    const [liveYearlyPrice, setLiveYearlyPrice] = useState<string | null>(null);
    const [liveYearlyProductId, setLiveYearlyProductId] = useState<string | null>(null);
    const subscriptionTier = useSubscriptionStore((s) => s.tier);
    const needsPromoCode = useSubscriptionStore((s) => s.needsPromoCode);

    const scrollY = useSharedValue(0);
    const scrollHandler = useAnimatedScrollHandler((event) => {
        scrollY.value = event.contentOffset.y;
    });

    const heroAnimStyle = useAnimatedStyle(() => ({
        transform: [
            {
                translateY: interpolate(
                    scrollY.value,
                    [0, 200],
                    [0, -40],
                    Extrapolation.CLAMP
                ),
            },
            {
                scale: interpolate(scrollY.value, [-50, 0, 200], [1.02, 1, 0.96], Extrapolation.CLAMP),
            },
        ],
        opacity: interpolate(scrollY.value, [0, 160], [1, 0], Extrapolation.CLAMP),
    }));

    const triggerLightHaptic = () => {
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    };

    const GlassPanel = ({
        children,
        style,
        radius = 30,
        intensity = 42,
    }: {
        children: React.ReactNode;
        style?: ViewStyle | ViewStyle[];
        radius?: number;
        intensity?: number;
    }) => (
        <View style={[styles.glassShadow, { borderRadius: radius }, style]}>
            <View style={[styles.glassPanel, { borderRadius: radius }]}>
                <BlurView
                    intensity={Platform.OS === 'ios' ? intensity : 100}
                    tint={D.tint}
                    style={StyleSheet.absoluteFillObject}
                />
                <LinearGradient
                    colors={D.panelHighlight}
                    start={{ x: 0.08, y: 0 }}
                    end={{ x: 0.92, y: 1 }}
                    style={StyleSheet.absoluteFillObject}
                />
                <View style={[styles.glassScrim, { borderRadius: radius }]} />
                <View style={styles.glassContent}>{children}</View>
            </View>
        </View>
    );

    const PrimaryGradientButton = ({
        label,
        icon,
        onPress,
        style,
        loading,
    }: {
        label: string;
        icon: IconName;
        onPress: () => void;
        style?: ViewStyle | ViewStyle[];
        loading?: boolean;
    }) => (
        <TouchableOpacity
            activeOpacity={0.85}
            style={style}
            onPress={() => {
                triggerLightHaptic();
                onPress();
            }}
        >
            <LinearGradient
                colors={[D.accentStart, D.accentEnd]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 1 }}
                style={styles.primaryButton}
            >
                {loading ? (
                    <ActivityIndicator color={D.white} size="small" />
                ) : (
                    <>
                        <Ionicons name={icon} size={16} color={D.white} />
                        <Text style={styles.primaryButtonText}>{label}</Text>
                    </>
                )}
            </LinearGradient>
        </TouchableOpacity>
    );

    const TIER_CARDS = {
        free: {
            label: t('subscription.tiers.free'),
            sublabel: t('subscription.tiers.freeSublabel'),
            price: null as string | null,
            features: [
                t('subscription.tiers.freeFeatures.0'),
                t('subscription.tiers.freeFeatures.1'),
                t('subscription.tiers.freeFeatures.2'),
                t('subscription.tiers.freeFeatures.3'),
            ],
            missing: [
                t('subscription.tiers.freeMissing.0'),
                t('subscription.tiers.freeMissing.1'),
                t('subscription.tiers.freeMissing.2'),
            ],
        },
        pro: {
            label: t('subscription.tiers.pro'),
            sublabel: t('subscription.tiers.proSublabel'),
            productId: 'com.aiwardrobe.premium.monthly' as const,
            price: null as string | null,
            period: t('subscription.tiers.period'),
            features: [
                t('subscription.tiers.proFeatures.0'),
                t('subscription.tiers.proFeatures.1'),
                t('subscription.tiers.proFeatures.2'),
                t('subscription.tiers.proFeatures.3'),
                t('subscription.tiers.proFeatures.4'),
                t('subscription.tiers.proFeatures.5'),
                t('subscription.tiers.proFeatures.6'),
                t('subscription.tiers.proFeatures.7'),
                t('subscription.tiers.proFeatures.8'),
            ],
        },
        max: {
            label: t('subscription.tiers.max'),
            sublabel: t('subscription.tiers.maxSublabel'),
            productId: 'com.aiwardrobe.premium.yearly' as const,
            price: null as string | null,
            period: t('subscription.tiers.maxPeriod'),
            features: [
                t('subscription.tiers.proFeatures.0'),
                t('subscription.tiers.proFeatures.1'),
                t('subscription.tiers.proFeatures.2'),
                t('subscription.tiers.proFeatures.3'),
                t('subscription.tiers.proFeatures.4'),
                t('subscription.tiers.proFeatures.5'),
                t('subscription.tiers.proFeatures.6'),
                t('subscription.tiers.proFeatures.7'),
                t('subscription.tiers.proFeatures.8'),
                t('subscription.tiers.maxFeatures.0'),
                t('subscription.tiers.maxFeatures.1'),
            ],
            savingsLabel: t('subscription.tiers.maxSavings'),
        },
    };
    const canGoBack = navigation.canGoBack();

    useEffect(() => {
        let cancelled = false;
        iapService.getProducts().then((products) => {
            if (cancelled) return;
            const proProduct = products.find(
                (p) => p.id === TIER_CARDS.pro.productId
            ) || products.find((p) => {
                const id = String(p.id || '').toLowerCase();
                const title = String(p.title || '').toLowerCase();
                return id.includes('premium') || id.includes('pro') || title.includes('premium') || title.includes('pro');
            }) || (products.length === 1 ? products[0] : undefined);
            if (proProduct?.price) {
                setLivePrice(proProduct.price);
            }
            if (proProduct?.id) {
                setLiveProductId(proProduct.id);
            }
            const yearlyProduct = products.find(
                (p) => p.id === TIER_CARDS.max.productId
            ) || products.find((p) => {
                const id = String(p.id || '').toLowerCase();
                return id.includes('yearly') || id.includes('annual');
            });
            if (yearlyProduct?.price) {
                setLiveYearlyPrice(yearlyProduct.price);
            }
            if (yearlyProduct?.id) {
                setLiveYearlyProductId(yearlyProduct.id);
            }
        }).catch(() => {
            // Keep fallback empty; UI will show nothing or a placeholder
        });
        return () => { cancelled = true; };
    }, []);

    /**
     * Return to the app after a successful purchase or restore.
     * The subscription store will automatically flip hasActiveSubscription → true,
     * which causes RootNavigator to render the 'Main' stack if it wasn't already.
     */
    const resetToMain = () => {
        setTimeout(() => {
            if (canGoBack) {
                navigation.goBack();
            } else if (navigationRef.isReady()) {
                // Safely reset stack and navigate to Main tab
                navigation.reset({
                    index: 0,
                    routes: [{ name: 'Main' }],
                });
            }
        }, 300);
    };

    const purchase = async (productId: string) => {
        console.log('[Paywall] purchase() called with productId:', productId);
        setIsLoading(productId);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        try {
            console.log('[Paywall] calling iapService.purchase...');
            const result = await iapService.purchase(productId);
            console.log('[Paywall] iapService.purchase returned:', result);
            if (result.success) {
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                resetToMain();
            } else if (result.error) {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
                Alert.alert(t('paywall.purchaseFailed'), result.error);
            }
        } catch (error) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            Alert.alert(
                t('paywall.purchaseFailed'),
                error instanceof Error ? error.message : t('paywall.somethingWrongPurchase')
            );
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
                resetToMain();
            } else {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
                Alert.alert(t('paywall.restoreFailed'), result.error || t('paywall.noPreviousPurchases'));
            }
        } catch (error) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            Alert.alert(
                t('paywall.restoreFailed'),
                error instanceof Error ? error.message : t('paywall.somethingWrongRestore')
            );
        } finally {
            setIsLoading(null);
        }
    };

    const closePaywall = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        // Never redirect to PromoCode — Apple Guideline 2.1(a) requires a free tier
        // path so users are never forced to subscribe. If we can go back, do so;
        // otherwise complete onboarding and land on Main (free tier).
        if (canGoBack) {
            navigation.goBack();
        } else {
            completeOnboarding();
            resetToMain();
        }
    };

    return (
        <View style={{ flex: 1, backgroundColor: D.bg }}>
            <StatusBar barStyle={D.isDark ? 'light-content' : 'dark-content'} />
            <LinearGradient
                colors={D.heroGradient}
                start={{ x: 0, y: 0 }}
                end={{ x: 0, y: 1 }}
                style={StyleSheet.absoluteFillObject}
            />

            {/* Background orbs */}
            <Animated.View style={[styles.orb, styles.orbPrimary]} entering={FadeIn.duration(800)} />
            <Animated.View style={[styles.orb, styles.orbSecondary]} entering={FadeIn.duration(1000).delay(200)} />
            <Animated.View style={[styles.orb, styles.orbWarm]} entering={FadeIn.duration(1200).delay(400)} />

            <Animated.ScrollView
                onScroll={scrollHandler}
                scrollEventThrottle={16}
                contentContainerStyle={{
                    paddingTop: insets.top + 12,
                    paddingBottom: insets.bottom + 32,
                }}
                showsVerticalScrollIndicator={false}
            >
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity style={styles.closeButton} onPress={closePaywall}>
                        <Ionicons name="close" size={26} color={D.text} />
                    </TouchableOpacity>
                </View>

                {/* Hero */}
                <Animated.View entering={FadeIn.duration(500)} style={[styles.hero, heroAnimStyle]}>
                    <View style={styles.iconContainer}>
                        <Ionicons name="sparkles" size={40} color={D.accent} />
                    </View>
                    <Text style={styles.heroTitle}>{t('paywall.heroTitle')}</Text>
                    <Text style={styles.heroSubtitle}>{t('paywall.heroSubtitle')}</Text>
                </Animated.View>

                {/* FREE tier (informational — already active) */}
                <Animated.View entering={FadeInUp.delay(100).springify()}>
                    <GlassPanel style={styles.card} radius={22} intensity={36}>
                        <View style={styles.cardHeaderRow}>
                            <Text style={styles.tierName}>{TIER_CARDS.free.label}</Text>
                            <View style={styles.currentBadge}>
                                <Text style={styles.currentBadgeText}>{t('paywall.current')}</Text>
                            </View>
                        </View>
                        <Text style={styles.tierSub}>{TIER_CARDS.free.sublabel}</Text>
                        <View style={styles.featuresBlock}>
                            {TIER_CARDS.free.features.map((f, i) => (
                                <View key={i} style={styles.featureRow}>
                                    <Ionicons name="checkmark" size={16} color={D.success} />
                                    <Text style={styles.featureText}>{f}</Text>
                                </View>
                            ))}
                            {TIER_CARDS.free.missing.map((f, i) => (
                                <View key={`m-${i}`} style={styles.featureRow}>
                                    <Ionicons name="lock-closed" size={14} color={D.textMute} />
                                    <Text style={[styles.featureText, styles.featureMuted]}>{f}</Text>
                                </View>
                            ))}
                        </View>
                    </GlassPanel>
                </Animated.View>

                {/* PRO tier */}
                <Animated.View entering={FadeInUp.delay(200).springify()}>
                    <PressableCard
                        style={styles.card}
                        onPress={() => purchase(liveProductId ?? TIER_CARDS.pro.productId)}
                    >
                        <GlassPanel style={styles.card} radius={22} intensity={48}>
                            <View style={styles.popularBadge}>
                                <Text style={styles.popularBadgeText}>{t('paywall.mostPopular')}</Text>
                            </View>
                            <Text style={styles.tierName}>{TIER_CARDS.pro.label}</Text>
                            <Text style={styles.tierSub}>{TIER_CARDS.pro.sublabel}</Text>
                            <View style={styles.priceRow}>
                                <Text style={styles.priceBig}>{livePrice ?? `$${SUBSCRIPTION_PRICING.premium.price.toFixed(2)}`}</Text>
                                <Text style={styles.pricePeriod}>{TIER_CARDS.pro.period}</Text>
                            </View>
                            <View style={styles.featuresBlock}>
                                {TIER_CARDS.pro.features.map((f, i) => (
                                    <View key={i} style={styles.featureRow}>
                                        <Ionicons name="checkmark-circle" size={16} color={D.accent} />
                                        <Text style={styles.featureText}>{f}</Text>
                                    </View>
                                ))}
                            </View>
                            <PrimaryGradientButton
                                label={t('paywall.getPro')}
                                icon="star"
                                loading={isLoading === (liveProductId ?? TIER_CARDS.pro.productId)}
                                onPress={() => purchase(liveProductId ?? TIER_CARDS.pro.productId)}
                            />
                        </GlassPanel>
                    </PressableCard>
                </Animated.View>

                {/* MAX (Yearly) tier */}
                <Animated.View entering={FadeInUp.delay(300).springify()}>
                    <PressableCard
                        style={styles.card}
                        onPress={() => purchase(liveYearlyProductId ?? TIER_CARDS.max.productId)}
                    >
                        <GlassPanel style={styles.card} radius={22} intensity={36}>
                            <View style={styles.savingsBadge}>
                                <Text style={styles.savingsBadgeText}>{TIER_CARDS.max.savingsLabel}</Text>
                            </View>
                            <Text style={styles.tierName}>{TIER_CARDS.max.label}</Text>
                            <Text style={styles.tierSub}>{TIER_CARDS.max.sublabel}</Text>
                            <View style={styles.priceRow}>
                                <Text style={styles.priceBig}>{liveYearlyPrice ?? `$${SUBSCRIPTION_PRICING.vip.price.toFixed(2)}`}</Text>
                                <Text style={styles.pricePeriod}>{TIER_CARDS.max.period}</Text>
                            </View>
                            <View style={styles.featuresBlock}>
                                {TIER_CARDS.max.features.map((f, i) => (
                                    <View key={i} style={styles.featureRow}>
                                        <Ionicons name="checkmark-circle" size={16} color={D.accent} />
                                        <Text style={styles.featureText}>{f}</Text>
                                    </View>
                                ))}
                            </View>
                            <PrimaryGradientButton
                                label={t('paywall.getMax')}
                                icon="diamond"
                                loading={isLoading === (liveYearlyProductId ?? TIER_CARDS.max.productId)}
                                onPress={() => purchase(liveYearlyProductId ?? TIER_CARDS.max.productId)}
                            />
                        </GlassPanel>
                    </PressableCard>
                </Animated.View>

                {/* Restore + Manage + Terms */}
                <View style={styles.actionsRow}>
                    <TouchableOpacity style={styles.restoreButton} onPress={handleRestore}>
                        {isLoading === 'restore' ? (
                            <ActivityIndicator color={D.accent} size="small" />
                        ) : (
                            <Text style={styles.restoreText}>{t('paywall.restorePurchases')}</Text>
                        )}
                    </TouchableOpacity>
                    {subscriptionTier !== 'free' && (
                        <TouchableOpacity style={styles.manageButton} onPress={() => iapService.manageSubscriptions()}>
                            <Ionicons name="settings-outline" size={16} color={D.accent} />
                            <Text style={styles.manageText}>{t('paywall.manageSubscription')}</Text>
                        </TouchableOpacity>
                    )}
                </View>

                <Text style={styles.termsText}>
                    {t('paywall.termsText', {
                        price: livePrice ?? `$${SUBSCRIPTION_PRICING.premium.price.toFixed(2)}`,
                        yearlyPrice: liveYearlyPrice ?? `$${SUBSCRIPTION_PRICING.vip.price.toFixed(2)}`,
                    })}
                </Text>

                <View style={styles.legalLinks}>
                    <TouchableOpacity
                        onPress={() => {
                            // Open hosted Privacy Policy URL (functional link required by Apple)
                            Linking.openURL(PRIVACY_POLICY_URL).catch(() =>
                                navigation.navigate('PrivacyPolicy')
                            );
                        }}
                    >
                        <Text style={styles.legalLink}>{t('paywall.privacyPolicy')}</Text>
                    </TouchableOpacity>
                    <Text style={styles.legalSeparator}>|</Text>
                    <TouchableOpacity
                        onPress={() => {
                            // Open Apple standard EULA URL — required by Guideline 3.1.2(c).
                            // This is the functional link Apple checks in the purchase flow.
                            Linking.openURL(APPLE_EULA_URL).catch(() =>
                                navigation.navigate('TermsOfService')
                            );
                        }}
                    >
                        <Text style={styles.legalLink}>{t('paywall.termsOfUse')}</Text>
                    </TouchableOpacity>
                </View>
            </Animated.ScrollView>
        </View>
    );
};

const createStyles = (D: DTokens) =>
    StyleSheet.create({
        // Orbs
        orb: { position: 'absolute', borderRadius: 999 },
        orbPrimary: {
            width: 340,
            height: 340,
            top: -60,
            left: -100,
            backgroundColor: D.orbPrimary,
        },
        orbSecondary: {
            width: 280,
            height: 280,
            top: 120,
            right: -80,
            backgroundColor: D.orbSecondary,
        },
        orbWarm: {
            width: 220,
            height: 220,
            bottom: 180,
            left: -40,
            backgroundColor: D.orbWarm,
        },

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
            backgroundColor: D.glassMuted,
            alignItems: 'center',
            justifyContent: 'center',
        },

        hero: { alignItems: 'center', marginBottom: 22 },
        iconContainer: {
            width: 76,
            height: 76,
            borderRadius: 38,
            backgroundColor: D.accentSoft,
            alignItems: 'center',
            justifyContent: 'center',
            marginBottom: 14,
        },
        heroTitle: {
            fontSize: 28,
            fontWeight: '800',
            color: D.text,
            marginBottom: 8,
            textAlign: 'center',
        },
        heroSubtitle: {
            fontSize: 14,
            color: D.textSub,
            textAlign: 'center',
            lineHeight: 20,
            paddingHorizontal: 20,
        },

        card: { marginBottom: 14, marginHorizontal: 20 },
        cardHeaderRow: {
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'space-between',
        },
        currentBadge: {
            backgroundColor: D.accentSoftStrong,
            paddingHorizontal: 10,
            paddingVertical: 4,
            borderRadius: 8,
        },
        currentBadgeText: {
            color: D.text,
            fontSize: 10,
            fontWeight: '700',
            letterSpacing: 0.5,
        },
        popularBadge: {
            position: 'absolute',
            top: 12,
            right: 12,
            backgroundColor: D.glassMuted,
            paddingHorizontal: 10,
            paddingVertical: 4,
            borderRadius: 10,
        },
        popularBadgeText: {
            color: D.text,
            fontSize: 10,
            fontWeight: '700',
            letterSpacing: 0.5,
        },
        tierName: {
            fontSize: 22,
            fontWeight: '800',
            color: D.text,
        },
        tierSub: {
            fontSize: 13,
            color: D.textSub,
            marginTop: 4,
            marginBottom: 8,
        },
        priceRow: {
            flexDirection: 'row',
            alignItems: 'flex-end',
            marginBottom: 4,
        },
        priceBig: {
            fontSize: 36,
            fontWeight: '800',
            color: D.text,
        },
        pricePeriod: {
            fontSize: 14,
            color: D.textSub,
            marginBottom: 6,
            marginLeft: 4,
        },

        featuresBlock: { marginTop: 12, marginBottom: 16 },
        featureRow: {
            flexDirection: 'row',
            alignItems: 'center',
            marginBottom: 8,
            gap: 10,
        },
        featureText: {
            fontSize: 14,
            color: D.text,
            fontWeight: '500',
        },
        featureMuted: { color: D.textMute },

        primaryButton: {
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 8,
            paddingVertical: 14,
            borderRadius: 12,
            marginTop: 6,
        },
        primaryButtonText: {
            fontSize: 15,
            fontWeight: '800',
            color: D.white,
        },

        savingsBadge: {
            position: 'absolute',
            top: 12,
            right: 12,
            backgroundColor: D.accent,
            paddingHorizontal: 10,
            paddingVertical: 4,
            borderRadius: 10,
        },
        savingsBadgeText: {
            color: D.bg,
            fontSize: 10,
            fontWeight: '700',
            letterSpacing: 0.5,
        },

        actionsRow: {
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 16,
            paddingVertical: 14,
        },
        restoreButton: { alignItems: 'center', paddingVertical: 14 },
        restoreText: {
            fontSize: 13,
            color: D.accent,
            fontWeight: '600',
        },
        manageButton: {
            flexDirection: 'row',
            alignItems: 'center',
            gap: 6,
        },
        manageText: {
            fontSize: 13,
            color: D.accent,
            fontWeight: '600',
        },

        termsText: {
            fontSize: 11,
            color: D.textSub,
            textAlign: 'center',
            lineHeight: 16,
            paddingHorizontal: 10,
            marginBottom: 20,
        },
        legalLinks: {
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 12,
            marginBottom: 20,
        },
        legalLink: {
            fontSize: 12,
            color: D.accent,
            fontWeight: '500',
        },
        legalSeparator: {
            fontSize: 12,
            color: D.textMute,
        },

        // Glass
        glassShadow: {
            ...Platform.select({
                ios: {
                    shadowColor: D.shadow,
                    shadowOffset: { width: 0, height: 8 },
                    shadowOpacity: 0.22,
                    shadowRadius: 16,
                },
                android: { elevation: 6 },
            }),
        },
        glassPanel: {
            overflow: 'hidden',
            borderWidth: 1,
            borderColor: D.glassBorder,
            backgroundColor: D.glass,
        },
        glassScrim: {
            ...StyleSheet.absoluteFillObject,
            backgroundColor: D.glassStrong,
        },
        glassContent: {
            padding: 20,
            position: 'relative',
            zIndex: 1,
        },
    });

export default PaywallScreen;
