import React, { useEffect, useMemo, useState } from 'react';
import { ActivityIndicator, Alert, Image, Linking, Platform, StatusBar, StyleSheet, TouchableOpacity, View } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';

import { ScaledText } from '../components/ui/ScaledText';
import { navigationRef } from '../navigation/navigationRef';
import { SUBSCRIPTION_PRICING } from '../store/subscriptionStore';
import useDailyUsageStore from '../store/dailyUsageStore';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import useWardrobeStore from '../store/wardrobeStore';
import { iapService } from '../src/services/iapService';

const APPLE_EULA_URL = 'https://www.apple.com/legal/internet-services/itunes/dev/stdeula/';
const PRIVACY_POLICY_URL = 'https://aiwardrobe.app/privacy';

type SelectedPlan = 'lite' | 'premium' | 'yearly';

type PlanCard = {
    id: SelectedPlan;
    title: string;
    price: string;
    period: string;
    badge?: string;
    oldPrice?: string;
    footer: string;
    bestFor: string;
    features: string[];
    accent: [string, string];
};

const PaywallScreen = () => {
    const insets = useSafeAreaInsets();
    const navigation = useNavigation<any>();
    const { completeOnboarding } = useStylePreferenceStore();
    const { t } = useTranslation();
    const [isLoading, setIsLoading] = useState<string | null>(null);
    const [selectedPlan, setSelectedPlan] = useState<SelectedPlan>('lite');
    const [liveLitePrice, setLiveLitePrice] = useState<string | null>(null);
    const [liveProPrice, setLiveProPrice] = useState<string | null>(null);
    const [liveYearlyPrice, setLiveYearlyPrice] = useState<string | null>(null);
    const [productsLoaded, setProductsLoaded] = useState(false);
    const [availableProductIds, setAvailableProductIds] = useState<string[]>([]);

    const wardrobeItemCount = useWardrobeStore((s) => s.items.length);
    const canGoBack = navigation.canGoBack();

    useEffect(() => {
        let cancelled = false;
        iapService.getProducts().then((products) => {
            if (cancelled) return;
            setAvailableProductIds(products.map((p) => p.id));
            setLiveLitePrice(products.find((p) => p.id === SUBSCRIPTION_PRICING.lite.productId)?.price ?? null);
            setLiveProPrice(products.find((p) => p.id === SUBSCRIPTION_PRICING.premium.productId)?.price ?? null);
            setLiveYearlyPrice(products.find((p) => p.id === SUBSCRIPTION_PRICING.vip.productId)?.price ?? null);
            setProductsLoaded(true);
        }).catch((err) => {
            console.warn('[Paywall] getProducts failed: ', err);
            if (!cancelled) {
                setAvailableProductIds([]);
                setProductsLoaded(true);
            }
        });
        return () => { cancelled = true; };
    }, []);

    const litePrice = liveLitePrice ?? '$2.99';
    const proPrice = liveProPrice ?? '$9.99';
    const yearlyPrice = liveYearlyPrice ?? '$99.99';

    const plans = useMemo<PlanCard[]>(() => [
        {
            id: 'lite',
            title: t('paywall.lite', 'Lite'),
            price: litePrice,
            period: '/ month',
            footer: '$0.69 / week',
            bestFor: 'More closet space, smarter outfit planning, and simple wardrobe analytics.',
            features: ['Unlimited AI outfit ideas', '100 wardrobe items', 'Wardrobe insights', 'Full outfit calendar'],
            accent: ['#73E2A7', '#64B5F6'],
        },
        {
            id: 'premium',
            title: t('paywall.pro', 'Pro'),
            price: proPrice,
            period: '/ month',
            badge: t('paywall.mostPopular', 'MOST POPULAR'),
            footer: '$2.31 / week',
            bestFor: 'Best for daily styling, travel looks, unlimited closet growth, and priority AI.',
            features: ['Unlimited AI outfit ideas', 'Unlimited wardrobe', 'AI trip planner', 'Priority model and no ads'],
            accent: ['#FFFFFF', '#A7F3FF'],
        },
        {
            id: 'yearly',
            title: t('paywall.yearlyPlan', 'Max Yearly'),
            price: yearlyPrice,
            period: '/ year',
            badge: 'SAVE 17%',
            oldPrice: '$119.88',
            footer: '$1.92 / week',
            bestFor: 'Everything in Pro with the lowest long-term price.',
            features: ['All Pro features', 'Annual savings', 'Seasonal collections', 'Early feature access'],
            accent: ['#FFD36A', '#FF7A90'],
        },
    ], [litePrice, proPrice, yearlyPrice, t]);

    const selected = plans.find((plan) => plan.id === selectedPlan) ?? plans[0];

    const getProductId = (plan: SelectedPlan): string => {
        switch (plan) {
            case 'lite': return SUBSCRIPTION_PRICING.lite.productId;
            case 'premium': return SUBSCRIPTION_PRICING.premium.productId;
            case 'yearly': return SUBSCRIPTION_PRICING.vip.productId;
        }
    };

    const resetToActivation = () => {
        const target =
            wardrobeItemCount >= 3
                ? { name: 'AIOutfit' as const, params: { source: 'wardrobe' as const } }
                : { name: 'ScanWardrobe' as const };

        setTimeout(() => {
            if (navigationRef.isReady()) {
                navigation.reset({ index: 1, routes: [{ name: 'Main' }, target] });
            } else if (canGoBack) {
                navigation.goBack();
            }
        }, 300);
    };

    const purchase = async () => {
        const productId = getProductId(selectedPlan);
        const isAvailable = !productsLoaded || availableProductIds.includes(productId);
        if (!isAvailable) {
            Alert.alert(
                t('paywall.subscriptionUnavailable'),
                `App Store is not returning ${productId}. Check that this product exists in App Store Connect, is Ready to Submit, is attached to the current app version, and is added to the RevenueCat offering.`
            );
            return;
        }

        setIsLoading(productId);
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        try {
            const result = await iapService.purchase(productId);
            if (result.success) {
                void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                resetToActivation();
            } else if (result.error) {
                void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
                Alert.alert(t('paywall.purchaseFailed'), result.error);
            }
        } catch (error) {
            void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
            Alert.alert(t('paywall.purchaseFailed'), error instanceof Error ? error.message : t('paywall.somethingWrongPurchase'));
        } finally {
            setIsLoading(null);
        }
    };

    const handleRestore = async () => {
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setIsLoading('restore');
        try {
            const result = await iapService.restorePurchases();
            if (result.success) {
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                resetToActivation();
            } else {
                Alert.alert(t('paywall.restoreFailed'), result.error || t('paywall.noPreviousPurchases'));
            }
        } catch (error) {
            Alert.alert(t('paywall.restoreFailed'), error instanceof Error ? error.message : t('paywall.somethingWrongRestore'));
        } finally {
            setIsLoading(null);
        }
    };

    const closePaywall = () => {
        if (canGoBack) navigation.goBack();
        else completeOnboarding();
    };

    return (
        <View style={styles.screen}>
            <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
            {/* TEMP: paywall_hero.png was never committed to the repo (pre-existing, unrelated to reorg) — swapped to an existing asset so bundling succeeds. Revert to paywall_hero.png once the real asset is added. */}
            <Image source={require('../assets/images/basic_outfit_1.png')} style={styles.backgroundImage} resizeMode="cover" />
            <LinearGradient
                colors={['rgba(5,10,22,0.15)', 'rgba(5,10,22,0.72)', '#050A16', '#050A16']}
                locations={[0, 0.42, 0.68, 1]}
                style={StyleSheet.absoluteFillObject}
            />

            <TouchableOpacity style={[styles.closeButton, { top: insets.top + 10 }]} onPress={closePaywall} activeOpacity={0.75}>
                <Ionicons name="close" size={22} color="#FFFFFF" />
            </TouchableOpacity>

            <Animated.ScrollView
                contentContainerStyle={[styles.scrollContent, { paddingTop: insets.top + 196, paddingBottom: insets.bottom + 28 }]}
                showsVerticalScrollIndicator={false}
            >
                <Animated.View entering={FadeIn.duration(500)} style={styles.hero}>
                    <ScaledText style={styles.title}>Style more from the closet you already own</ScaledText>
                    <ScaledText style={styles.subtitle}>
                        Pick the plan that matches how you use AIWardrobe. Lite is simple organization power; Pro unlocks deeper planning and priority AI.
                    </ScaledText>
                </Animated.View>

                <Animated.View entering={FadeInUp.delay(140).duration(560)} style={styles.planGrid}>
                    {plans.map((plan) => {
                        const isSelected = selectedPlan === plan.id;
                        const productId = getProductId(plan.id);
                        const isAvailable = !productsLoaded || availableProductIds.includes(productId);
                        return (
                            <TouchableOpacity
                                key={plan.id}
                                activeOpacity={0.9}
                                onPress={() => {
                                    setSelectedPlan(plan.id);
                                    void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                }}
                                style={[
                                    styles.planCard,
                                    isSelected && styles.planCardSelected,
                                    !isAvailable && styles.planCardUnavailable,
                                ]}
                            >
                                {plan.badge && (
                                    <View style={[styles.inlineBadge, isSelected && styles.inlineBadgeSelected]}>
                                        <ScaledText style={[styles.inlineBadgeText, isSelected && styles.inlineBadgeTextSelected]}>
                                            {plan.badge}
                                        </ScaledText>
                                    </View>
                                )}
                                {!isAvailable && (
                                    <View style={styles.inlineBadge}>
                                        <ScaledText style={styles.inlineBadgeText}>UNAVAILABLE</ScaledText>
                                    </View>
                                )}

                                <View>
                                    <ScaledText style={[styles.planTitle, isSelected && styles.planTitleSelected]} numberOfLines={1}>
                                        {plan.title}
                                    </ScaledText>
                                    <ScaledText style={[styles.price, isSelected && styles.priceSelected]} numberOfLines={1} adjustsFontSizeToFit>
                                        {plan.price}
                                    </ScaledText>
                                    {plan.oldPrice && (
                                        <ScaledText style={styles.oldPrice} numberOfLines={1}>
                                            {plan.oldPrice}
                                        </ScaledText>
                                    )}
                                </View>

                                <View style={styles.cardBottom}>
                                    <ScaledText style={[styles.period, isSelected && styles.periodSelected]}>{plan.period}</ScaledText>
                                    <ScaledText style={styles.cardFooter}>{plan.footer}</ScaledText>
                                </View>
                            </TouchableOpacity>
                        );
                    })}
                </Animated.View>

                <Animated.View entering={FadeInUp.delay(180).duration(560)} style={styles.selectedDetails}>
                    <ScaledText style={styles.selectedDetailTitle}>{selected.bestFor}</ScaledText>
                    <View style={styles.featureGrid}>
                        {selected.features.map((feature) => (
                            <View key={feature} style={styles.featureChip}>
                                <Ionicons name="checkmark" size={13} color="#73E2A7" />
                                <ScaledText style={styles.featureText}>{feature}</ScaledText>
                            </View>
                        ))}
                    </View>
                </Animated.View>

                <Animated.View entering={FadeInUp.delay(220).duration(560)} style={styles.ctaPanel}>
                    <View>
                        <ScaledText style={styles.ctaEyebrow}>Selected plan</ScaledText>
                        <ScaledText style={styles.ctaPlan}>{selected.title} · {selected.price}</ScaledText>
                    </View>
                    <TouchableOpacity activeOpacity={0.88} disabled={isLoading !== null} onPress={purchase}>
                        <LinearGradient colors={['#FFFFFF', '#DCEBFF']} start={{ x: 0, y: 0 }} end={{ x: 1, y: 1 }} style={styles.continueButton}>
                            {isLoading === getProductId(selectedPlan) ? (
                                <ActivityIndicator color="#050A16" size="small" />
                            ) : (
                                <>
                                    <ScaledText style={styles.continueButtonText}>Continue</ScaledText>
                                    <Ionicons name="arrow-forward" size={18} color="#050A16" />
                                </>
                            )}
                        </LinearGradient>
                    </TouchableOpacity>
                </Animated.View>

                <View style={styles.footerActionsBlock}>
                    <TouchableOpacity style={styles.footerLinkTouch} onPress={handleRestore}>
                        {isLoading === 'restore' ? (
                            <ActivityIndicator color="#FFFFFF" size="small" />
                        ) : (
                            <ScaledText style={styles.footerActionLinkText}>{t('paywall.restorePurchases')}</ScaledText>
                        )}
                    </TouchableOpacity>
                </View>

                <ScaledText style={styles.legalNotesText}>{t('paywall.termsText')}</ScaledText>

                <View style={styles.legalLinksBlock}>
                    <TouchableOpacity onPress={() => Linking.openURL(PRIVACY_POLICY_URL).catch(() => navigation.navigate('PrivacyPolicy'))}>
                        <ScaledText style={styles.legalLink}>{t('paywall.privacyPolicy')}</ScaledText>
                    </TouchableOpacity>
                    <ScaledText style={styles.legalDivider}>|</ScaledText>
                    <TouchableOpacity onPress={() => Linking.openURL(APPLE_EULA_URL).catch(() => navigation.navigate('TermsOfService'))}>
                        <ScaledText style={styles.legalLink}>{t('paywall.termsOfUse')}</ScaledText>
                    </TouchableOpacity>
                </View>
            </Animated.ScrollView>
        </View>
    );
};

const styles = StyleSheet.create({
    screen: {
        flex: 1,
        backgroundColor: '#050A16',
    },
    backgroundImage: {
        position: 'absolute',
        top: -90,
        left: 0,
        width: '100%',
        height: '58%',
    },
    closeButton: {
        position: 'absolute',
        right: 18,
        width: 38,
        height: 38,
        borderRadius: 19,
        backgroundColor: 'rgba(255,255,255,0.14)',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.18)',
    },
    scrollContent: {
        paddingHorizontal: 16,
    },
    hero: {
        marginBottom: 18,
    },
    title: {
        color: '#FFFFFF',
        fontSize: 34,
        lineHeight: 38,
        fontWeight: '900',
        letterSpacing: 0,
    },
    subtitle: {
        color: 'rgba(255,255,255,0.76)',
        fontSize: 15,
        lineHeight: 22,
        marginTop: 10,
        fontWeight: '600',
    },
    planGrid: {
        flexDirection: 'row',
        gap: 8,
        marginTop: 4,
    },
    planCard: {
        flex: 1,
        minHeight: 174,
        borderRadius: 28,
        borderWidth: 1.5,
        borderColor: 'rgba(255,255,255,0.18)',
        paddingHorizontal: 10,
        paddingTop: 14,
        paddingBottom: 14,
        backgroundColor: 'rgba(255,255,255,0.08)',
        justifyContent: 'space-between',
        position: 'relative',
        ...Platform.select({
            ios: {
                shadowColor: '#000000',
                shadowOffset: { width: 0, height: 10 },
                shadowOpacity: 0.18,
                shadowRadius: 18,
            },
            android: { elevation: 5 },
        }),
    },
    planCardSelected: {
        borderColor: 'rgba(255,255,255,0.72)',
        borderWidth: 2,
        backgroundColor: 'rgba(255,255,255,0.15)',
    },
    planCardUnavailable: {
        opacity: 0.48,
    },
    planTitle: {
        color: 'rgba(255,255,255,0.72)',
        fontSize: 16,
        fontWeight: '800',
        letterSpacing: 0,
    },
    price: {
        color: '#FFFFFF',
        fontSize: 24,
        fontWeight: '900',
        letterSpacing: 0,
        marginTop: 10,
    },
    planTitleSelected: {
        color: '#FFFFFF',
    },
    priceSelected: {
        color: '#FFFFFF',
    },
    oldPrice: {
        color: 'rgba(255,255,255,0.50)',
        fontSize: 13,
        fontWeight: '700',
        textDecorationLine: 'line-through',
        marginTop: 8,
    },
    inlineBadge: {
        alignSelf: 'flex-start',
        borderRadius: 8,
        paddingHorizontal: 7,
        paddingVertical: 5,
        backgroundColor: 'rgba(255,255,255,0.12)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.18)',
        marginBottom: 8,
    },
    inlineBadgeSelected: {
        backgroundColor: 'rgba(255,255,255,0.22)',
        borderColor: 'rgba(255,255,255,0.34)',
    },
    inlineBadgeText: {
        color: 'rgba(255,255,255,0.72)',
        fontSize: 9,
        fontWeight: '900',
        textAlign: 'center',
    },
    inlineBadgeTextSelected: {
        color: '#FFFFFF',
    },
    cardBottom: {
        gap: 4,
    },
    period: {
        color: 'rgba(255,255,255,0.62)',
        fontSize: 11,
        fontWeight: '800',
    },
    periodSelected: {
        color: 'rgba(255,255,255,0.80)',
    },
    cardFooter: {
        color: 'rgba(255,255,255,0.50)',
        fontSize: 11,
        fontWeight: '700',
    },
    selectedDetails: {
        marginTop: 12,
        borderRadius: 14,
        padding: 14,
        backgroundColor: 'rgba(255,255,255,0.08)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.12)',
    },
    selectedDetailTitle: {
        color: 'rgba(255,255,255,0.82)',
        fontSize: 13,
        lineHeight: 18,
        fontWeight: '700',
    },
    featureGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 7,
        marginTop: 14,
    },
    featureChip: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 5,
        maxWidth: '100%',
        borderRadius: 8,
        paddingHorizontal: 8,
        paddingVertical: 6,
        backgroundColor: 'rgba(255,255,255,0.10)',
    },
    featureText: {
        color: 'rgba(255,255,255,0.90)',
        fontSize: 11,
        fontWeight: '700',
    },
    ctaPanel: {
        marginTop: 16,
        borderRadius: 12,
        padding: 14,
        backgroundColor: 'rgba(255,255,255,0.10)',
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.14)',
        gap: 12,
    },
    ctaEyebrow: {
        color: 'rgba(255,255,255,0.58)',
        fontSize: 11,
        fontWeight: '800',
        textTransform: 'uppercase',
    },
    ctaPlan: {
        color: '#FFFFFF',
        fontSize: 18,
        fontWeight: '900',
        marginTop: 2,
    },
    continueButton: {
        height: 54,
        borderRadius: 10,
        alignItems: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
        gap: 8,
    },
    continueButtonText: {
        color: '#050A16',
        fontSize: 16,
        fontWeight: '900',
    },
    footerActionsBlock: {
        alignItems: 'center',
        paddingVertical: 14,
    },
    footerLinkTouch: {
        paddingVertical: 6,
        paddingHorizontal: 10,
    },
    footerActionLinkText: {
        fontSize: 13,
        color: '#FFFFFF',
        fontWeight: '800',
    },
    legalNotesText: {
        fontSize: 10,
        color: 'rgba(255,255,255,0.48)',
        textAlign: 'center',
        lineHeight: 15,
        paddingHorizontal: 8,
    },
    legalLinksBlock: {
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        gap: 12,
        marginTop: 14,
    },
    legalLink: {
        fontSize: 12,
        color: '#FFFFFF',
        fontWeight: '700',
        textDecorationLine: 'underline',
    },
    legalDivider: {
        fontSize: 12,
        color: 'rgba(255,255,255,0.20)',
    },
});

export default PaywallScreen;
