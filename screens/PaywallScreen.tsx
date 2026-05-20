/**
 * PaywallScreen — Iridescent Frosted Glass Full-Screen Background Upgrade Surface.
 * Fully customized using the user-provided 'Italian Coast Old Money' Speed Boat image.
 * Implements a full-bleed parallax background with subtle dark contrast mask overlay.
 * Floating content sits inside an iridescent frosted-glass card (glassmorphism) that
 * blends BlurView and pastel shimmers for state-of-the-art premium visual aesthetics.
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
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
    Image,
} from 'react-native';

// Apple Standard EULA — required by Guideline 3.1.2(c)
const APPLE_EULA_URL = 'https://www.apple.com/legal/internet-services/itunes/dev/stdeula/';
const PRIVACY_POLICY_URL = 'https://aiwardrobe.app/privacy'; // Privacy Policy URL

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
import useWardrobeStore from '../store/wardrobeStore';
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
        text: colors.text.primary,
        textSub: colors.text.secondary,
        textMute: colors.text.muted,
        white: '#FFFFFF',
        brandAccent: '#7B61FF',
        brandAccentSoft: 'rgba(123, 97, 255, 0.16)',
        brandAccentSoftStrong: 'rgba(123, 97, 255, 0.28)',
        glassBorder: 'rgba(255, 255, 255, 0.20)',
        glassBorderSoft: 'rgba(255, 255, 255, 0.10)',
    };
}

type DTokens = ReturnType<typeof useDesignTokens>;

const PaywallScreen = () => {
    const D = useDesignTokens();
    const styles = useMemo(() => createStyles(D), [D]);
    const insets = useSafeAreaInsets();
    const navigation = useNavigation<any>();
    const scrollViewRef = useRef<any>(null);
    const { completeOnboarding } = useStylePreferenceStore();
    const { t } = useTranslation();
    const [isLoading, setIsLoading] = useState<string | null>(null);

    // Offer Code States
    const [isRedeemingOffer, setIsRedeemingOffer] = useState(false);

    // RevenueCat Live Pricing
    const [livePrice, setLivePrice] = useState<string | null>(null);
    const [liveProductId, setLiveProductId] = useState<string | null>(null);
    const [liveYearlyPrice, setLiveYearlyPrice] = useState<string | null>(null);
    const [liveYearlyProductId, setLiveYearlyProductId] = useState<string | null>(null);

    // Active Subscription & Wardrobe length
    const subscriptionTier = useSubscriptionStore((s) => s.tier);
    const wardrobeItemCount = useWardrobeStore((s) => s.items.length);
    const canGoBack = navigation.canGoBack();

    // 3-Card Select State: default selected is Weekly
    const [selectedProductId, setSelectedProductId] = useState<string>('com.aiwardrobe.premium.weekly');

    const scrollY = useSharedValue(0);
    const scrollHandler = useAnimatedScrollHandler((event) => {
        scrollY.value = event.contentOffset.y;
    });

    // Premium elastic overscroll zoom effect on the full-screen background image
    const bgImageAnimStyle = useAnimatedStyle(() => ({
        transform: [
            {
                scale: interpolate(
                    scrollY.value,
                    [-150, 0],
                    [1.20, 1],
                    Extrapolation.CLAMP
                ),
            },
        ],
    }));

    const triggerLightHaptic = () => {
        void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    };

    const triggerSuccessHaptic = () => {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    };

    const triggerErrorHaptic = () => {
        void Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
    };

    // Load available products from RevenueCat
    useEffect(() => {
        let cancelled = false;
        iapService.getProducts().then((products) => {
            if (cancelled) return;
            const proProduct = products.find((p) => p.id === 'com.aiwardrobe.premium.monthly') || products.find((p) => {
                const id = String(p.id || '').toLowerCase();
                const title = String(p.title || '').toLowerCase();
                return id.includes('premium') || id.includes('pro') || title.includes('premium') || title.includes('pro');
            });
            if (proProduct?.price) {
                setLivePrice(proProduct.price);
            }
            if (proProduct?.id) {
                setLiveProductId(proProduct.id);
            }

            const yearlyProduct = products.find((p) => p.id === 'com.aiwardrobe.premium.yearly') || products.find((p) => {
                const id = String(p.id || '').toLowerCase();
                return id.includes('yearly') || id.includes('annual');
            });
            if (yearlyProduct?.price) {
                setLiveYearlyPrice(yearlyProduct.price);
            }
            if (yearlyProduct?.id) {
                setLiveYearlyProductId(yearlyProduct.id);
            }
        }).catch((err) => {
            console.warn('[Paywall] getProducts failed: ', err);
        });
        return () => { cancelled = true; };
    }, []);

    // Navigate users back to closet scan or outfit creator
    const resetToActivation = () => {
        const target =
            wardrobeItemCount >= 3
                ? { name: 'AIOutfit' as const, params: { source: 'wardrobe' as const } }
                : { name: 'ScanWardrobe' as const };

        setTimeout(() => {
            if (navigationRef.isReady()) {
                navigation.reset({
                    index: 1,
                    routes: [
                        { name: 'Main' },
                        target,
                    ],
                });
            } else if (canGoBack) {
                navigation.goBack();
            }
        }, 300);
    };

    // Execute in-app purchase flow
    const purchase = async (productId: string) => {
        console.log('[Paywall] Purchase initialized for Product ID:', productId);
        setIsLoading(productId);
        triggerSuccessHaptic();
        try {
            const result = await iapService.purchase(productId);
            if (result.success) {
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                resetToActivation();
            } else if (result.error) {
                triggerErrorHaptic();
                Alert.alert(t('paywall.purchaseFailed'), result.error);
            }
        } catch (error) {
            triggerErrorHaptic();
            Alert.alert(
                t('paywall.purchaseFailed'),
                error instanceof Error ? error.message : t('paywall.somethingWrongPurchase')
            );
        } finally {
            setIsLoading(null);
        }
    };

    // Restore premium purchase history
    const handleRestore = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setIsLoading('restore');
        try {
            const result = await iapService.restorePurchases();
            if (result.success) {
                triggerSuccessHaptic();
                completeOnboarding();
                await useDailyUsageStore.getState().resetToday();
                resetToActivation();
            } else {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
                Alert.alert(t('paywall.restoreFailed'), result.error || t('paywall.noPreviousPurchases'));
            }
        } catch (error) {
            triggerErrorHaptic();
            Alert.alert(
                t('paywall.restoreFailed'),
                error instanceof Error ? error.message : t('paywall.somethingWrongRestore')
            );
        } finally {
            setIsLoading(null);
        }
    };

    // Close paywall and resume free plan path
    const closePaywall = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        if (canGoBack) {
            navigation.goBack();
        } else {
            completeOnboarding();
        }
    };

    // Present Apple's native Offer Code redemption sheet
    const handleOfferCode = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setIsRedeemingOffer(true);
        try {
            await iapService.presentCodeRedemptionSheet();
            // After the sheet dismisses, check subscription status from state
            const { hasActiveSubscription } = useSubscriptionStore.getState();
            if (hasActiveSubscription) {
                triggerSuccessHaptic();
                completeOnboarding();
                resetToActivation();
            }
        } catch (error) {
            console.warn('[Paywall] Apple offer code presentation error:', error);
            triggerErrorHaptic();
        } finally {
            setIsRedeemingOffer(false);
        }
    };

    // Features array checklist mapping
    const rawFeaturesList = t('paywall.featuresList', { returnObjects: true });
    const featuresList: string[] = Array.isArray(rawFeaturesList)
        ? rawFeaturesList
        : [
            "Примеряйте любой образ на аватар",
            "Находите одежду по любому фото",
            "Создавайте образы из гардероба",
            "Организуйте свой гардероб",
            "Найдите свои лучшие цвета",
            "Превращайте фото одежды в студийные снимки"
        ];

    // Horizontal cards schema details
    const CARDS = [
        {
            id: 'com.aiwardrobe.premium.weekly',
            title: t('paywall.weekly', 'Неделя'),
            price: 'US$4.99',
            footer: t('paywall.weeklySubtitle', 'US$4.99 / нед.'),
            hasBadge: false,
            badgeText: '',
            strikethrough: null,
        },
        {
            id: liveProductId || 'com.aiwardrobe.premium.monthly',
            title: t('paywall.monthly', 'Месяц'),
            price: livePrice ? livePrice : 'US$14.99',
            footer: t('paywall.monthlySubtitle', 'US$3.44 / нед.'),
            hasBadge: true,
            badgeText: t('paywall.discount29', 'Скидка 29%'),
            strikethrough: 'US$21.41',
        },
        {
            id: liveYearlyProductId || 'com.aiwardrobe.premium.yearly',
            title: t('paywall.annually', 'Год'),
            price: liveYearlyPrice ? liveYearlyPrice : 'US$79.99',
            footer: t('paywall.annualSubtitle', 'US$1.53 / нед.'),
            hasBadge: true,
            badgeText: t('paywall.discount69', 'Скидка 69%'),
            strikethrough: 'US$260.18',
        },
    ];

    return (
        <View style={{ flex: 1, backgroundColor: '#0A1931' }}>
            <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />

            {/* Full-Screen Background Image stretched exactly 100% height & width */}
            <Image
                source={require('../assets/images/paywall_hero.png')}
                style={styles.backgroundImage}
                resizeMode="cover"
            />

            {/* Dark translucent LinearGradient mask overlay that fades the photo bottom edge seamlessly into solid dark blue */}
            <LinearGradient
                colors={[
                    'rgba(10, 25, 49, 0.10)',
                    'rgba(10, 25, 49, 0.35)',
                    'rgba(10, 25, 49, 0.88)',
                    '#0A1931',
                    '#0A1931'
                ]}
                locations={[0.0, 0.35, 0.65, 0.78, 1.0]}
                style={StyleSheet.absoluteFillObject}
            />

            {/* Scrollable Container with floating translucent glass card panel */}
            <Animated.ScrollView
                ref={scrollViewRef}
                onScroll={scrollHandler}
                scrollEventThrottle={16}
                contentContainerStyle={{
                    paddingTop: insets.top + 240,
                    paddingBottom: insets.bottom + 32,
                    paddingHorizontal: 16,
                }}
                showsVerticalScrollIndicator={false}
            >
                {/* Iridescent Frosted Glass Card Panel */}
                <View style={styles.glassCardWrapper}>
                    <BlurView
                        intensity={Platform.OS === 'ios' ? 45 : 100}
                        tint="dark"
                        style={StyleSheet.absoluteFillObject}
                    />

                    {/* Iridescent Pastel Shimmer Shaded Gradient Mask */}
                    <LinearGradient
                        colors={['rgba(123, 97, 255, 0.14)', 'rgba(168, 192, 218, 0.08)']}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={StyleSheet.absoluteFillObject}
                    />

                    {/* Inner Content of the Glass Card */}
                    <View style={styles.glassCardInnerContent}>
                        {/* Title */}
                        <Animated.View entering={FadeIn.duration(600)} style={styles.titleSection}>
                            <Text style={styles.paywallTitle}>{t('paywall.openPro', 'Открыть Pro')}</Text>
                        </Animated.View>

                        {/* 6 premium benefit checklist rows */}
                        <Animated.View entering={FadeInUp.delay(100).duration(600)} style={styles.checklistBlock}>
                            {featuresList.map((item, idx) => (
                                <View key={idx} style={styles.checkRow}>
                                    <View style={styles.checkCircle}>
                                        <Ionicons name="checkmark" size={12} color="#FFFFFF" />
                                    </View>
                                    <Text style={styles.checkText}>{item}</Text>
                                </View>
                            ))}
                        </Animated.View>


                        {/* Horizontal 3-card grid subscription switcher */}
                        <Animated.View entering={FadeInUp.delay(150).duration(600)} style={styles.cardsGridRow}>
                            {CARDS.map((card) => {
                                const isSelected = selectedProductId === card.id;
                                return (
                                    <TouchableOpacity
                                        key={card.id}
                                        style={[
                                            styles.planCard,
                                            isSelected && styles.planCardSelected,
                                            !isSelected && styles.planCardUnselected,
                                        ]}
                                        activeOpacity={0.9}
                                        onPress={() => {
                                            triggerLightHaptic();
                                            setSelectedProductId(card.id);
                                            if (card.id !== 'com.aiwardrobe.premium.weekly') {
                                                setTimeout(() => {
                                                    scrollViewRef.current?.scrollToEnd({ animated: true });
                                                }, 100);
                                            }
                                        }}
                                    >
                                        {/* Discount badge on top card borders */}
                                        {card.hasBadge && (
                                            <View style={styles.cardDiscountBadge}>
                                                <Text style={styles.cardDiscountBadgeText}>{card.badgeText}</Text>
                                            </View>
                                        )}

                                        {/* Selected Indicator dot */}
                                        <View style={styles.cardSelectionRow}>
                                            <View style={[
                                                styles.selectionDotOuter,
                                                isSelected && styles.selectionDotOuterActive
                                            ]}>
                                                {isSelected && <View style={styles.selectionDotInner} />}
                                            </View>
                                        </View>

                                        {/* Duration Title */}
                                        <Text style={[
                                            styles.cardTitle,
                                            isSelected ? styles.cardTextHighlight : styles.cardTextMuted
                                        ]}>
                                            {card.title}
                                        </Text>

                                        {/* Price Tags */}
                                        <View style={styles.cardPriceBlock}>
                                            {card.strikethrough && (
                                                <Text style={styles.cardOriginalPrice}>
                                                    {card.strikethrough}
                                                </Text>
                                            )}
                                            <Text style={[
                                                styles.cardCurrentPrice,
                                                isSelected ? styles.cardTextHighlight : styles.cardTextNormal
                                            ]}>
                                                {card.price}
                                            </Text>
                                        </View>

                                        {/* Weekly Breakdown footer rate */}
                                        <View style={styles.cardFooter}>
                                            <Text style={styles.cardFooterText} numberOfLines={2}>
                                                {card.footer}
                                            </Text>
                                        </View>
                                    </TouchableOpacity>
                                );
                            })}
                        </Animated.View>

                        {/* Promo Code Input Wrapper -> Replaced with Apple Offer Code Button */}
                        <Animated.View entering={FadeInUp.delay(180).duration(600)} style={styles.promoInputCardWrapper}>
                            {/* Accent Subtitle Header */}
                            <Text style={styles.promoHeaderTitle}>
                                {t('paywall.activateCodeHeader', 'У вас есть промокод?')}
                            </Text>

                            <TouchableOpacity
                                style={styles.activatePromoButton}
                                activeOpacity={0.8}
                                disabled={isRedeemingOffer}
                                onPress={handleOfferCode}
                            >
                                <LinearGradient
                                    colors={['#7B61FF', '#6366F1']}
                                    start={{ x: 0, y: 0 }}
                                    end={{ x: 1, y: 1 }}
                                    style={styles.activatePromoGradient}
                                >
                                    {isRedeemingOffer ? (
                                        <ActivityIndicator color="#FFFFFF" size="small" />
                                    ) : (
                                        <>
                                            <Ionicons name="ticket-outline" size={20} color="#FFFFFF" style={{ marginRight: 8 }} />
                                            <Text style={styles.activatePromoButtonText}>
                                                {t('promo.redeemOfferCode', 'Активировать промокод (Apple)')}
                                            </Text>
                                        </>
                                    )}
                                </LinearGradient>
                            </TouchableOpacity>
                        </Animated.View>

                        {/* Main Action Pill button */}
                        <Animated.View entering={FadeInUp.delay(200).duration(600)} style={styles.buttonWrapper}>
                            <TouchableOpacity
                                activeOpacity={0.88}
                                disabled={isLoading !== null}
                                onPress={() => purchase(selectedProductId)}
                            >
                                <LinearGradient
                                    colors={['#FFFFFF', '#E6EEFF']}
                                    start={{ x: 0, y: 0 }}
                                    end={{ x: 1, y: 1 }}
                                    style={styles.continueButton}
                                >
                                    {isLoading === selectedProductId ? (
                                        <ActivityIndicator color="#0A1931" size="small" />
                                    ) : (
                                        <Text style={styles.continueButtonText}>
                                            {t('styleQuiz.continue', 'Продолжить')}
                                        </Text>
                                    )}
                                </LinearGradient>
                            </TouchableOpacity>
                        </Animated.View>

                        {/* Footer legal actions row (Restore purchases only) */}
                        <View style={styles.footerActionsBlock}>
                            <TouchableOpacity style={styles.footerLinkTouch} onPress={handleRestore}>
                                {isLoading === 'restore' ? (
                                    <ActivityIndicator color="#FFFFFF" size="small" />
                                ) : (
                                    <Text style={styles.footerActionLinkText}>
                                        {t('paywall.restorePurchases', 'Восстановить покупки')}
                                    </Text>
                                )}
                            </TouchableOpacity>
                        </View>

                        {/* Legal compliance descriptions */}
                        <Text style={styles.legalNotesText}>
                            {t('paywall.termsText', {
                                price: livePrice ?? `$${SUBSCRIPTION_PRICING.premium.price.toFixed(2)}`,
                                yearlyPrice: liveYearlyPrice ?? `$${SUBSCRIPTION_PRICING.vip.price.toFixed(2)}`,
                            })}
                        </Text>

                        {/* Legal standard linkages */}
                        <View style={styles.legalLinksBlock}>
                            <TouchableOpacity
                                onPress={() => {
                                    Linking.openURL(PRIVACY_POLICY_URL).catch(() =>
                                        navigation.navigate('PrivacyPolicy')
                                    );
                                }}
                            >
                                <Text style={styles.legalLink}>{t('paywall.privacyPolicy')}</Text>
                            </TouchableOpacity>
                            <Text style={styles.legalDivider}>|</Text>
                            <TouchableOpacity
                                onPress={() => {
                                    Linking.openURL(APPLE_EULA_URL).catch(() =>
                                        navigation.navigate('TermsOfService')
                                    );
                                }}
                            >
                                <Text style={styles.legalLink}>{t('paywall.termsOfUse')}</Text>
                            </TouchableOpacity>
                        </View>
                    </View>
                </View>
            </Animated.ScrollView>
        </View>
    );
};

const createStyles = (D: DTokens) =>
    StyleSheet.create({
        backgroundImage: {
            position: 'absolute',
            top: -120,
            left: 0,
            width: '100%',
            height: '82%',
            zIndex: 0,
        },
        closeButton: {
            position: 'absolute',
            left: 20,
            width: 38,
            height: 38,
            borderRadius: 19,
            overflow: 'hidden',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 10,
        },
        // Iridescent Glass Card styling (Frosted Glassmorphism)
        glassCardWrapper: {
            borderRadius: 24,
            borderWidth: 1.5,
            borderColor: 'rgba(255, 255, 255, 0.16)',
            overflow: 'hidden',
            backgroundColor: 'rgba(10, 25, 49, 0.40)',
            ...Platform.select({
                ios: {
                    shadowColor: '#000000',
                    shadowOffset: { width: 0, height: 12 },
                    shadowOpacity: 0.35,
                    shadowRadius: 20,
                },
                android: { elevation: 8 },
            }),
        },
        glassCardInnerContent: {
            paddingTop: 24,
            paddingHorizontal: 16,
            paddingBottom: 24,
            zIndex: 1,
        },
        titleSection: {
            alignItems: 'center',
            marginVertical: 14,
            paddingHorizontal: 10,
        },
        paywallTitle: {
            fontSize: 28,
            fontWeight: '900',
            color: '#FFFFFF',
            textAlign: 'center',
            letterSpacing: 0.5,
        },
        checklistBlock: {
            paddingHorizontal: 12,
            marginTop: 6,
            gap: 12,
        },
        checkRow: {
            flexDirection: 'row',
            alignItems: 'center',
            gap: 12,
        },
        checkCircle: {
            width: 22,
            height: 22,
            borderRadius: 11,
            backgroundColor: D.brandAccent,
            alignItems: 'center',
            justifyContent: 'center',
        },
        checkText: {
            fontSize: 14,
            fontWeight: '600',
            color: 'rgba(255, 255, 255, 0.90)',
            flex: 1,
            lineHeight: 19,
        },

        // 3-Card horizontal grid switcher styles (floating glass style)
        cardsGridRow: {
            flexDirection: 'row',
            justifyContent: 'space-between',
            marginTop: 29,
            marginBottom: 20,
            gap: 6,
        },
        planCard: {
            flex: 1,
            borderRadius: 16,
            borderWidth: 1.5,
            paddingVertical: 14,
            paddingHorizontal: 8,
            alignItems: 'center',
            justifyContent: 'space-between',
            position: 'relative',
            height: 146,
        },
        planCardSelected: {
            borderColor: D.brandAccent,
            backgroundColor: D.brandAccentSoft,
        },
        planCardUnselected: {
            borderColor: 'rgba(255, 255, 255, 0.12)',
            backgroundColor: 'rgba(255, 255, 255, 0.04)',
        },
        cardDiscountBadge: {
            position: 'absolute',
            bottom: -10,
            alignSelf: 'center',
            backgroundColor: D.brandAccent,
            paddingHorizontal: 7,
            paddingVertical: 3,
            borderRadius: 6,
            zIndex: 2,
        },
        cardDiscountBadgeText: {
            color: '#FFFFFF',
            fontSize: 8,
            fontWeight: '900',
            textAlign: 'center',
        },
        cardSelectionRow: {
            width: '100%',
            alignItems: 'flex-end',
            marginBottom: 2,
        },
        selectionDotOuter: {
            width: 14,
            height: 14,
            borderRadius: 7,
            borderWidth: 1.5,
            borderColor: 'rgba(255, 255, 255, 0.30)',
            alignItems: 'center',
            justifyContent: 'center',
        },
        selectionDotOuterActive: {
            borderColor: D.brandAccent,
        },
        selectionDotInner: {
            width: 8,
            height: 8,
            borderRadius: 4,
            backgroundColor: D.brandAccent,
        },
        cardTitle: {
            fontSize: 14,
            fontWeight: '800',
            textAlign: 'center',
            marginBottom: 2,
        },
        cardPriceBlock: {
            alignItems: 'center',
            marginVertical: 4,
        },
        cardOriginalPrice: {
            fontSize: 10,
            color: 'rgba(255, 255, 255, 0.40)',
            textDecorationLine: 'line-through',
            marginBottom: 2,
        },
        cardCurrentPrice: {
            fontSize: 16,
            fontWeight: '900',
            textAlign: 'center',
        },
        cardFooter: {
            borderTopWidth: 1,
            borderTopColor: 'rgba(255, 255, 255, 0.10)',
            paddingTop: 6,
            width: '100%',
            alignItems: 'center',
        },
        cardFooterText: {
            fontSize: 9,
            fontWeight: '700',
            color: 'rgba(255, 255, 255, 0.60)',
            textAlign: 'center',
            lineHeight: 11,
        },
        cardTextHighlight: {
            color: '#FFFFFF',
        },
        cardTextNormal: {
            color: 'rgba(255, 255, 255, 0.90)',
        },
        cardTextMuted: {
            color: 'rgba(255, 255, 255, 0.60)',
        },
        // Promo Input Styling
        promoInputCardWrapper: {
            marginHorizontal: 12,
            marginTop: 10,
            marginBottom: 16,
            gap: 6,
        },
        promoHeaderTitle: {
            fontSize: 11,
            fontWeight: '800',
            color: 'rgba(255, 255, 255, 0.50)',
            textTransform: 'uppercase',
            letterSpacing: 1,
            marginBottom: 2,
            paddingLeft: 4,
        },
        activatePromoButton: {
            borderRadius: 14,
            height: 52,
            overflow: 'hidden',
        },
        activatePromoGradient: {
            flex: 1,
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'center',
            paddingHorizontal: 16,
        },
        activatePromoButtonText: {
            color: '#FFFFFF',
            fontSize: 14,
            fontWeight: '800',
            letterSpacing: 0.5,
        },
        // Continue Action Button
        buttonWrapper: {
            marginTop: 6,
            marginBottom: 16,
        },
        continueButton: {
            height: 52,
            borderRadius: 26,
            alignItems: 'center',
            justifyContent: 'center',
            ...Platform.select({
                ios: {
                    shadowColor: '#FFFFFF',
                    shadowOffset: { width: 0, height: 6 },
                    shadowOpacity: 0.15,
                    shadowRadius: 10,
                },
                android: { elevation: 4 },
            }),
        },
        continueButtonText: {
            fontSize: 16,
            fontWeight: '800',
            letterSpacing: 0.5,
            color: '#0A1931',
        },
        // Legal Footers
        footerActionsBlock: {
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 16,
            paddingVertical: 10,
        },
        footerLinkTouch: {
            paddingVertical: 6,
            paddingHorizontal: 8,
        },
        footerActionLinkText: {
            fontSize: 13,
            color: '#FFFFFF',
            fontWeight: '700',
        },
        footerActionsDivider: {
            width: 1,
            height: 14,
            backgroundColor: 'rgba(255, 255, 255, 0.20)',
        },
        legalNotesText: {
            fontSize: 11,
            color: 'rgba(255, 255, 255, 0.50)',
            textAlign: 'center',
            lineHeight: 16,
            paddingHorizontal: 12,
            marginTop: 14,
            marginBottom: 16,
        },
        legalLinksBlock: {
            flexDirection: 'row',
            justifyContent: 'center',
            alignItems: 'center',
            gap: 12,
            marginBottom: 10,
        },
        legalLink: {
            fontSize: 12,
            color: '#FFFFFF',
            fontWeight: '600',
            textDecorationLine: 'underline',
        },
        legalDivider: {
            fontSize: 12,
            color: 'rgba(255, 255, 255, 0.20)',
        },
    });

export default PaywallScreen;
