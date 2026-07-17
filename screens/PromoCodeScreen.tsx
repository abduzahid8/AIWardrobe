/**
 * PromoCodeScreen — Offer Code redemption surface.
 *
 * PRODUCTION: Shows Apple's native Offer Code redemption sheet
 * (presentCodeRedemptionSheet) — the only Apple-compliant way to
 * redeem codes per Guideline 3.1.1. A "Continue with Free Plan"
 * button is always visible so users are never trapped (Guideline 2.1a).
 *
 * DEV ONLY: Also shows a custom promo code text field for internal testing.
 */

import React, { useState } from 'react';
import { View, TextInput, TouchableOpacity, StyleSheet, ScrollView, ActivityIndicator,  } from 'react-native'
import { ScaledText } from '../components/ui/ScaledText';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import { useNavigation } from '@react-navigation/native';
import usePromoCodeStore from '../store/promoCodeStore';
import AppColors from '../constants/AppColors';
import { useTranslation } from 'react-i18next';
import { iapService } from '../src/services/iapService';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';
import { navigationRef } from '../navigation/navigationRef';

const COLORS = {
    background: '#0A0A0A',
    surface: '#1A1A1A',
    surfaceLight: '#2A2A2A',
    text: '#FFFFFF',
    textSecondary: AppColors.textMuted,
    premium: AppColors.premium,
    premiumDark: AppColors.premiumDark,
    accent: AppColors.accent,
    success: AppColors.success,
    error: '#EF4444',
    border: '#333333',
    freeGreen: '#22C55E',
};

const PromoCodeScreen = () => {
    const navigation = useNavigation<any>();
    const { t } = useTranslation();
    const { completeOnboarding } = useStylePreferenceStore();

    const [promoCode, setPromoCode] = useState('');
    const [isRedeeming, setIsRedeeming] = useState(false);
    const [isRedeemingOffer, setIsRedeemingOffer] = useState(false);
    const [promoError, setPromoError] = useState<string | null>(null);
    const redeemPromoCode = usePromoCodeStore((s) => s.redeemPromoCode);
    const skipPromo = usePromoCodeStore((s) => s.skipPromo);

    /** Navigate to Main (free tier) — always available. */
    const handleContinueFree = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        await skipPromo();
        completeOnboarding();
        setTimeout(() => {
            if (navigationRef.isReady()) {
                navigationRef.reset({ index: 0, routes: [{ name: 'Main' }] });
            }
        }, 100);
    };

    /**
     * Present Apple's native Offer Code redemption sheet.
     * This is the only Apple-compliant way to give users free/discounted
     * access in production (Guideline 3.1.1).
     */
    const handleOfferCode = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setIsRedeemingOffer(true);
        try {
            await iapService.presentCodeRedemptionSheet();
            // After the sheet dismisses, check if subscription status changed
            // (RevenueCat syncs automatically via the customer info listener).
            // If the user successfully redeemed, iapService will have updated
            // the subscription store — navigate to Main.
            setTimeout(() => {
                if (navigationRef.isReady()) {
                    navigationRef.reset({ index: 0, routes: [{ name: 'Main' }] });
                }
            }, 500);
        } catch {
            // Sheet dismissed without redemption — stay on screen
        } finally {
            setIsRedeemingOffer(false);
        }
    };

    /** DEV ONLY: Custom promo code input for internal testing. */
    const handleRedeemCode = async () => {
        const code = promoCode.trim();
        if (!code) {
            setPromoError(t('promo.enterCode'));
            return;
        }

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setIsRedeeming(true);
        setPromoError(null);

        try {
            const result = await redeemPromoCode(code);
            if (result.success) {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                navigation.reset({
                    index: 0,
                    routes: [{ name: 'Main' as never }],
                });
            } else {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
                setPromoError(result.error || t('promo.invalidCode'));
            }
        } catch {
            setPromoError(t('promo.invalidCode'));
        } finally {
            setIsRedeeming(false);
        }
    };

    const handlePaywall = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        navigation.navigate('Paywall' as never);
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#0A0A0A', '#1A1A2E', '#16213E']}
                style={styles.gradient}
            >
                <SafeAreaView style={styles.safeArea}>
                    <ScrollView
                        contentContainerStyle={styles.scrollContent}
                        showsVerticalScrollIndicator={false}
                    >
                        {/* Hero */}
                        <Animated.View entering={FadeIn.duration(500)} style={styles.hero}>
                            <View style={styles.iconContainer}>
                                <Ionicons name="gift" size={40} color={COLORS.premium} />
                            </View>
                            <ScaledText style={styles.heroTitle}>{t('promo.heroTitle')}</ScaledText>
                            <ScaledText style={styles.heroSubtitle}>
                                {t('promo.heroSubtitle')}
                            </ScaledText>
                        </Animated.View>

                        {/* Apple Offer Code button (production) */}
                        <Animated.View entering={FadeInUp.delay(100).springify()}>
                            <TouchableOpacity
                                style={styles.offerCodeButton}
                                onPress={handleOfferCode}
                                activeOpacity={0.8}
                                disabled={isRedeemingOffer}
                                accessibilityLabel={t('promo.redeemOfferCodeA11y')}
                                accessibilityHint={t('promo.offerCodeHintA11y')}
                            >
                                {isRedeemingOffer ? (
                                    <ActivityIndicator color="#0A0A0A" />
                                ) : (
                                    <>
                                        <Ionicons name="ticket-outline" size={20} color="#0A0A0A" />
                                        <ScaledText style={styles.offerCodeButtonText}>
                                            {t('promo.redeemOfferCode', 'Redeem Offer Code')}
                                        </ScaledText>
                                    </>
                                )}
                            </TouchableOpacity>
                        </Animated.View>

                        {/* DEV ONLY: custom promo code text input */}
                        {__DEV__ && (
                            <Animated.View entering={FadeInUp.delay(150).springify()}>
                                <View style={styles.devBadge}>
                                    <Ionicons name="code-slash" size={12} color={COLORS.textSecondary} />
                                    <ScaledText style={styles.devBadgeText}>{t('promo.devOnly')}</ScaledText>
                                </View>
                                <View style={styles.inputContainer}>
                                    <TextInput
                                        style={styles.codeInput}
                                        placeholder={t('promo.codePlaceholder')}
                                        placeholderTextColor={COLORS.textSecondary}
                                        value={promoCode}
                                        onChangeText={(text) => {
                                            setPromoCode(text.toUpperCase());
                                            setPromoError(null);
                                        }}
                                        autoCapitalize="characters"
                                        autoCorrect={false}
                                        maxLength={20}
                                        editable={!isRedeeming}
                                    />
                                    {promoError && (
                                        <ScaledText style={styles.errorText}>{promoError}</ScaledText>
                                    )}
                                </View>
                                <TouchableOpacity
                                    style={styles.redeemButton}
                                    onPress={handleRedeemCode}
                                    activeOpacity={0.8}
                                    disabled={isRedeeming}
                                >
                                    {isRedeeming ? (
                                        <ActivityIndicator color="#0A0A0A" />
                                    ) : (
                                        <>
                                            <Ionicons name="checkmark-circle" size={20} color="#0A0A0A" />
                                            <ScaledText style={styles.redeemButtonText}>{t('promo.activateTrial')}</ScaledText>
                                        </>
                                    )}
                                </TouchableOpacity>
                            </Animated.View>
                        )}

                        {/* Divider */}
                        <Animated.View entering={FadeInUp.delay(200).springify()}>
                            <View style={styles.dividerRow}>
                                <View style={styles.dividerLine} />
                                <ScaledText style={styles.dividerText}>{t('promo.or')}</ScaledText>
                                <View style={styles.dividerLine} />
                            </View>
                        </Animated.View>

                        {/* Upgrade to Pro */}
                        <Animated.View entering={FadeInUp.delay(250).springify()}>
                            <TouchableOpacity
                                style={styles.upgradeCard}
                                onPress={handlePaywall}
                                activeOpacity={0.8}
                                accessibilityLabel={t('promo.subscribeProA11y')}
                            >
                                <View style={styles.upgradeLeft}>
                                    <Ionicons name="sparkles" size={24} color={COLORS.premium} />
                                    <View style={styles.upgradeTextBlock}>
                                        <ScaledText style={styles.upgradeTitle}>{t('promo.goPro')}</ScaledText>
                                        <ScaledText style={styles.upgradeSubtitle}>{t('promo.goProSubtitle')}</ScaledText>
                                    </View>
                                </View>
                                <Ionicons name="chevron-forward" size={20} color={COLORS.textSecondary} />
                            </TouchableOpacity>
                        </Animated.View>

                        {/* Continue with Free Plan — REQUIRED by Apple Guideline 2.1(a) */}
                        <Animated.View entering={FadeInUp.delay(300).springify()}>
                            <TouchableOpacity
                                style={styles.freeButton}
                                onPress={handleContinueFree}
                                activeOpacity={0.8}
                                accessibilityLabel={t('promo.continueFreeA11y')}
                                accessibilityHint={t('promo.continueFreeHintA11y')}
                            >
                                <Ionicons name="checkmark-circle-outline" size={18} color={COLORS.freeGreen} />
                                <ScaledText style={styles.freeButtonText}>
                                    {t('promo.continueFree', 'Continue with Free Plan')}
                                </ScaledText>
                            </TouchableOpacity>
                            <ScaledText style={styles.freeNote}>
                                {t('promo.freeNote', 'Free plan includes 10 AI outfits/day and wardrobe scanning.')}
                            </ScaledText>
                        </Animated.View>

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

    scrollContent: {
        paddingHorizontal: 24,
        paddingVertical: 20,
        justifyContent: 'center',
        flexGrow: 1,
    },

    hero: { alignItems: 'center', marginBottom: 32 },
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

    // Apple Offer Code button (production primary action)
    offerCodeButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 10,
        backgroundColor: COLORS.premium,
        paddingVertical: 16,
        borderRadius: 14,
        marginBottom: 20,
    },
    offerCodeButtonText: {
        fontSize: 16,
        fontWeight: '800',
        color: '#0A0A0A',
    },

    // DEV-only section
    devBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        justifyContent: 'center',
        marginBottom: 10,
    },
    devBadgeText: {
        fontSize: 11,
        color: COLORS.textSecondary,
        fontStyle: 'italic',
    },
    inputContainer: { marginBottom: 12 },
    codeInput: {
        backgroundColor: COLORS.surface,
        borderWidth: 1.5,
        borderColor: COLORS.border,
        borderRadius: 14,
        paddingHorizontal: 18,
        paddingVertical: 16,
        fontSize: 18,
        fontWeight: '700',
        color: COLORS.text,
        letterSpacing: 2,
        textAlign: 'center',
    },
    errorText: {
        color: COLORS.error,
        fontSize: 13,
        marginTop: 8,
        textAlign: 'center',
        fontWeight: '500',
    },
    redeemButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 10,
        backgroundColor: COLORS.surface,
        borderWidth: 1.5,
        borderColor: COLORS.border,
        paddingVertical: 16,
        borderRadius: 14,
        marginBottom: 20,
    },
    redeemButtonText: {
        fontSize: 16,
        fontWeight: '700',
        color: COLORS.text,
    },

    dividerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 20,
    },
    dividerLine: { flex: 1, height: 1, backgroundColor: COLORS.border },
    dividerText: {
        fontSize: 13,
        color: COLORS.textSecondary,
        marginHorizontal: 12,
        fontWeight: '500',
    },

    upgradeCard: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 18,
        borderWidth: 1,
        borderColor: COLORS.border,
        marginBottom: 16,
    },
    upgradeLeft: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 14,
        flex: 1,
    },
    upgradeTextBlock: { flex: 1 },
    upgradeTitle: {
        fontSize: 16,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 2,
    },
    upgradeSubtitle: { fontSize: 13, color: COLORS.textSecondary },

    // Free plan button — required by Apple Guideline 2.1(a)
    freeButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 8,
        paddingVertical: 14,
        borderRadius: 14,
        borderWidth: 1.5,
        borderColor: COLORS.freeGreen,
        marginBottom: 10,
    },
    freeButtonText: {
        fontSize: 15,
        fontWeight: '700',
        color: COLORS.freeGreen,
    },
    freeNote: {
        fontSize: 11,
        color: COLORS.textSecondary,
        textAlign: 'center',
        lineHeight: 16,
        paddingHorizontal: 10,
    },
});

export default PromoCodeScreen;
