/**
 * PromoCodeScreen — shown after auth for free-tier users.
 * Users can enter a promo code for a free 7-day trial,
 * or go straight to the paywall to subscribe.
 * Promo codes are used for influencer tracking and referral payments.
 */

import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    ScrollView,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import { useNavigation } from '@react-navigation/native';
import useSubscriptionStore from '../store/subscriptionStore';
import usePromoCodeStore from '../store/promoCodeStore';
import AppColors from '../constants/AppColors';
import { useTranslation } from 'react-i18next';

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
};

const PromoCodeScreen = () => {
    const navigation = useNavigation<any>();
    const { t } = useTranslation();
    const [promoCode, setPromoCode] = useState('');
    const [isRedeeming, setIsRedeeming] = useState(false);
    const [promoError, setPromoError] = useState<string | null>(null);
    const redeemPromoCode = usePromoCodeStore((s) => s.redeemPromoCode);

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
                // Navigate to main — subscription store already updated
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

    const handleSkip = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        navigation.reset({
            index: 0,
            routes: [{ name: 'Main' as never }],
        });
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
                            <Text style={styles.heroTitle}>{t('promo.heroTitle')}</Text>
                            <Text style={styles.heroSubtitle}>
                                {t('promo.heroSubtitle')}
                            </Text>
                        </Animated.View>

                        {/* Promo Code Input */}
                        <Animated.View entering={FadeInUp.delay(100).springify()}>
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
                                    <Text style={styles.errorText}>{promoError}</Text>
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
                                        <Text style={styles.redeemButtonText}>{t('promo.activateTrial')}</Text>
                                    </>
                                )}
                            </TouchableOpacity>
                        </Animated.View>

                        {/* Divider */}
                        <Animated.View entering={FadeInUp.delay(150).springify()}>
                            <View style={styles.dividerRow}>
                                <View style={styles.dividerLine} />
                                <Text style={styles.dividerText}>{t('promo.or')}</Text>
                                <View style={styles.dividerLine} />
                            </View>
                        </Animated.View>

                        {/* Upgrade to Pro */}
                        <Animated.View entering={FadeInUp.delay(200).springify()}>
                            <TouchableOpacity
                                style={styles.upgradeCard}
                                onPress={handlePaywall}
                                activeOpacity={0.8}
                            >
                                <View style={styles.upgradeLeft}>
                                    <Ionicons name="sparkles" size={24} color={COLORS.premium} />
                                    <View style={styles.upgradeTextBlock}>
                                        <Text style={styles.upgradeTitle}>{t('promo.goPro')}</Text>
                                        <Text style={styles.upgradeSubtitle}>{t('promo.goProSubtitle')}</Text>
                                    </View>
                                </View>
                                <Ionicons name="chevron-forward" size={20} color={COLORS.textSecondary} />
                            </TouchableOpacity>
                        </Animated.View>

                        {/* Skip link */}
                        <Animated.View entering={FadeInUp.delay(250).springify()}>
                            <TouchableOpacity style={styles.skipButton} onPress={handleSkip}>
                                <Text style={styles.skipText}>{t('promo.skip')}</Text>
                            </TouchableOpacity>
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

    inputContainer: {
        marginBottom: 12,
    },
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
        backgroundColor: COLORS.premium,
        paddingVertical: 16,
        borderRadius: 14,
        marginBottom: 20,
    },
    redeemButtonText: {
        fontSize: 16,
        fontWeight: '800',
        color: '#0A0A0A',
    },

    dividerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 20,
    },
    dividerLine: {
        flex: 1,
        height: 1,
        backgroundColor: COLORS.border,
    },
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
    upgradeTextBlock: {
        flex: 1,
    },
    upgradeTitle: {
        fontSize: 16,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 2,
    },
    upgradeSubtitle: {
        fontSize: 13,
        color: COLORS.textSecondary,
    },

    skipButton: {
        alignItems: 'center',
        paddingVertical: 14,
    },
    skipText: {
        fontSize: 14,
        color: COLORS.textSecondary,
        fontWeight: '500',
    },
});

export default PromoCodeScreen;
