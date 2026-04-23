/**
 * PromoCodeScreen — shown after auth for free-tier users who haven't
 * redeemed a promo code yet. Entering a valid code unlocks a 7-day
 * free trial. Users can also skip and go straight to the paywall.
 */

import React, { useState } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    KeyboardAvoidingView,
    Platform,
    ScrollView,
    ActivityIndicator,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import Animated, { FadeIn, FadeInUp } from 'react-native-reanimated';
import { useNavigation } from '@react-navigation/native';
import usePromoCodeStore from '../store/promoCodeStore';
import useSubscriptionStore from '../store/subscriptionStore';
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
    const [code, setCode] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const redeemPromoCode = usePromoCodeStore.getState().redeemPromoCode;
    const skipPromo = usePromoCodeStore.getState().skipPromo;

    const handleRedeem = async () => {
        const trimmed = code.trim();
        if (!trimmed) {
            setError(t('promo.enterCode'));
            return;
        }

        setIsLoading(true);
        setError(null);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

        try {
            const result = await redeemPromoCode(trimmed);

            if (result.success) {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                Alert.alert(
                    t('promo.trialActivated'),
                    t('promo.trialActivatedMessage', { days: result.trialDays || 7 }),
                    [
                        {
                            text: t('common.next') || 'Continue',
                            onPress: () => {
                                navigation.reset({
                                    index: 0,
                                    routes: [{ name: 'Main' as never }],
                                });
                            },
                        },
                    ],
                );
            } else {
                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
                setError(result.error || t('promo.invalidCode'));
            }
        } catch (err) {
            setError(t('promo.invalidCode'));
        } finally {
            setIsLoading(false);
        }
    };

    const handleSkip = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        await skipPromo();
        navigation.navigate('Paywall' as never);
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
                    <KeyboardAvoidingView
                        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                        style={styles.flex1}
                    >
                        <ScrollView
                            contentContainerStyle={styles.scrollContent}
                            keyboardShouldPersistTaps="handled"
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
                                <View style={styles.inputCard}>
                                    <Text style={styles.inputLabel}>{t('promo.enterCodeLabel')}</Text>
                                    <View style={styles.inputRow}>
                                        <Ionicons name="ticket" size={20} color={COLORS.textSecondary} />
                                        <TextInput
                                            style={styles.input}
                                            value={code}
                                            onChangeText={(text) => {
                                                setCode(text.toUpperCase());
                                                setError(null);
                                            }}
                                            placeholder={t('promo.placeholder')}
                                            placeholderTextColor="rgba(255,255,255,0.3)"
                                            autoCapitalize="characters"
                                            autoCorrect={false}
                                            maxLength={30}
                                            editable={!isLoading}
                                            returnKeyType="go"
                                            onSubmitEditing={handleRedeem}
                                        />
                                    </View>
                                    {error && (
                                        <View style={styles.errorRow}>
                                            <Ionicons name="alert-circle" size={16} color={COLORS.error} />
                                            <Text style={styles.errorText}>{error}</Text>
                                        </View>
                                    )}
                                </View>
                            </Animated.View>

                            {/* Redeem Button */}
                            <Animated.View entering={FadeInUp.delay(200).springify()}>
                                <TouchableOpacity
                                    style={[styles.redeemButton, (!code.trim() || isLoading) && styles.redeemButtonDisabled]}
                                    onPress={handleRedeem}
                                    disabled={!code.trim() || isLoading}
                                    activeOpacity={0.8}
                                >
                                    {isLoading ? (
                                        <ActivityIndicator color="#0A0A0A" />
                                    ) : (
                                        <Text style={styles.redeemButtonText}>{t('promo.redeem')}</Text>
                                    )}
                                </TouchableOpacity>
                            </Animated.View>

                            {/* Divider */}
                            <Animated.View entering={FadeInUp.delay(250).springify()}>
                                <View style={styles.dividerRow}>
                                    <View style={styles.dividerLine} />
                                    <Text style={styles.dividerText}>{t('promo.or')}</Text>
                                    <View style={styles.dividerLine} />
                                </View>
                            </Animated.View>

                            {/* Upgrade to Pro */}
                            <Animated.View entering={FadeInUp.delay(300).springify()}>
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
                            <Animated.View entering={FadeInUp.delay(350).springify()}>
                                <TouchableOpacity style={styles.skipButton} onPress={handleSkip}>
                                    <Text style={styles.skipText}>{t('promo.skip')}</Text>
                                </TouchableOpacity>
                            </Animated.View>

                            <View style={{ height: 40 }} />
                        </ScrollView>
                    </KeyboardAvoidingView>
                </SafeAreaView>
            </LinearGradient>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: COLORS.background },
    gradient: { flex: 1 },
    safeArea: { flex: 1 },
    flex1: { flex: 1 },

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

    inputCard: {
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 18,
        borderWidth: 1,
        borderColor: COLORS.border,
        marginBottom: 16,
    },
    inputLabel: {
        fontSize: 13,
        fontWeight: '600',
        color: COLORS.textSecondary,
        marginBottom: 10,
    },
    inputRow: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,255,255,0.05)',
        borderRadius: 12,
        paddingHorizontal: 14,
        paddingVertical: 4,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.1)',
    },
    input: {
        flex: 1,
        fontSize: 18,
        fontWeight: '700',
        color: COLORS.text,
        paddingVertical: 14,
        paddingHorizontal: 10,
        letterSpacing: 2,
    },
    errorRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        marginTop: 10,
    },
    errorText: {
        fontSize: 13,
        color: COLORS.error,
        fontWeight: '500',
    },

    redeemButton: {
        backgroundColor: COLORS.premium,
        paddingVertical: 16,
        borderRadius: 14,
        alignItems: 'center',
        marginBottom: 20,
    },
    redeemButtonDisabled: {
        opacity: 0.5,
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
