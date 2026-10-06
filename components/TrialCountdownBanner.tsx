/**
 * TrialCountdownBanner (+ expired-trial upsell)
 *
 * - During the 7-day trial: slim amber/red countdown → Paywall.
 * - After the trial ends: non-blocking purple "Trial ended — Upgrade"
 *   banner → Paywall. The app itself stays fully usable on the free tier;
 *   only individual premium features redirect to Paywall via
 *   useSubscriptionGate.
 *
 * Returns null for paid users.
 */

import React, { useEffect } from 'react';
import { TouchableOpacity, View, StyleSheet, Platform,  } from 'react-native'
import { ScaledText } from './ui/ScaledText';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withTiming,
    withSequence,
    FadeInDown,
} from 'react-native-reanimated';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import { useTranslation } from 'react-i18next';
import { useTrialStatus } from '../hooks/useTrialStatus';

export const TrialCountdownBanner: React.FC = () => {
    const navigation = useNavigation<any>();
    const { isTrialActive, isTrialExpired, daysRemaining, hoursRemaining } = useTrialStatus();
    const { t } = useTranslation();

    // Pulse animation — draws the eye without being obnoxious
    const opacity = useSharedValue(1);
    const animatedStyle = useAnimatedStyle(() => ({ opacity: opacity.value }));

    useEffect(() => {
        // Pulse only on the last day (< 24 h remaining)
        if (isTrialActive && daysRemaining <= 1) {
            opacity.value = withRepeat(
                withSequence(
                    withTiming(0.65, { duration: 900 }),
                    withTiming(1, { duration: 900 }),
                ),
                -1,
                true,
            );
        } else {
            opacity.value = 1;
        }
    }, [isTrialActive, daysRemaining]);

    // Paid / undetermined users: no banner. Expired trials get a
    // non-blocking upsell banner below (app stays usable).
    if (!isTrialActive && !isTrialExpired) return null;

    const isExpired = !isTrialActive && isTrialExpired;
    const isLastDay = !isExpired && daysRemaining <= 1;
    const label = isExpired
        ? t('trialExpired.banner', 'Your free trial ended — Upgrade to Pro')
        : isLastDay
        ? hoursRemaining <= 1
            ? t('trialCountdown.lessThanOneHour')
            : t('trialCountdown.hoursLeft', { hours: hoursRemaining })
        : t('trialCountdown.daysLeft', { days: daysRemaining, suffix: daysRemaining !== 1 ? 's' : '' });

    const gradientColors: [string, string] = isExpired
        ? ['#6D28D9', '#8B5CF6']   // purple upsell after expiry (non-blocking)
        : isLastDay
        ? ['#C0392B', '#E74C3C']   // urgent red on last day
        : ['#E67E22', '#F39C12'];   // amber during trial

    return (
        // Use a plain View wrapper for the entering animation to avoid the
        // Reanimated warning: "Property 'opacity' of AnimatedComponent(View)
        // may be overwritten by a layout animation." The inner Animated.View
        // owns the pulse opacity; the outer Animated.View owns the enter animation.
        <Animated.View entering={FadeInDown.duration(400)}>
            <Animated.View style={animatedStyle}>
                <TouchableOpacity
                activeOpacity={0.85}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    navigation.navigate('Paywall');
                }}
                accessibilityLabel={t('trialCountdown.openUpgradePlans')}
                accessibilityRole="button"
            >
                <LinearGradient
                    colors={gradientColors}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.banner}
                >
                    <View style={styles.inner}>
                        <Ionicons
                            name={isExpired ? 'diamond-outline' : isLastDay ? 'alert-circle' : 'time-outline'}
                            size={15}
                            color="#FFF"
                            style={styles.icon}
                        />
                        <ScaledText style={styles.text} numberOfLines={2}>
                            {label}
                        </ScaledText>
                    </View>
                    <View style={styles.cta}>
                        <ScaledText style={styles.ctaText}>{t('trialCountdown.upgrade')}</ScaledText>
                        <Ionicons name="chevron-forward" size={13} color="#FFF" />
                    </View>
                </LinearGradient>
            </TouchableOpacity>
            </Animated.View>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    banner: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingVertical: 9,
        paddingHorizontal: 16,
        ...Platform.select({
            ios: {
                shadowColor: '#E67E22',
                shadowOffset: { width: 0, height: 2 },
                shadowOpacity: 0.35,
                shadowRadius: 6,
            },
            android: { elevation: 4 },
        }),
    },
    inner: {
        flexDirection: 'row',
        alignItems: 'center',
        flexShrink: 1,
    },
    icon: { marginRight: 6 },
    text: {
        fontSize: 12,
        fontWeight: '600',
        color: '#FFF',
        flexShrink: 1,
    },
    cta: {
        flexDirection: 'row',
        alignItems: 'center',
        marginLeft: 10,
        backgroundColor: 'rgba(255,255,255,0.25)',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 20,
        gap: 2,
    },
    ctaText: {
        fontSize: 11,
        fontWeight: '800',
        color: '#FFF',
        letterSpacing: 0.3,
    },
});

export default TrialCountdownBanner;
