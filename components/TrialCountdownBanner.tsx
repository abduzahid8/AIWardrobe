/**
 * TrialCountdownBanner
 *
 * A slim, animated amber-gradient banner shown at the top of key screens
 * while the user's 7-day free trial is active. Tapping it navigates to
 * the Paywall so users can upgrade.
 *
 * Returns null for paid users or when the trial has already ended.
 */

import React, { useEffect } from 'react';
import {
    TouchableOpacity,
    View,
    Text,
    StyleSheet,
    Platform,
} from 'react-native';
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
import { useTrialStatus } from '../hooks/useTrialStatus';

export const TrialCountdownBanner: React.FC = () => {
    const navigation = useNavigation<any>();
    const { isTrialActive, daysRemaining, hoursRemaining } = useTrialStatus();

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

    if (!isTrialActive) return null;

    const isLastDay = daysRemaining <= 1;
    const label = isLastDay
        ? hoursRemaining <= 1
            ? 'Less than 1 hour left in your free trial'
            : `${hoursRemaining}h left in your free trial — Upgrade now`
        : `${daysRemaining} day${daysRemaining !== 1 ? 's' : ''} left in your free trial`;

    const gradientColors: [string, string] = isLastDay
        ? ['#C0392B', '#E74C3C']   // urgent red on last day
        : ['#E67E22', '#F39C12'];   // amber during trial

    return (
        <Animated.View entering={FadeInDown.duration(400)} style={animatedStyle}>
            <TouchableOpacity
                activeOpacity={0.85}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    navigation.navigate('Paywall');
                }}
                accessibilityLabel="Open upgrade plans"
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
                            name={isLastDay ? 'alert-circle' : 'time-outline'}
                            size={15}
                            color="#FFF"
                            style={styles.icon}
                        />
                        <Text style={styles.text} numberOfLines={1}>
                            {label}
                        </Text>
                    </View>
                    <View style={styles.cta}>
                        <Text style={styles.ctaText}>Upgrade</Text>
                        <Ionicons name="chevron-forward" size={13} color="#FFF" />
                    </View>
                </LinearGradient>
            </TouchableOpacity>
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
