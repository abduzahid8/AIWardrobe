/**
 * QuotaBadge — compact "X of 10 outfits left today" pill.
 *
 * Conversion strategy:
 *   - Hidden for Max tier (unlimited → no friction)
 *   - Shown for Free/Pro with running counter
 *   - Color shifts yellow → red as quota depletes (urgency cue)
 *   - When fully exhausted, tapping opens the Paywall
 *
 * Usage:
 *   <QuotaBadge feature="aiOutfits" label="outfits" />
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import { useSubscriptionGate } from '../../src/hooks/useSubscriptionGate';
import type { FeatureKey } from '../../store/subscriptionStore';

interface QuotaBadgeProps {
    feature: Extract<FeatureKey, 'aiOutfits' | 'tryOns' | 'wardrobeScans'>;
    /** Short label e.g. "outfits", "try-ons". */
    label: string;
    /** Optional style override for the container. */
    style?: any;
    /** Hide entirely for unlimited tiers (default true). */
    hideWhenUnlimited?: boolean;
}

const QuotaBadge: React.FC<QuotaBadgeProps> = ({
    feature,
    label,
    style,
    hideWhenUnlimited = true,
}) => {
    const navigation = useNavigation<any>();
    const { getRemaining, getDailyLimit, tier } = useSubscriptionGate();

    const limit = getDailyLimit(feature);
    const remaining = getRemaining(feature);

    // Unlimited (-1) or user has no access at all (0 + locked feature)
    if (limit === -1) {
        if (hideWhenUnlimited) return null;
        return (
            <View style={[styles.pill, styles.pillUnlimited, style]}>
                <Ionicons name="infinite" size={12} color="#FFD700" />
                <Text style={[styles.text, styles.textLight]}>Unlimited {label}</Text>
            </View>
        );
    }

    // Feature fully locked (0 allowed) — e.g. Free trying tryOns
    if (limit === 0) {
        return (
            <TouchableOpacity
                style={[styles.pill, styles.pillLocked, style]}
                onPress={() => navigation.navigate('Paywall')}
                activeOpacity={0.85}
            >
                <Ionicons name="lock-closed" size={11} color="#FFFFFF" />
                <Text style={[styles.text, styles.textLight]}>Unlock {label}</Text>
            </TouchableOpacity>
        );
    }

    // Daily quota remaining
    const pct = remaining / limit;
    const exhausted = remaining === 0;
    const low = pct <= 0.3;

    const pillStyle = exhausted
        ? styles.pillExhausted
        : low
            ? styles.pillLow
            : styles.pillOk;
    const textColor = exhausted || low ? '#FFFFFF' : '#FFFFFF';

    return (
        <TouchableOpacity
            style={[styles.pill, pillStyle, style]}
            onPress={() => {
                if (exhausted) {
                    navigation.navigate('Paywall');
                }
            }}
            activeOpacity={exhausted ? 0.85 : 1}
            disabled={!exhausted}
        >
            <Ionicons
                name={exhausted ? 'alert-circle' : 'flash'}
                size={12}
                color={textColor}
            />
            <Text style={[styles.text, { color: textColor }]}>
                {exhausted
                    ? `Out of ${label} — tap to upgrade`
                    : `${remaining} of ${limit} ${label} left today`}
            </Text>
        </TouchableOpacity>
    );
};

const styles = StyleSheet.create({
    pill: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        paddingHorizontal: 12,
        paddingVertical: 7,
        borderRadius: 14,
        alignSelf: 'flex-start',
    },
    pillOk: {
        backgroundColor: 'rgba(139, 92, 246, 0.18)',
        borderWidth: 1,
        borderColor: 'rgba(139, 92, 246, 0.4)',
    },
    pillLow: {
        backgroundColor: 'rgba(251, 146, 60, 0.22)',
        borderWidth: 1,
        borderColor: 'rgba(251, 146, 60, 0.6)',
    },
    pillExhausted: {
        backgroundColor: 'rgba(239, 68, 68, 0.25)',
        borderWidth: 1,
        borderColor: 'rgba(239, 68, 68, 0.7)',
    },
    pillUnlimited: {
        backgroundColor: 'rgba(255, 215, 0, 0.18)',
        borderWidth: 1,
        borderColor: 'rgba(255, 215, 0, 0.5)',
    },
    pillLocked: {
        backgroundColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.25)',
    },
    text: {
        fontSize: 12,
        fontWeight: '700',
        letterSpacing: 0.2,
    },
    textLight: { color: '#FFFFFF' },
});

export default QuotaBadge;
