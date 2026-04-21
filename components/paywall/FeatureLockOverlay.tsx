/**
 * FeatureLockOverlay — full-screen (or container-fill) locked state that
 * replaces a gated feature. Shows a beautiful tease of what's behind
 * the paywall and a single CTA to upgrade.
 *
 * This is intentionally NOT a modal — users should feel like the feature
 * is "almost theirs, just one tap away". That's what converts.
 */

import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Platform } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as Haptics from 'expo-haptics';
import { useNavigation } from '@react-navigation/native';
import { SafeAreaView } from 'react-native-safe-area-context';

interface FeatureLockOverlayProps {
    /** Which tier unlocks this feature (displayed in copy). */
    requiredTier: 'Pro' | 'Max';
    /** Feature name, shown in the title. */
    featureName: string;
    /** One-line tagline about the feature. */
    tagline: string;
    /** Bullet points shown under the tagline (3–4 recommended). */
    bullets: string[];
    /** Icon name from Ionicons. */
    icon?: keyof typeof Ionicons.glyphMap;
    /** Whether to show a safe-area top inset (default true). */
    withSafeArea?: boolean;
}

const FeatureLockOverlay: React.FC<FeatureLockOverlayProps> = ({
    requiredTier,
    featureName,
    tagline,
    bullets,
    icon = 'sparkles',
    withSafeArea = true,
}) => {
    const navigation = useNavigation<any>();

    const handleUpgrade = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        navigation.navigate('Paywall');
    };

    const accent = requiredTier === 'Max' ? '#FFD700' : '#8B5CF6';
    const accentDark = requiredTier === 'Max' ? '#B8860B' : '#6D28D9';

    const Content = (
        <View style={styles.container}>
            <LinearGradient
                colors={['#0A0A0A', '#1A1A2E', '#16213E']}
                style={StyleSheet.absoluteFill}
            />

            <View style={styles.inner}>
                <View style={[styles.iconCircle, { backgroundColor: `${accent}22`, borderColor: `${accent}55` }]}>
                    <Ionicons name={icon} size={48} color={accent} />
                    <View style={styles.lockBadge}>
                        <Ionicons name="lock-closed" size={16} color="#FFFFFF" />
                    </View>
                </View>

                <View style={[styles.tierPill, { backgroundColor: `${accent}22`, borderColor: accent }]}>
                    <Ionicons
                        name={requiredTier === 'Max' ? 'diamond' : 'star'}
                        size={12}
                        color={accent}
                    />
                    <Text style={[styles.tierPillText, { color: accent }]}>
                        {requiredTier.toUpperCase()} FEATURE
                    </Text>
                </View>

                <Text style={styles.title}>{featureName}</Text>
                <Text style={styles.tagline}>{tagline}</Text>

                <View style={styles.bullets}>
                    {bullets.map((b, i) => (
                        <View key={i} style={styles.bulletRow}>
                            <View style={[styles.bulletDot, { backgroundColor: accent }]} />
                            <Text style={styles.bulletText}>{b}</Text>
                        </View>
                    ))}
                </View>

                <TouchableOpacity
                    onPress={handleUpgrade}
                    activeOpacity={0.9}
                    style={styles.ctaWrap}
                >
                    <LinearGradient
                        colors={[accent, accentDark]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 1 }}
                        style={styles.cta}
                    >
                        <Ionicons name="rocket" size={18} color="#0A0A0A" />
                        <Text style={styles.ctaText}>Unlock with {requiredTier}</Text>
                    </LinearGradient>
                </TouchableOpacity>

                <Text style={styles.subCta}>Cancel anytime · No hidden fees</Text>
            </View>
        </View>
    );

    if (withSafeArea) {
        return <SafeAreaView style={{ flex: 1, backgroundColor: '#0A0A0A' }}>{Content}</SafeAreaView>;
    }
    return Content;
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#0A0A0A',
    },
    inner: {
        flex: 1,
        paddingHorizontal: 28,
        alignItems: 'center',
        justifyContent: 'center',
    },
    iconCircle: {
        width: 120,
        height: 120,
        borderRadius: 60,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 2,
        marginBottom: 24,
        position: 'relative',
    },
    lockBadge: {
        position: 'absolute',
        bottom: 4,
        right: 4,
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: '#0A0A0A',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 2,
        borderColor: '#1A1A2E',
    },
    tierPill: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 6,
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 12,
        borderWidth: 1,
        marginBottom: 16,
    },
    tierPillText: {
        fontSize: 11,
        fontWeight: '800',
        letterSpacing: 0.8,
    },
    title: {
        fontSize: 28,
        fontWeight: '800',
        color: '#FFFFFF',
        textAlign: 'center',
        marginBottom: 8,
    },
    tagline: {
        fontSize: 15,
        color: 'rgba(255,255,255,0.7)',
        textAlign: 'center',
        lineHeight: 22,
        marginBottom: 28,
        paddingHorizontal: 8,
    },
    bullets: {
        width: '100%',
        gap: 12,
        marginBottom: 32,
    },
    bulletRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    bulletDot: {
        width: 6,
        height: 6,
        borderRadius: 3,
    },
    bulletText: {
        fontSize: 14,
        color: 'rgba(255,255,255,0.85)',
        fontWeight: '500',
        flex: 1,
    },
    ctaWrap: {
        width: '100%',
        borderRadius: 16,
        overflow: 'hidden',
        ...Platform.select({
            ios: {
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 8 },
                shadowOpacity: 0.3,
                shadowRadius: 16,
            },
            android: { elevation: 8 },
        }),
    },
    cta: {
        paddingVertical: 16,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 10,
    },
    ctaText: {
        fontSize: 16,
        fontWeight: '800',
        color: '#0A0A0A',
    },
    subCta: {
        fontSize: 12,
        color: 'rgba(255,255,255,0.5)',
        marginTop: 14,
    },
});

export default FeatureLockOverlay;
