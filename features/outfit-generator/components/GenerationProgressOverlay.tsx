import React, { useEffect } from 'react';
import { View, Text, TouchableOpacity, StyleSheet } from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withSequence,
    withTiming,
    Easing,
    cancelAnimation,
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import LiquidGlass2026Theme from '../../../constants/LiquidGlass2026Theme';

// ─── Constants ────────────────────────────────────────────────────────────────

const ACCENT = '#007AFF';
const ACCENT_GLOW = 'rgba(0, 122, 255, 0.25)';

// ─── LiquidGlassSpinner (local copy — matches MyClosetScreen implementation) ──

const LiquidGlassSpinner: React.FC = () => {
    const rotation = useSharedValue(0);
    const pulse = useSharedValue(1);
    const innerPulse = useSharedValue(0.6);

    useEffect(() => {
        rotation.value = withRepeat(
            withTiming(360, { duration: 2400, easing: Easing.linear }),
            -1,
            false,
        );
        pulse.value = withRepeat(
            withSequence(
                withTiming(1.08, { duration: 1200, easing: Easing.inOut(Easing.sin) }),
                withTiming(1, { duration: 1200, easing: Easing.inOut(Easing.sin) }),
            ),
            -1,
            true,
        );
        innerPulse.value = withRepeat(
            withSequence(
                withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.sin) }),
                withTiming(0.6, { duration: 1800, easing: Easing.inOut(Easing.sin) }),
            ),
            -1,
            true,
        );

        return () => {
            cancelAnimation(rotation);
            cancelAnimation(pulse);
            cancelAnimation(innerPulse);
        };
    }, []);

    const outerStyle = useAnimatedStyle(() => ({
        transform: [{ scale: pulse.value }],
    }));
    const ringStyle = useAnimatedStyle(() => ({
        transform: [{ rotate: `${rotation.value}deg` }],
    }));
    const glowStyle = useAnimatedStyle(() => ({
        opacity: innerPulse.value,
    }));

    return (
        <Animated.View style={[styles.spinnerContainer, outerStyle]}>
            <Animated.View style={[styles.spinnerGlow, glowStyle]} />
            <Animated.View style={[styles.spinnerRing, ringStyle]}>
                <LinearGradient
                    colors={[
                        'rgba(0,122,255,0.6)',
                        'rgba(0,122,255,0)',
                        'rgba(0,122,255,0.3)',
                    ]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.spinnerRingGradient}
                />
            </Animated.View>
            <BlurView intensity={40} tint="light" style={styles.spinnerInner}>
                <Ionicons name="sparkles" size={28} color={ACCENT} />
            </BlurView>
        </Animated.View>
    );
};

// ─── Props ────────────────────────────────────────────────────────────────────

interface GenerationProgressOverlayProps {
    visible: boolean;
    statusMessage: string;
    onCancel: () => void;
}

// ─── Component ────────────────────────────────────────────────────────────────

const GenerationProgressOverlay: React.FC<GenerationProgressOverlayProps> = ({
    visible,
    statusMessage,
    onCancel,
}) => {
    if (!visible) {
        return null;
    }

    return (
        // pointerEvents="box-only" blocks touches on the overlay itself while
        // still allowing child views (the cancel button) to receive events.
        <View style={styles.overlay} pointerEvents="box-only">
            <View style={styles.content} pointerEvents="box-none">
                <LiquidGlassSpinner />

                {/* Status message — announced by VoiceOver/TalkBack on change */}
                <Text
                    style={styles.statusMessage}
                    accessibilityLiveRegion="polite"
                    accessibilityRole="text"
                >
                    {statusMessage}
                </Text>

                {/* Cancel button — minimum 44 pt touch target (iOS HIG) */}
                <TouchableOpacity
                    style={styles.cancelButton}
                    onPress={onCancel}
                    activeOpacity={0.75}
                    accessibilityLabel="Cancel outfit generation"
                    accessibilityRole="button"
                    accessibilityHint="Stops the current outfit generation and returns to your wardrobe"
                >
                    <Text style={styles.cancelButtonText}>Cancel</Text>
                </TouchableOpacity>
            </View>
        </View>
    );
};

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
    overlay: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 999,
    },
    content: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingHorizontal: 32,
    },

    // ── Spinner ──────────────────────────────────────────────────────────────
    spinnerContainer: {
        width: 80,
        height: 80,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
    },
    spinnerGlow: {
        position: 'absolute',
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: ACCENT_GLOW,
    },
    spinnerRing: {
        position: 'absolute',
        width: 80,
        height: 80,
        borderRadius: 40,
        overflow: 'hidden',
    },
    spinnerRingGradient: {
        width: 80,
        height: 80,
        borderRadius: 40,
        borderWidth: 3,
        borderColor: 'transparent',
    },
    spinnerInner: {
        width: 60,
        height: 60,
        borderRadius: 30,
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        backgroundColor: 'rgba(255, 255, 255, 0.55)',
    },

    // ── Status message ────────────────────────────────────────────────────────
    statusMessage: {
        color: '#FFFFFF',
        fontSize: 16,
        fontWeight: '600',
        textAlign: 'center',
        marginBottom: 32,
        letterSpacing: 0.2,
    },

    // ── Cancel button ─────────────────────────────────────────────────────────
    cancelButton: {
        minWidth: 44,
        minHeight: 44,
        paddingHorizontal: 28,
        paddingVertical: 12,
        borderRadius: LiquidGlass2026Theme.radius.pill,
        backgroundColor: 'rgba(255, 255, 255, 0.18)',
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.35)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    cancelButtonText: {
        color: '#FFFFFF',
        fontSize: 15,
        fontWeight: '600',
        letterSpacing: 0.1,
    },
});

export default GenerationProgressOverlay;
