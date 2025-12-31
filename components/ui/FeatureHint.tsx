import React, { useEffect, useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Dimensions,
    Modal,
    Pressable,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withRepeat,
    withSequence,
    withTiming,
    withDelay,
    interpolate,
    Extrapolate,
    runOnJS,
    Easing,
    FadeIn,
    FadeOut,
    SlideInDown,
    SlideOutDown,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';
import { colors, spacing, borderRadius, shadows } from '../../src/theme';
import { useFeatureHints } from '../../src/hooks/useFeatureHints';

const { width, height } = Dimensions.get('window');

interface FeatureHintProps {
    id: string;
    title: string;
    description: string;
    icon: keyof typeof Ionicons.glyphMap;
    gradient?: string[];
    position?: 'top' | 'center' | 'bottom';
    onDismiss?: () => void;
}

// Floating 3D orb animation
const AnimatedOrb = ({ icon, gradient }: { icon: keyof typeof Ionicons.glyphMap; gradient: readonly [string, string] }) => {
    const floatY = useSharedValue(0);
    const scale = useSharedValue(1);
    const rotateZ = useSharedValue(0);
    const glowOpacity = useSharedValue(0.5);

    useEffect(() => {
        // Floating animation
        floatY.value = withRepeat(
            withSequence(
                withTiming(-12, { duration: 1500, easing: Easing.bezier(0.25, 0.1, 0.25, 1) }),
                withTiming(0, { duration: 1500, easing: Easing.bezier(0.25, 0.1, 0.25, 1) })
            ),
            -1,
            true
        );

        // Subtle pulse
        scale.value = withRepeat(
            withSequence(
                withTiming(1.08, { duration: 2000 }),
                withTiming(1, { duration: 2000 })
            ),
            -1,
            true
        );

        // Slow rotation
        rotateZ.value = withRepeat(
            withSequence(
                withTiming(5, { duration: 3000 }),
                withTiming(-5, { duration: 3000 })
            ),
            -1,
            true
        );

        // Glow pulse
        glowOpacity.value = withRepeat(
            withSequence(
                withTiming(0.8, { duration: 1500 }),
                withTiming(0.4, { duration: 1500 })
            ),
            -1,
            true
        );
    }, []);

    const orbStyle = useAnimatedStyle(() => ({
        transform: [
            { translateY: floatY.value },
            { scale: scale.value },
            { rotateZ: `${rotateZ.value}deg` },
        ],
    }));

    const glowStyle = useAnimatedStyle(() => ({
        opacity: glowOpacity.value,
    }));

    return (
        <View style={styles.orbContainer}>
            {/* Outer glow */}
            <Animated.View style={[styles.orbGlow, glowStyle]}>
                <LinearGradient
                    colors={[gradient[0], gradient[1], 'transparent']}
                    style={styles.orbGlowGradient}
                />
            </Animated.View>

            {/* Main orb */}
            <Animated.View style={[styles.orbInner, orbStyle]}>
                <LinearGradient
                    colors={[gradient[0], gradient[1]]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.orbGradient}
                >
                    <View style={styles.orbHighlight} />
                    <Ionicons name={icon} size={40} color="#fff" />
                </LinearGradient>
            </Animated.View>
        </View>
    );
};

// Animated hand gesture indicator
const GestureIndicator = ({ type = 'tap' }: { type?: 'tap' | 'swipe' | 'drag' }) => {
    const handY = useSharedValue(0);
    const handOpacity = useSharedValue(1);
    const rippleScale = useSharedValue(0);

    useEffect(() => {
        if (type === 'tap') {
            // Tap animation
            handY.value = withRepeat(
                withSequence(
                    withTiming(-10, { duration: 300 }),
                    withTiming(0, { duration: 200 }),
                    withDelay(800, withTiming(0, { duration: 0 }))
                ),
                -1,
                false
            );

            rippleScale.value = withRepeat(
                withSequence(
                    withDelay(300, withTiming(1.5, { duration: 400 })),
                    withTiming(0, { duration: 0 }),
                    withDelay(800, withTiming(0, { duration: 0 }))
                ),
                -1,
                false
            );
        } else if (type === 'swipe' || type === 'drag') {
            // Swipe animation
            handY.value = withRepeat(
                withSequence(
                    withTiming(-40, { duration: 600, easing: Easing.bezier(0.25, 0.1, 0.25, 1) }),
                    withTiming(0, { duration: 400 }),
                    withDelay(600, withTiming(0, { duration: 0 }))
                ),
                -1,
                false
            );
        }
    }, [type]);

    const handStyle = useAnimatedStyle(() => ({
        transform: [{ translateY: handY.value }],
        opacity: handOpacity.value,
    }));

    const rippleStyle = useAnimatedStyle(() => ({
        transform: [{ scale: rippleScale.value }],
        opacity: interpolate(rippleScale.value, [0, 1.5], [0.6, 0], Extrapolate.CLAMP),
    }));

    return (
        <View style={styles.gestureContainer}>
            {type === 'tap' && (
                <Animated.View style={[styles.ripple, rippleStyle]} />
            )}
            <Animated.View style={handStyle}>
                <Ionicons
                    name={type === 'drag' ? 'hand-left' : 'finger-print'}
                    size={32}
                    color="rgba(255,255,255,0.9)"
                />
            </Animated.View>
        </View>
    );
};

export const FeatureHint: React.FC<FeatureHintProps> = ({
    id,
    title,
    description,
    icon,
    gradient = ['#8B5CF6', '#6366F1'] as const,
    position = 'center',
    onDismiss,
}) => {
    const { hasSeenHint, markHintAsSeen, isLoading } = useFeatureHints();
    const [visible, setVisible] = useState(false);

    useEffect(() => {
        if (!isLoading && !hasSeenHint(id)) {
            // Small delay before showing hint
            const timer = setTimeout(() => {
                setVisible(true);
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            }, 800);
            return () => clearTimeout(timer);
        }
    }, [isLoading, id]);

    const handleDismiss = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        await markHintAsSeen(id);
        setVisible(false);
        onDismiss?.();
    };

    if (!visible) return null;

    return (
        <Modal
            visible={visible}
            transparent
            animationType="none"
            statusBarTranslucent
        >
            <Animated.View
                entering={FadeIn.duration(300)}
                exiting={FadeOut.duration(200)}
                style={styles.overlay}
            >
                <Pressable style={styles.overlayPress} onPress={handleDismiss}>
                    <BlurView intensity={30} tint="dark" style={StyleSheet.absoluteFill} />
                </Pressable>

                <Animated.View
                    entering={SlideInDown.springify().damping(15)}
                    exiting={SlideOutDown.duration(200)}
                    style={[
                        styles.hintCard,
                        position === 'top' && styles.positionTop,
                        position === 'bottom' && styles.positionBottom,
                    ]}
                >
                    {/* 3D Animated Orb */}
                    <AnimatedOrb icon={icon} gradient={[gradient[0], gradient[1]] as const} />

                    {/* Content */}
                    <View style={styles.content}>
                        <Text style={styles.title}>{title}</Text>
                        <Text style={styles.description}>{description}</Text>
                    </View>

                    {/* Gesture Indicator */}
                    <GestureIndicator type="tap" />

                    {/* CTA Button */}
                    <TouchableOpacity
                        style={styles.ctaButton}
                        onPress={handleDismiss}
                        activeOpacity={0.8}
                    >
                        <LinearGradient
                            colors={[gradient[0], gradient[1]]}
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 0 }}
                            style={styles.ctaGradient}
                        >
                            <Text style={styles.ctaText}>Got it!</Text>
                        </LinearGradient>
                    </TouchableOpacity>

                    {/* Skip hint */}
                    <TouchableOpacity onPress={handleDismiss} style={styles.skipButton}>
                        <Text style={styles.skipText}>Tap anywhere to dismiss</Text>
                    </TouchableOpacity>
                </Animated.View>
            </Animated.View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    overlay: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    overlayPress: {
        ...StyleSheet.absoluteFillObject,
    },
    hintCard: {
        width: width * 0.85,
        backgroundColor: 'rgba(30, 30, 35, 0.95)',
        borderRadius: borderRadius.xxl,
        padding: spacing.xl,
        alignItems: 'center',
        ...shadows.strong,
        borderWidth: 1,
        borderColor: 'rgba(255, 255, 255, 0.1)',
    },
    positionTop: {
        marginTop: 100,
    },
    positionBottom: {
        marginBottom: 100,
    },
    orbContainer: {
        width: 120,
        height: 120,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: spacing.l,
    },
    orbGlow: {
        position: 'absolute',
        width: 140,
        height: 140,
    },
    orbGlowGradient: {
        flex: 1,
        borderRadius: 70,
    },
    orbInner: {
        width: 90,
        height: 90,
        borderRadius: 45,
        ...shadows.glow,
    },
    orbGradient: {
        flex: 1,
        borderRadius: 45,
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
    },
    orbHighlight: {
        position: 'absolute',
        top: 8,
        left: 12,
        width: 25,
        height: 25,
        borderRadius: 15,
        backgroundColor: 'rgba(255, 255, 255, 0.3)',
    },
    content: {
        alignItems: 'center',
        marginBottom: spacing.l,
    },
    title: {
        fontSize: 24,
        fontWeight: '700',
        color: '#FFFFFF',
        textAlign: 'center',
        marginBottom: spacing.s,
    },
    description: {
        fontSize: 16,
        color: 'rgba(255, 255, 255, 0.7)',
        textAlign: 'center',
        lineHeight: 24,
    },
    gestureContainer: {
        width: 60,
        height: 60,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: spacing.l,
    },
    ripple: {
        position: 'absolute',
        width: 40,
        height: 40,
        borderRadius: 20,
        borderWidth: 2,
        borderColor: 'rgba(255, 255, 255, 0.5)',
    },
    ctaButton: {
        borderRadius: borderRadius.full,
        overflow: 'hidden',
        marginBottom: spacing.m,
    },
    ctaGradient: {
        paddingVertical: spacing.m,
        paddingHorizontal: spacing.xxl,
    },
    ctaText: {
        fontSize: 17,
        fontWeight: '600',
        color: '#FFFFFF',
    },
    skipButton: {
        padding: spacing.s,
    },
    skipText: {
        fontSize: 13,
        color: 'rgba(255, 255, 255, 0.5)',
    },
});

export default FeatureHint;
