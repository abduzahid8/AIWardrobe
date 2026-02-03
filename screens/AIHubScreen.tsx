/**
 * AIHubScreen - 2026 Redesign
 * Agentic AI Co-Pilot with Liquid Glass aesthetics and Bento Grid layout
 * Based on 2026 Digital Experience Report guidelines
 */

import React, { useState, useEffect, useRef } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    KeyboardAvoidingView,
    Platform,
    ScrollView,
    Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import moment from 'moment';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withRepeat,
    withSequence,
    withTiming,
    withSpring,
    FadeIn,
    FadeInUp,
    FadeInDown,
    ZoomIn,
    ZoomOut,
    Easing,
    Layout,
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

// 2026 Design System
import { LiquidGlass2026Theme } from '../constants/LiquidGlass2026Theme';
import {
    BentoGrid,
    BentoItem,
    LiquidGlassCard,
    FrostedGlassCard,
    PressableGlassCard,
} from '../components/ui';
import { useAccessibility } from '../hooks/useAccessibility';
import { TahoeIconButton } from '../components/TahoeButton';
import WeatherWidget from '../components/WeatherWidget';

const { width, height } = Dimensions.get('window');
const { colors, spacing, typography, radius, animation } = LiquidGlass2026Theme;

// Quick Action Suggestions (Agentic Style Goals)
const STYLE_GOALS = [
    { id: '1', text: 'Plan my outfits better', icon: 'calendar-outline', gradient: colors.gradients.coolWave },
    { id: '2', text: 'Look professional at work', icon: 'briefcase-outline', gradient: colors.gradients.primaryAccent },
    { id: '3', text: 'Expand my wardrobe', icon: 'add-circle-outline', gradient: colors.gradients.warmGlow },
    { id: '4', text: 'Evolve my style', icon: 'trending-up-outline', gradient: colors.gradients.primaryAccent },
    { id: '5', text: 'Wear my clothes more', icon: 'shirt-outline', gradient: colors.gradients.coolWave },
];

interface StyleGoalType {
    id: string;
    text: string;
    icon: string;
    gradient: readonly string[];
}

// Agentic AI Status Indicator
const AgentStatusIndicator = ({ isActive }: { isActive: boolean }) => {
    const pulse = useSharedValue(1);

    useEffect(() => {
        if (isActive) {
            pulse.value = withRepeat(
                withSequence(
                    withTiming(1.2, { duration: 800 }),
                    withTiming(1, { duration: 800 })
                ),
                -1,
                true
            );
        }
    }, [isActive]);

    const pulseStyle = useAnimatedStyle(() => ({
        transform: [{ scale: pulse.value }],
    }));

    return (
        <View style={styles.agentStatus}>
            <Animated.View style={[styles.agentStatusDot, pulseStyle, isActive && styles.agentStatusActive]} />
            <Text style={styles.agentStatusText}>
                {isActive ? 'AI is thinking...' : 'Ready to help'}
            </Text>
        </View>
    );
};

// Floating AI Avatar with Liquid Glass effect
const FloatingAIAvatar = () => {
    const { isReducedMotionEnabled } = useAccessibility();
    const floatY = useSharedValue(0);
    const glowOpacity = useSharedValue(0.5);

    useEffect(() => {
        if (isReducedMotionEnabled) return;

        floatY.value = withRepeat(
            withSequence(
                withTiming(-8, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) }),
                withTiming(0, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) })
            ),
            -1,
            true
        );

        glowOpacity.value = withRepeat(
            withSequence(
                withTiming(0.8, { duration: 1500 }),
                withTiming(0.4, { duration: 1500 })
            ),
            -1,
            true
        );
    }, [isReducedMotionEnabled]);

    const floatStyle = useAnimatedStyle(() => ({
        transform: [{ translateY: floatY.value }],
    }));

    const glowStyle = useAnimatedStyle(() => ({
        opacity: glowOpacity.value,
    }));

    return (
        <Animated.View style={[styles.avatarContainer, floatStyle]}>
            {/* Glow effect */}
            <Animated.View style={[styles.avatarGlow, glowStyle]} />

            {/* Glass avatar circle */}
            <BlurView intensity={60} tint="light" style={styles.avatarBlur}>
                <LinearGradient
                    colors={colors.gradients.primaryAccent as [string, string]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.avatarGradient}
                >
                    <Ionicons name="sparkles" size={32} color="#FFF" />
                </LinearGradient>
            </BlurView>
        </Animated.View>
    );
};

// Style Goal Button with Glass effect
const StyleGoalButton = ({ goal, index, onPress }: { goal: StyleGoalType; index: number; onPress: () => void }) => {
    const { isReducedMotionEnabled } = useAccessibility();
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, animation.spring.snappy) }],
    }));

    const handlePressIn = () => { scale.value = 0.98; };
    const handlePressOut = () => { scale.value = 1; };

    return (
        <Animated.View
            entering={isReducedMotionEnabled ? undefined : FadeInUp.delay(200 + index * 50).springify()}
        >
            <TouchableOpacity
                onPressIn={handlePressIn}
                onPressOut={handlePressOut}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    onPress();
                }}
                activeOpacity={1}
                accessibilityRole="button"
                accessibilityLabel={goal.text}
            >
                <Animated.View style={[styles.goalButton, animatedStyle]}>
                    <LinearGradient
                        colors={goal.gradient as [string, string]}
                        style={styles.goalIconContainer}
                    >
                        <Ionicons name={goal.icon as any} size={18} color="#FFF" />
                    </LinearGradient>
                    <Text style={styles.goalText}>{goal.text}</Text>
                    <Ionicons name="chevron-forward" size={18} color={colors.text.tertiary} />
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// Quick Action Card with Liquid Glass
const QuickActionCard = ({
    icon,
    title,
    subtitle,
    gradient,
    onPress,
    index,
}: {
    icon: string;
    title: string;
    subtitle: string;
    gradient: readonly string[];
    onPress: () => void;
    index: number;
}) => {
    const { isReducedMotionEnabled } = useAccessibility();

    return (
        <BentoItem colSpan={1} aspectRatio="square" index={index} animated={!isReducedMotionEnabled}>
            <PressableGlassCard
                style={styles.quickActionCard}
                contentStyle={styles.quickActionContent}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    onPress();
                }}
                accessibilityLabel={`${title}: ${subtitle}`}
            >
                <LinearGradient
                    colors={gradient as [string, string]}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.quickActionIcon}
                >
                    <Ionicons name={icon as any} size={22} color="#FFF" />
                </LinearGradient>
                <Text style={styles.quickActionTitle}>{title}</Text>
                <Text style={styles.quickActionSubtitle}>{subtitle}</Text>
            </PressableGlassCard>
        </BentoItem>
    );
};

const AIHubScreen = () => {
    const navigation = useNavigation();
    const { isReducedMotionEnabled, scaleFontSize } = useAccessibility();
    const [message, setMessage] = useState('');
    const [isAgentActive, setIsAgentActive] = useState(false);

    const getGreeting = () => {
        const hour = moment().hour();
        if (hour < 12) return 'Good morning';
        if (hour < 18) return 'Good afternoon';
        return 'Good evening';
    };

    const handleSend = () => {
        if (message.trim()) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            setIsAgentActive(true);
            (navigation as any).navigate('AIChat', { initialMessage: message });
            setMessage('');
            setTimeout(() => setIsAgentActive(false), 500);
        }
    };

    const handleGoalPress = (goal: StyleGoalType) => {
        (navigation as any).navigate('AIChat', { initialMessage: goal.text });
    };

    // Quick actions data
    const quickActions = [
        { icon: 'shirt-outline', title: 'Scan wardrobe', subtitle: 'Add clothes', gradient: colors.gradients.warmGlow, screen: 'Camera' },
        { icon: 'grid-outline', title: 'My Closet', subtitle: 'Browse items', gradient: colors.gradients.coolWave, screen: 'MyCloset' },
        { icon: 'sparkles-outline', title: 'AI Stylist', subtitle: 'Get ideas', gradient: colors.gradients.primaryAccent, screen: 'OutfitAI' },
        { icon: 'person-outline', title: 'Try On', subtitle: 'Virtual fit', gradient: colors.gradients.warmGlow, screen: 'AITryOn' },
        { icon: 'calendar-outline', title: 'Plan outfits', subtitle: 'Weekly', gradient: colors.gradients.coolWave, screen: 'Calendar' },
        { icon: 'people-outline', title: 'Meeting', subtitle: 'Event outfit', gradient: colors.gradients.primaryAccent, screen: 'MeetingOutfit' },
    ];

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header with Liquid Glass */}
                <BlurView intensity={Platform.OS === 'ios' ? 80 : 100} tint="light" style={styles.header}>
                    <TouchableOpacity
                        onPress={() => (navigation as any).navigate('Profile')}
                        style={styles.headerButton}
                        accessibilityLabel="Menu"
                    >
                        <Ionicons name="menu-outline" size={26} color={colors.text.primary} />
                    </TouchableOpacity>

                    <AgentStatusIndicator isActive={isAgentActive} />

                    <TouchableOpacity
                        onPress={() => (navigation as any).navigate('Profile')}
                        style={styles.headerButton}
                        accessibilityLabel="Profile"
                    >
                        <View style={styles.headerAvatar}>
                            <Ionicons name="person" size={18} color={colors.text.secondary} />
                        </View>
                    </TouchableOpacity>
                </BlurView>

                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                    keyboardShouldPersistTaps="handled"
                >
                    {/* Hero Section */}
                    <Animated.View
                        entering={isReducedMotionEnabled ? undefined : FadeIn.delay(100).duration(600)}
                        style={styles.heroSection}
                    >
                        <FloatingAIAvatar />

                        <Text style={styles.heroTitle}>Dress with{'\n'}confidence</Text>
                        <Text style={styles.heroSubtitle}>
                            Your AI stylist that gets you — from the clothes in your closet to the looks you love.
                        </Text>
                    </Animated.View>

                    {/* Talk CTA Button */}
                    <Animated.View
                        entering={isReducedMotionEnabled ? undefined : FadeInUp.delay(150).springify()}
                        style={styles.ctaSection}
                    >
                        <TouchableOpacity
                            style={styles.talkButton}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                (navigation as any).navigate('AIChat');
                            }}
                            accessibilityRole="button"
                            accessibilityLabel="Talk to AI Stylist"
                        >
                            <LinearGradient
                                colors={colors.gradients.primaryAccent as [string, string]}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 0 }}
                                style={styles.talkButtonGradient}
                            >
                                <Ionicons name="chatbubble-ellipses" size={20} color="#FFF" />
                                <Text style={styles.talkButtonText}>Talk to AI Stylist</Text>
                            </LinearGradient>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* Style Goals Section */}
                    <Animated.View
                        entering={isReducedMotionEnabled ? undefined : FadeInUp.delay(180).springify()}
                        style={styles.goalsSection}
                    >
                        <Text style={styles.sectionTitle}>What's your style goal?</Text>

                        <FrostedGlassCard contentStyle={styles.goalsCard}>
                            {STYLE_GOALS.map((goal, index) => (
                                <StyleGoalButton
                                    key={goal.id}
                                    goal={goal}
                                    index={index}
                                    onPress={() => handleGoalPress(goal)}
                                />
                            ))}
                        </FrostedGlassCard>
                    </Animated.View>

                    {/* Quick Actions Bento Grid */}
                    <View style={styles.quickActionsSection}>
                        <Text style={styles.sectionTitle}>Get started</Text>

                        <BentoGrid columns={2} gap={spacing.md} padding={spacing.screenPadding}>
                            {quickActions.map((action, index) => (
                                <QuickActionCard
                                    key={action.screen}
                                    icon={action.icon}
                                    title={action.title}
                                    subtitle={action.subtitle}
                                    gradient={action.gradient}
                                    index={index}
                                    onPress={() => (navigation as any).navigate(action.screen)}
                                />
                            ))}
                        </BentoGrid>
                    </View>

                    {/* Bottom spacing */}
                    <View style={{ height: 160 }} />
                </ScrollView>

                {/* Floating Input with Liquid Glass */}
                <KeyboardAvoidingView
                    behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                    keyboardVerticalOffset={10}
                    style={styles.floatingInputKeyboardView}
                >
                    <BlurView intensity={90} tint="light" style={styles.floatingInputWrapper}>
                        <View style={styles.floatingInputContainer}>
                            <TextInput
                                style={styles.textInput}
                                placeholder="Ask anything about style..."
                                placeholderTextColor={colors.text.tertiary}
                                value={message}
                                onChangeText={setMessage}
                                returnKeyType="send"
                                onSubmitEditing={handleSend}
                                accessibilityLabel="Type your style question"
                            />

                            {/* Morphing Action Button */}
                            <TouchableOpacity
                                onPress={() => {
                                    if (message.trim()) {
                                        handleSend();
                                    } else {
                                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                        (navigation as any).navigate('MagicMirror');
                                    }
                                }}
                                activeOpacity={0.8}
                                accessibilityRole="button"
                                accessibilityLabel={message.trim() ? 'Send message' : 'Open Magic Mirror'}
                            >
                                <Animated.View
                                    style={[
                                        styles.sendButton,
                                        message.trim() ? styles.sendButtonActive : styles.magicButtonActive
                                    ]}
                                    layout={Layout.springify()}
                                >
                                    {message.trim() ? (
                                        <Animated.View
                                            entering={ZoomIn.duration(200)}
                                            exiting={ZoomOut.duration(200)}
                                            style={styles.iconCenter}
                                        >
                                            <Ionicons name="arrow-up" size={20} color="#FFF" />
                                        </Animated.View>
                                    ) : (
                                        <Animated.View
                                            entering={ZoomIn.duration(200)}
                                            exiting={ZoomOut.duration(200)}
                                            style={styles.iconCenter}
                                        >
                                            <Ionicons name="sparkles" size={20} color={colors.accent.primary} />
                                        </Animated.View>
                                    )}
                                </Animated.View>
                            </TouchableOpacity>
                        </View>
                    </BlurView>
                </KeyboardAvoidingView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: colors.background.primary,
    },
    safeArea: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: spacing.screenPadding,
        paddingVertical: spacing.sm + 2,
        backgroundColor: colors.glass.frosted,
        borderBottomWidth: 0.5,
        borderBottomColor: colors.border.subtle,
    },
    headerButton: {
        width: spacing.touchTarget.minimum,
        height: spacing.touchTarget.minimum,
        alignItems: 'center',
        justifyContent: 'center',
    },
    headerAvatar: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: colors.glass.frosted,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: colors.border.glass,
    },

    // Agent Status
    agentStatus: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: spacing.xs,
    },
    agentStatusDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: colors.accent.success,
    },
    agentStatusActive: {
        backgroundColor: colors.accent.primary,
    },
    agentStatusText: {
        ...typography.scale.labelMedium,
        color: colors.text.secondary,
    },

    scrollContent: {
        paddingTop: spacing.lg,
    },

    // Hero Section
    heroSection: {
        alignItems: 'center',
        paddingHorizontal: spacing.screenPadding,
        paddingBottom: spacing.xl,
    },
    avatarContainer: {
        marginBottom: spacing.lg,
        alignItems: 'center',
        justifyContent: 'center',
    },
    avatarGlow: {
        position: 'absolute',
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: colors.accent.primary,
        opacity: 0.2,
    },
    avatarBlur: {
        width: 72,
        height: 72,
        borderRadius: 36,
        overflow: 'hidden',
        borderWidth: 1,
        borderColor: colors.border.glass,
    },
    avatarGradient: {
        width: '100%',
        height: '100%',
        alignItems: 'center',
        justifyContent: 'center',
    },
    heroTitle: {
        ...typography.scale.displayMedium,
        color: colors.text.primary,
        textAlign: 'center',
        marginBottom: spacing.md,
    },
    heroSubtitle: {
        ...typography.scale.bodyLarge,
        color: colors.text.secondary,
        textAlign: 'center',
        paddingHorizontal: spacing.lg,
        lineHeight: 24,
    },

    // CTA Section
    ctaSection: {
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.xl,
    },
    talkButton: {
        borderRadius: radius.pill,
        overflow: 'hidden',
        ...LiquidGlass2026Theme.elevation.getShadow(8),
    },
    talkButtonGradient: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: spacing.md,
        gap: spacing.sm,
    },
    talkButtonText: {
        ...typography.scale.titleMedium,
        color: '#FFF',
        fontWeight: '600',
    },

    // Goals Section
    goalsSection: {
        paddingHorizontal: spacing.screenPadding,
        marginBottom: spacing.xl,
    },
    sectionTitle: {
        ...typography.scale.headlineSmall,
        color: colors.text.primary,
        marginBottom: spacing.md,
    },
    goalsCard: {
        padding: spacing.xs,
    },
    goalButton: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: spacing.md,
        paddingHorizontal: spacing.sm,
        borderBottomWidth: 1,
        borderBottomColor: colors.border.subtle,
    },
    goalIconContainer: {
        width: 32,
        height: 32,
        borderRadius: radius.sm,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: spacing.md,
    },
    goalText: {
        flex: 1,
        ...typography.scale.bodyLarge,
        color: colors.text.primary,
    },

    // Quick Actions
    quickActionsSection: {
        marginBottom: spacing.lg,
    },
    quickActionCard: {
        flex: 1,
        height: '100%',
    },
    quickActionContent: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: spacing.md,
    },
    quickActionIcon: {
        width: 44,
        height: 44,
        borderRadius: radius.md,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: spacing.sm,
    },
    quickActionTitle: {
        ...typography.scale.titleSmall,
        color: colors.text.primary,
        textAlign: 'center',
        marginBottom: spacing.xs,
    },
    quickActionSubtitle: {
        ...typography.scale.bodySmall,
        color: colors.text.secondary,
        textAlign: 'center',
    },

    // Floating Input
    floatingInputKeyboardView: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
    },
    floatingInputWrapper: {
        paddingHorizontal: spacing.screenPadding,
        paddingVertical: spacing.md,
        paddingBottom: Platform.OS === 'ios' ? spacing.xl : spacing.md,
        borderTopWidth: 0.5,
        borderTopColor: colors.border.subtle,
    },
    floatingInputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: colors.glass.frosted,
        borderRadius: radius.pill,
        paddingHorizontal: spacing.md,
        paddingVertical: spacing.xs,
        borderWidth: 1,
        borderColor: colors.border.glass,
    },
    textInput: {
        flex: 1,
        height: 44,
        ...typography.scale.bodyLarge,
        color: colors.text.primary,
        marginRight: spacing.sm,
    },
    sendButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
    },
    sendButtonActive: {
        backgroundColor: colors.accent.primary,
    },
    magicButtonActive: {
        backgroundColor: colors.glass.opaque,
    },
    iconCenter: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
    },
});

export default AIHubScreen;
