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
    Easing,
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';
import { TahoeIconButton } from '../components/TahoeButton';
import AppColors from '../constants/AppColors';

const { width, height } = Dimensions.get('window');

// Use unified AppColors
const ALTA = {
    background: AppColors.background,
    surface: AppColors.surface,
    surfaceLight: AppColors.surfaceSecondary,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    textMuted: AppColors.textMuted,
    border: AppColors.border,
    glass: AppColors.glassDark,
};

// Quick Action Suggestions (Like Alta's Style Goals)
const STYLE_GOALS = [
    { id: '1', text: 'Plan my outfits better', icon: 'calendar-outline' },
    { id: '2', text: 'Look professional at work', icon: 'briefcase-outline' },
    { id: '3', text: 'Expand my wardrobe', icon: 'add-circle-outline' },
    { id: '4', text: 'Evolve my style', icon: 'trending-up-outline' },
    { id: '5', text: 'Wear my clothes more', icon: 'shirt-outline' },
];

// iOS 26 Tahoe Press Hook
const useTahoePress = () => {
    const scale = useSharedValue(1);
    const opacity = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 20, stiffness: 400 }) }],
        opacity: opacity.value,
    }));

    const onPressIn = () => {
        scale.value = 0.97;
        opacity.value = withTiming(0.9, { duration: 60 });
    };

    const onPressOut = () => {
        scale.value = 1;
        opacity.value = withTiming(1, { duration: 100 });
    };

    return { animatedStyle, onPressIn, onPressOut };
};

// Floating AI Avatar
const FloatingAIAvatar = () => {
    const floatY = useSharedValue(0);
    const glowOpacity = useSharedValue(0.5);

    useEffect(() => {
        floatY.value = withRepeat(
            withSequence(
                withTiming(-10, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) }),
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
    }, []);

    const floatStyle = useAnimatedStyle(() => ({
        transform: [{ translateY: floatY.value }],
    }));

    return (
        <Animated.View style={[styles.avatarContainer, floatStyle]}>
            {/* Simple elegant avatar circle */}
            <View style={styles.avatarCircle}>
                <Ionicons name="sparkles" size={36} color={ALTA.primary} />
            </View>
        </Animated.View>
    );
};

// Style Goal Button (Like Alta's homepage)
const StyleGoalButton = ({ goal, index, onPress }: { goal: any; index: number; onPress: () => void }) => {
    const { animatedStyle, onPressIn, onPressOut } = useTahoePress();

    return (
        <Animated.View
            entering={FadeInUp.delay(200 + index * 50).springify()}
        >
            <TouchableOpacity
                onPressIn={onPressIn}
                onPressOut={onPressOut}
                onPress={() => {
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                    onPress();
                }}
                activeOpacity={1}
            >
                <Animated.View style={[styles.goalButton, animatedStyle]}>
                    <Ionicons name={goal.icon} size={20} color={ALTA.primary} style={styles.goalIcon} />
                    <Text style={styles.goalText}>{goal.text}</Text>
                    <Ionicons name="arrow-forward" size={16} color={ALTA.textMuted} />
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// Quick Action Card
const QuickActionCard = ({
    icon,
    title,
    subtitle,
    onPress
}: {
    icon: any;
    title: string;
    subtitle: string;
    onPress: () => void;
}) => {
    const { animatedStyle, onPressIn, onPressOut } = useTahoePress();

    return (
        <TouchableOpacity
            onPressIn={onPressIn}
            onPressOut={onPressOut}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                onPress();
            }}
            activeOpacity={1}
        >
            <Animated.View style={[styles.quickActionCard, animatedStyle]}>
                <View style={styles.quickActionIcon}>
                    <Ionicons name={icon} size={24} color={ALTA.primary} />
                </View>
                <Text style={styles.quickActionTitle}>{title}</Text>
                <Text style={styles.quickActionSubtitle}>{subtitle}</Text>
            </Animated.View>
        </TouchableOpacity>
    );
};

const AIHubScreen = () => {
    const navigation = useNavigation();
    const [message, setMessage] = useState('');

    const getGreeting = () => {
        const hour = moment().hour();
        if (hour < 12) return 'Good morning';
        if (hour < 18) return 'Good afternoon';
        return 'Good evening';
    };

    const handleSend = () => {
        if (message.trim()) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
            (navigation as any).navigate('AIChat', { initialMessage: message });
            setMessage('');
        }
    };

    const handleGoalPress = (goal: any) => {
        (navigation as any).navigate('AIChat', { initialMessage: goal.text });
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <Animated.View
                    entering={FadeInDown.delay(50).springify()}
                    style={styles.header}
                >
                    <TahoeIconButton
                        icon="menu-outline"
                        onPress={() => (navigation as any).navigate('Profile')}
                        color={ALTA.text}
                    />

                    <Image
                        source={{ uri: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=100' }}
                        style={styles.headerAvatar}
                    />
                </Animated.View>

                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                    keyboardShouldPersistTaps="handled"
                >
                    {/* Hero Section */}
                    <Animated.View
                        entering={FadeIn.delay(100).duration(600)}
                        style={styles.heroSection}
                    >
                        <FloatingAIAvatar />

                        <Text style={styles.heroTitle}>Dress with confidence</Text>
                        <Text style={styles.heroSubtitle}>
                            Your personal AI stylist that truly gets you - from the clothes in your closet to the looks you love.
                        </Text>
                    </Animated.View>

                    {/* Talk to Alta Input */}
                    <Animated.View
                        entering={FadeInUp.delay(150).springify()}
                        style={styles.inputSection}
                    >
                        <TouchableOpacity
                            style={styles.talkButton}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                (navigation as any).navigate('AIChat');
                            }}
                        >
                            <Text style={styles.talkButtonText}>Talk to Alta</Text>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* What's your style goal? */}
                    <Animated.View
                        entering={FadeInUp.delay(180).springify()}
                        style={styles.goalsSection}
                    >
                        <Text style={styles.goalsTitle}>What's your{'\n'}style goal?</Text>

                        {STYLE_GOALS.map((goal, index) => (
                            <StyleGoalButton
                                key={goal.id}
                                goal={goal}
                                index={index}
                                onPress={() => handleGoalPress(goal)}
                            />
                        ))}
                    </Animated.View>

                    {/* Quick Actions Grid */}
                    <Animated.View
                        entering={FadeInUp.delay(400).springify()}
                        style={styles.quickActionsSection}
                    >
                        <Text style={styles.quickActionsTitle}>Get started</Text>

                        <View style={styles.quickActionsGrid}>
                            <QuickActionCard
                                icon="shirt-outline"
                                title="Scan wardrobe"
                                subtitle="Add your clothes"
                                onPress={() => (navigation as any).navigate('WardrobeVideo')}
                            />
                            <QuickActionCard
                                icon="sparkles-outline"
                                title="AI Stylist"
                                subtitle="Smart recommendations"
                                onPress={() => (navigation as any).navigate('OutfitAI')}
                            />
                            <QuickActionCard
                                icon="person-outline"
                                title="Try on looks"
                                subtitle="Virtual fitting"
                                onPress={() => (navigation as any).navigate('AITryOn')}
                            />
                            <QuickActionCard
                                icon="calendar-outline"
                                title="Plan outfits"
                                subtitle="Weekly calendar"
                                onPress={() => (navigation as any).navigate('Calendar')}
                            />
                        </View>
                    </Animated.View>

                    {/* Bottom spacing */}
                    <View style={{ height: 120 }} />
                </ScrollView>

                {/* Floating Input */}
                <KeyboardAvoidingView
                    behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
                    keyboardVerticalOffset={10}
                >
                    <View style={styles.floatingInputWrapper}>
                        <View style={styles.floatingInputContainer}>
                            <TextInput
                                style={styles.textInput}
                                placeholder="Ask anything about style..."
                                placeholderTextColor={ALTA.textMuted}
                                value={message}
                                onChangeText={setMessage}
                                returnKeyType="send"
                                onSubmitEditing={handleSend}
                            />
                            <TouchableOpacity
                                style={[
                                    styles.sendButton,
                                    message.trim() && styles.sendButtonActive
                                ]}
                                onPress={handleSend}
                            >
                                <Ionicons
                                    name="arrow-up"
                                    size={20}
                                    color={message.trim() ? ALTA.background : ALTA.textMuted}
                                />
                            </TouchableOpacity>
                        </View>
                    </View>
                </KeyboardAvoidingView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: ALTA.background,
    },
    safeArea: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 10,
    },
    headerAvatar: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: ALTA.surface,
    },

    scrollContent: {
        paddingHorizontal: 20,
    },

    // Hero Section
    heroSection: {
        alignItems: 'center',
        paddingTop: 20,
        paddingBottom: 30,
    },
    avatarContainer: {
        marginBottom: 24,
    },
    avatarCircle: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: ALTA.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: ALTA.border,
    },
    heroTitle: {
        fontSize: 32,
        fontWeight: '700',
        color: ALTA.text,
        textAlign: 'center',
        marginBottom: 12,
        letterSpacing: -0.5,
    },
    heroSubtitle: {
        fontSize: 16,
        color: ALTA.textSecondary,
        textAlign: 'center',
        lineHeight: 24,
        paddingHorizontal: 20,
    },

    // Input Section
    inputSection: {
        marginBottom: 40,
    },
    talkButton: {
        backgroundColor: ALTA.primary,
        paddingVertical: 16,
        borderRadius: 30,
        alignItems: 'center',
    },
    talkButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: ALTA.background,
    },

    // Goals Section
    goalsSection: {
        marginBottom: 40,
    },
    goalsTitle: {
        fontSize: 28,
        fontWeight: '700',
        color: ALTA.text,
        marginBottom: 20,
        letterSpacing: -0.5,
    },
    goalButton: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 16,
        borderBottomWidth: 1,
        borderBottomColor: ALTA.border,
    },
    goalIcon: {
        marginRight: 14,
    },
    goalText: {
        flex: 1,
        fontSize: 16,
        color: ALTA.text,
    },

    // Quick Actions
    quickActionsSection: {
        marginBottom: 20,
    },
    quickActionsTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: ALTA.text,
        marginBottom: 16,
    },
    quickActionsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 12,
    },
    quickActionCard: {
        width: (width - 52) / 2,
        backgroundColor: ALTA.surfaceLight,
        borderRadius: 16,
        padding: 16,
        borderWidth: 1,
        borderColor: ALTA.border,
    },
    quickActionIcon: {
        width: 44,
        height: 44,
        borderRadius: 12,
        backgroundColor: ALTA.background,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
    },
    quickActionTitle: {
        fontSize: 15,
        fontWeight: '600',
        color: ALTA.text,
        marginBottom: 4,
    },
    quickActionSubtitle: {
        fontSize: 13,
        color: ALTA.textSecondary,
    },

    // Floating Input
    floatingInputWrapper: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: 16,
        paddingBottom: Platform.OS === 'ios' ? 0 : 16,
        backgroundColor: ALTA.background,
        borderTopWidth: 1,
        borderTopColor: ALTA.border,
    },
    floatingInputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: ALTA.surfaceLight,
        borderRadius: 24,
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderWidth: 1,
        borderColor: ALTA.border,
    },
    textInput: {
        flex: 1,
        height: 40,
        fontSize: 16,
        color: ALTA.text,
        marginRight: 10,
    },
    sendButton: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: ALTA.surface,
        alignItems: 'center',
        justifyContent: 'center',
    },
    sendButtonActive: {
        backgroundColor: ALTA.primary,
    },
});

export default AIHubScreen;
