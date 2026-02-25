/**
 * Alta-Style Landing Page
 * 
 * This is the marketing/onboarding screen that matches Alta's homepage
 * 
 * Features:
 * - "Dress with confidence" hero
 * - Feature highlights
 * - Style goal cards
 * - "Talk to Alta" CTA
 * 
 * CUSTOMIZE THIS FOR YOUR APP
 */

import React from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    ScrollView,
    TouchableOpacity,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';

const { width } = Dimensions.get('window');

// Alta monochromatic palette
const ALTA = {
    bg: '#FFFFFF',
    surface: '#F5F5F5',
    text: '#0A1931',
    textSecondary: '#666666',
    textMuted: '#8E8E8E',
    border: '#E5E5E5',
};

// Style Goal Card
const StyleGoalCard = ({ text, icon, delay }: { text: string; icon: string; delay: number }) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    return (
        <Animated.View entering={FadeInUp.delay(delay).springify()}>
            <TouchableOpacity
                activeOpacity={1}
                onPressIn={() => {
                    scale.value = withSpring(0.97);
                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                }}
                onPressOut={() => {
                    scale.value = withSpring(1);
                }}
            >
                <Animated.View style={[styles.goalCard, animatedStyle]}>
                    <Ionicons name={icon as any} size={20} color={ALTA.text} />
                    <Text style={styles.goalText}>{text}</Text>
                    <Ionicons name="arrow-forward" size={16} color={ALTA.textMuted} />
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// Feature Card
const FeatureCard = ({ title, subtitle, icon, delay }: { title: string; subtitle: string; icon: string; delay: number }) => {
    return (
        <Animated.View
            entering={FadeInUp.delay(delay).springify()}
            style={styles.featureCard}
        >
            <View style={styles.featureIcon}>
                <Ionicons name={icon as any} size={24} color={ALTA.text} />
            </View>
            <Text style={styles.featureTitle}>{title}</Text>
            <Text style={styles.featureSubtitle}>{subtitle}</Text>
        </Animated.View>
    );
};

const AltaLandingScreen = () => {
    const navigation = useNavigation();

    const styleGoals = [
        { text: 'Plan my outfits better', icon: 'calendar-outline' },
        { text: 'Look professional at work', icon: 'briefcase-outline' },
        { text: 'Expand my wardrobe', icon: 'add-circle-outline' },
        { text: 'Evolve my style', icon: 'trending-up-outline' },
        { text: 'Wear my clothes more', icon: 'shirt-outline' },
    ];

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {/* Hero Section */}
                    <Animated.View
                        entering={FadeIn.delay(100).duration(600)}
                        style={styles.heroSection}
                    >
                        <Text style={styles.heroTitle}>Dress with confidence</Text>
                        <Text style={styles.heroSubtitle}>
                            Your personal AI stylist that truly gets you - from the clothes in your closet to the looks you love.
                        </Text>

                        <Text style={styles.featuredIn}>
                            Featured in <Text style={styles.featuredBrand}>Vogue</Text>, Alta knows your wardrobe inside out, styling you for{' '}
                            <Text style={styles.italicText}>date nights</Text>,{' '}
                            <Text style={styles.italicText}>job interviews</Text>, and everything in between.
                        </Text>

                        <Text style={styles.tagline}>
                            Never be stressed getting dressed again. With Alta you'll always look your best.
                        </Text>
                    </Animated.View>

                    {/* Talk to Alta Button */}
                    <Animated.View
                        entering={FadeInUp.delay(200).springify()}
                        style={styles.ctaSection}
                    >
                        <TouchableOpacity
                            style={styles.ctaButton}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                (navigation as any).navigate('AIChat');
                            }}
                        >
                            <Text style={styles.ctaText}>Talk to Alta</Text>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* Features */}
                    <View style={styles.featuresSection}>
                        <FeatureCard
                            title="Get the perfect look"
                            subtitle="Tell Alta your plans, Alta picks the ideal outfit. And you can adjust it until it's just right."
                            icon="sparkles-outline"
                            delay={300}
                        />
                        <FeatureCard
                            title="Rediscover your wardrobe"
                            subtitle="See all your clothes in one place."
                            icon="shirt-outline"
                            delay={350}
                        />
                        <FeatureCard
                            title="Try it on your avatar"
                            subtitle="See how to style potential new pieces with your closet."
                            icon="person-outline"
                            delay={400}
                        />
                    </View>

                    {/* Setting the trend */}
                    <Animated.View
                        entering={FadeInUp.delay(450).springify()}
                        style={styles.trendSection}
                    >
                        <Text style={styles.trendTitle}>Setting the trend</Text>
                        <Text style={styles.trendSubtitle}>
                            Join the world's most stylish people using Alta
                        </Text>
                    </Animated.View>

                    {/* Style Goals */}
                    <View style={styles.goalsSection}>
                        <Text style={styles.goalsTitle}>What's your{'\n'}style goal?</Text>

                        {styleGoals.map((goal, index) => (
                            <StyleGoalCard
                                key={goal.text}
                                text={goal.text}
                                icon={goal.icon}
                                delay={500 + index * 50}
                            />
                        ))}
                    </View>

                    {/* Footer */}
                    <View style={styles.footer}>
                        <TouchableOpacity>
                            <Text style={styles.footerLink}>Privacy Policy</Text>
                        </TouchableOpacity>
                        <TouchableOpacity>
                            <Text style={styles.footerLink}>Terms</Text>
                        </TouchableOpacity>
                    </View>

                    <View style={{ height: 50 }} />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: ALTA.bg,
    },
    safeArea: {
        flex: 1,
    },
    scrollContent: {
        paddingHorizontal: 24,
        paddingTop: 40,
    },

    // Hero
    heroSection: {
        marginBottom: 32,
    },
    heroTitle: {
        fontSize: 42,
        fontWeight: '700',
        color: ALTA.text,
        lineHeight: 48,
        marginBottom: 20,
    },
    heroSubtitle: {
        fontSize: 18,
        color: ALTA.textSecondary,
        lineHeight: 26,
        marginBottom: 20,
    },
    featuredIn: {
        fontSize: 16,
        color: ALTA.textSecondary,
        lineHeight: 24,
        marginBottom: 16,
    },
    featuredBrand: {
        fontWeight: '600',
        fontStyle: 'italic',
    },
    italicText: {
        fontStyle: 'italic',
    },
    tagline: {
        fontSize: 16,
        color: ALTA.textSecondary,
        lineHeight: 24,
    },

    // CTA
    ctaSection: {
        marginBottom: 40,
    },
    ctaButton: {
        backgroundColor: ALTA.text,
        paddingVertical: 16,
        borderRadius: 30,
        alignItems: 'center',
    },
    ctaText: {
        fontSize: 17,
        fontWeight: '600',
        color: ALTA.bg,
    },

    // Features
    featuresSection: {
        marginBottom: 40,
    },
    featureCard: {
        marginBottom: 24,
    },
    featureIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: ALTA.surface,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
    },
    featureTitle: {
        fontSize: 20,
        fontWeight: '600',
        color: ALTA.text,
        marginBottom: 6,
    },
    featureSubtitle: {
        fontSize: 15,
        color: ALTA.textSecondary,
        lineHeight: 22,
    },

    // Trend
    trendSection: {
        marginBottom: 32,
    },
    trendTitle: {
        fontSize: 24,
        fontWeight: '600',
        color: ALTA.text,
        marginBottom: 8,
    },
    trendSubtitle: {
        fontSize: 15,
        color: ALTA.textSecondary,
    },

    // Goals
    goalsSection: {
        marginBottom: 40,
    },
    goalsTitle: {
        fontSize: 32,
        fontWeight: '700',
        color: ALTA.text,
        lineHeight: 40,
        marginBottom: 24,
    },
    goalCard: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 16,
        paddingHorizontal: 20,
        backgroundColor: ALTA.surface,
        borderRadius: 16,
        marginBottom: 10,
        gap: 12,
    },
    goalText: {
        flex: 1,
        fontSize: 16,
        fontWeight: '500',
        color: ALTA.text,
    },

    // Footer
    footer: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 24,
    },
    footerLink: {
        fontSize: 14,
        color: ALTA.textMuted,
    },
});

export default AltaLandingScreen;
