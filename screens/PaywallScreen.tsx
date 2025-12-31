import React, { useState } from "react";
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Platform,
    ActivityIndicator,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { LinearGradient } from "expo-linear-gradient";
import { useNavigation } from "@react-navigation/native";
import * as Haptics from "expo-haptics";
import Animated, {
    FadeIn,
    FadeInUp,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withTiming,
} from "react-native-reanimated";
import useSubscriptionStore, { SUBSCRIPTION_PRICING, TIER_FEATURES } from "../store/subscriptionStore";
import AppColors from "../constants/AppColors";

const { width, height } = Dimensions.get("window");

// Use unified AppColors
const COLORS = {
    background: '#0A0A0A',
    surface: '#1A1A1A',
    surfaceLight: '#2A2A2A',
    text: '#FFFFFF',
    textSecondary: AppColors.textMuted,
    premium: AppColors.premium,
    premiumDark: AppColors.premiumDark,
    vip: AppColors.vip,
    vipDark: AppColors.vipDark,
    accent: AppColors.accent,
    success: AppColors.success,
    border: '#333333',
};

// Subscription Card Component
const SubscriptionCard = ({
    tier,
    price,
    period,
    features,
    isPopular,
    gradient,
    onSelect,
    isLoading,
}: {
    tier: string;
    price: number;
    period: string;
    features: string[];
    isPopular?: boolean;
    gradient: string[];
    onSelect: () => void;
    isLoading: boolean;
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 15, stiffness: 400 }) }],
    }));

    const handlePressIn = () => {
        scale.value = 0.98;
    };

    const handlePressOut = () => {
        scale.value = 1;
    };

    return (
        <TouchableOpacity
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={() => {
                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                onSelect();
            }}
            activeOpacity={1}
        >
            <Animated.View style={[styles.card, animatedStyle]}>
                <LinearGradient
                    colors={gradient as any}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.cardGradient}
                >
                    {/* Popular Badge */}
                    {isPopular && (
                        <View style={styles.popularBadge}>
                            <Text style={styles.popularBadgeText}>MOST POPULAR</Text>
                        </View>
                    )}

                    {/* Tier Name */}
                    <Text style={styles.tierName}>{tier}</Text>

                    {/* Price */}
                    <View style={styles.priceContainer}>
                        <Text style={styles.currencySign}>$</Text>
                        <Text style={styles.priceAmount}>{price.toFixed(2)}</Text>
                        <Text style={styles.pricePeriod}>/{period}</Text>
                    </View>

                    {/* Features */}
                    <View style={styles.featuresContainer}>
                        {features.map((feature, index) => (
                            <View key={index} style={styles.featureRow}>
                                <Ionicons name="checkmark-circle" size={18} color={COLORS.success} />
                                <Text style={styles.featureText}>{feature}</Text>
                            </View>
                        ))}
                    </View>

                    {/* CTA Button */}
                    <View style={styles.ctaButton}>
                        {isLoading ? (
                            <ActivityIndicator color={gradient[0]} />
                        ) : (
                            <Text style={[styles.ctaText, { color: gradient[0] }]}>
                                Get {tier}
                            </Text>
                        )}
                    </View>
                </LinearGradient>
            </Animated.View>
        </TouchableOpacity>
    );
};

// Main Paywall Screen
const PaywallScreen = () => {
    const navigation = useNavigation();
    const { setSubscription } = useSubscriptionStore();
    const [isLoading, setIsLoading] = useState<string | null>(null);

    const handlePurchase = async (tier: 'premium' | 'vip') => {
        setIsLoading(tier);
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);

        // Simulate purchase (replace with actual IAP logic)
        setTimeout(async () => {
            await setSubscription(tier);
            setIsLoading(null);
            navigation.goBack();
        }, 2000);
    };

    const handleRestore = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        // TODO: Implement restore purchases
        console.log('Restore purchases');
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#0A0A0A', '#1A1A2E', '#16213E']}
                style={styles.gradient}
            >
                <SafeAreaView style={styles.safeArea}>
                    {/* Header */}
                    <View style={styles.header}>
                        <TouchableOpacity
                            style={styles.closeButton}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                navigation.goBack();
                            }}
                        >
                            <Ionicons name="close" size={28} color={COLORS.text} />
                        </TouchableOpacity>
                    </View>

                    <ScrollView
                        contentContainerStyle={styles.scrollContent}
                        showsVerticalScrollIndicator={false}
                    >
                        {/* Hero */}
                        <Animated.View entering={FadeIn.duration(600)} style={styles.hero}>
                            <View style={styles.iconContainer}>
                                <Ionicons name="sparkles" size={48} color={COLORS.premium} />
                            </View>
                            <Text style={styles.heroTitle}>Unlock Premium</Text>
                            <Text style={styles.heroSubtitle}>
                                Get unlimited access to all AI features and take your style to the next level
                            </Text>
                        </Animated.View>

                        {/* Subscription Cards */}
                        <View style={styles.cardsContainer}>
                            {/* Premium Card */}
                            <Animated.View entering={FadeInUp.delay(200).springify()}>
                                <SubscriptionCard
                                    tier="Premium"
                                    price={SUBSCRIPTION_PRICING.premium.price}
                                    period="month"
                                    features={[
                                        "Unlimited AI Outfits",
                                        "Unlimited Wardrobe Scans",
                                        "50 Virtual Try-Ons/month",
                                        "Wardrobe Analytics",
                                        "Ad-free Experience",
                                    ]}
                                    isPopular
                                    gradient={[COLORS.premium, COLORS.premiumDark]}
                                    onSelect={() => handlePurchase('premium')}
                                    isLoading={isLoading === 'premium'}
                                />
                            </Animated.View>

                            {/* VIP Card */}
                            <Animated.View entering={FadeInUp.delay(300).springify()}>
                                <SubscriptionCard
                                    tier="VIP"
                                    price={SUBSCRIPTION_PRICING.vip.price}
                                    period="year"
                                    features={[
                                        "Everything in Premium",
                                        "Unlimited Virtual Try-Ons",
                                        "Priority Support",
                                        "Unlimited Cloud Storage",
                                        "Early Access Features",
                                        "Exclusive VIP Badge",
                                    ]}
                                    gradient={[COLORS.vip, COLORS.vipDark]}
                                    onSelect={() => handlePurchase('vip')}
                                    isLoading={isLoading === 'vip'}
                                />
                            </Animated.View>
                        </View>

                        {/* Restore Purchases */}
                        <TouchableOpacity style={styles.restoreButton} onPress={handleRestore}>
                            <Text style={styles.restoreText}>Restore Purchases</Text>
                        </TouchableOpacity>

                        {/* Terms */}
                        <Text style={styles.termsText}>
                            Payment will be charged to your Apple ID account at confirmation of purchase.
                            Subscription automatically renews unless canceled at least 24 hours before the end of the current period.
                        </Text>

                        <View style={{ height: 40 }} />
                    </ScrollView>
                </SafeAreaView>
            </LinearGradient>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    gradient: {
        flex: 1,
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        paddingHorizontal: 20,
        paddingVertical: 10,
    },
    closeButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: COLORS.surfaceLight,
        alignItems: 'center',
        justifyContent: 'center',
    },
    scrollContent: {
        paddingHorizontal: 20,
    },

    // Hero
    hero: {
        alignItems: 'center',
        marginBottom: 32,
    },
    iconContainer: {
        width: 90,
        height: 90,
        borderRadius: 45,
        backgroundColor: 'rgba(255, 215, 0, 0.15)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 20,
    },
    heroTitle: {
        fontSize: 32,
        fontWeight: '800',
        color: COLORS.text,
        marginBottom: 12,
        textAlign: 'center',
    },
    heroSubtitle: {
        fontSize: 16,
        color: COLORS.textSecondary,
        textAlign: 'center',
        lineHeight: 24,
        paddingHorizontal: 20,
    },

    // Cards
    cardsContainer: {
        gap: 16,
        marginBottom: 24,
    },
    card: {
        borderRadius: 20,
        overflow: 'hidden',
        ...Platform.select({
            ios: {
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 8 },
                shadowOpacity: 0.3,
                shadowRadius: 16,
            },
            android: {
                elevation: 8,
            },
        }),
    },
    cardGradient: {
        padding: 24,
        position: 'relative',
    },
    popularBadge: {
        position: 'absolute',
        top: 12,
        right: 12,
        backgroundColor: 'rgba(0,0,0,0.3)',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 10,
    },
    popularBadgeText: {
        fontSize: 10,
        fontWeight: '700',
        color: COLORS.text,
        letterSpacing: 0.5,
    },
    tierName: {
        fontSize: 24,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 8,
    },
    priceContainer: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        marginBottom: 20,
    },
    currencySign: {
        fontSize: 18,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: 6,
    },
    priceAmount: {
        fontSize: 42,
        fontWeight: '800',
        color: COLORS.text,
    },
    pricePeriod: {
        fontSize: 16,
        color: 'rgba(255,255,255,0.7)',
        marginBottom: 8,
        marginLeft: 4,
    },
    featuresContainer: {
        marginBottom: 20,
    },
    featureRow: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 10,
        gap: 10,
    },
    featureText: {
        fontSize: 14,
        color: COLORS.text,
        fontWeight: '500',
    },
    ctaButton: {
        backgroundColor: COLORS.text,
        paddingVertical: 14,
        borderRadius: 12,
        alignItems: 'center',
    },
    ctaText: {
        fontSize: 16,
        fontWeight: '700',
    },

    // Restore
    restoreButton: {
        alignItems: 'center',
        paddingVertical: 16,
    },
    restoreText: {
        fontSize: 14,
        color: COLORS.accent,
        fontWeight: '600',
    },

    // Terms
    termsText: {
        fontSize: 11,
        color: COLORS.textSecondary,
        textAlign: 'center',
        lineHeight: 16,
        paddingHorizontal: 20,
    },
});

export default PaywallScreen;
