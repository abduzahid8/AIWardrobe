import React, { useState, useEffect } from "react";
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Image,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import moment from "moment";
import { useNavigation } from "@react-navigation/native";
import { useTranslation } from "react-i18next";
import * as Haptics from "expo-haptics";
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withRepeat,
    withSequence,
    withTiming,
    withSpring,
    FadeInUp,
    FadeInDown,
    FadeIn,
    Easing,
    interpolate,
} from "react-native-reanimated";

import { useTheme } from "../../theme/ThemeContext";
import { useWardrobeItems } from "../../hooks";

const { width, height } = Dimensions.get("window");

// ==========================================
// 3D FLOATING OUTFIT HERO (Alta-inspired)
// ==========================================
const FloatingOutfitHero = ({ isDark }: { isDark: boolean }) => {
    const floatY = useSharedValue(0);
    const rotate3D = useSharedValue(0);
    const scale = useSharedValue(1);

    useEffect(() => {
        // Floating animation
        floatY.value = withRepeat(
            withSequence(
                withTiming(-12, { duration: 2500, easing: Easing.bezier(0.4, 0, 0.2, 1) }),
                withTiming(0, { duration: 2500, easing: Easing.bezier(0.4, 0, 0.2, 1) })
            ),
            -1,
            true
        );

        // 3D rotation
        rotate3D.value = withRepeat(
            withSequence(
                withTiming(3, { duration: 4000 }),
                withTiming(-3, { duration: 4000 })
            ),
            -1,
            true
        );

        // Subtle pulse
        scale.value = withRepeat(
            withSequence(
                withTiming(1.02, { duration: 2000 }),
                withTiming(1, { duration: 2000 })
            ),
            -1,
            true
        );
    }, []);

    const heroStyle = useAnimatedStyle(() => ({
        transform: [
            { translateY: floatY.value },
            { rotateY: `${rotate3D.value}deg` },
            { scale: scale.value },
        ],
    }));

    return (
        <Animated.View style={[styles.heroCard, heroStyle, { backgroundColor: isDark ? '#60A5FA' : '#DBEAFE' }]}>
            {/* Outfit illustration placeholder */}
            <View style={styles.heroContent}>
                <Ionicons name="shirt" size={80} color={isDark ? '#000' : '#3B82F6'} />
                <Text style={[styles.heroLabel, { color: isDark ? '#000' : '#1E40AF' }]}>
                    Today's Pick
                </Text>
            </View>

            {/* 3D shine effect */}
            <View style={styles.heroShine} />
        </Animated.View>
    );
};

// ==========================================
// STATS CARD (Agrilo-style)
// ==========================================
const StatsCard = ({
    icon,
    value,
    label,
    isAccent = false,
    isDark,
    delay = 0
}: {
    icon: string;
    value: string | number;
    label: string;
    isAccent?: boolean;
    isDark: boolean;
    delay?: number;
}) => {
    const scale = useSharedValue(1);

    const animStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    const bgColor = isAccent
        ? (isDark ? '#60A5FA' : '#DBEAFE')
        : (isDark ? '#252525' : '#F8FAFC');

    const textColor = isAccent
        ? (isDark ? '#000' : '#1E40AF')
        : (isDark ? '#fff' : '#000');

    const secondaryColor = isAccent
        ? (isDark ? '#1E3A5F' : '#3B82F6')
        : (isDark ? '#9CA3AF' : '#6B7280');

    return (
        <Animated.View entering={FadeInUp.delay(delay).springify()}>
            <TouchableOpacity
                onPressIn={() => { scale.value = withSpring(0.95); }}
                onPressOut={() => { scale.value = withSpring(1); }}
                activeOpacity={1}
            >
                <Animated.View style={[styles.statsCard, animStyle, { backgroundColor: bgColor }]}>
                    <View style={[styles.statsIconBadge, { backgroundColor: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.05)' }]}>
                        <Ionicons name={icon as any} size={20} color={secondaryColor} />
                    </View>
                    <Text style={[styles.statsValue, { color: textColor }]}>{value}</Text>
                    <Text style={[styles.statsLabel, { color: secondaryColor }]}>{label}</Text>
                </Animated.View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// ==========================================
// RECENT ACTIVITY ITEM
// ==========================================
const ActivityItem = ({
    title,
    subtitle,
    isDark,
    delay = 0
}: {
    title: string;
    subtitle: string;
    isDark: boolean;
    delay?: number;
}) => (
    <Animated.View entering={FadeInUp.delay(delay).springify()}>
        <TouchableOpacity
            style={[styles.activityItem, { backgroundColor: isDark ? '#252525' : '#F8FAFC' }]}
            activeOpacity={0.7}
        >
            <View style={styles.activityContent}>
                <Text style={[styles.activityTitle, { color: isDark ? '#fff' : '#000' }]}>{title}</Text>
                <Text style={[styles.activitySubtitle, { color: isDark ? '#9CA3AF' : '#6B7280' }]}>{subtitle}</Text>
            </View>
            <Ionicons name="chevron-forward" size={20} color={isDark ? '#6B7280' : '#9CA3AF'} />
        </TouchableOpacity>
    </Animated.View>
);

// ==========================================
// MAIN COMPONENT
// ==========================================
const DailyBriefScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const { isDark, colors, toggleTheme } = useTheme();
    const [activeTab, setActiveTab] = useState<'overview' | 'suggestions'>('overview');

    const { itemCount } = useWardrobeItems({ includePopularItems: false });

    const getGreeting = () => {
        const hour = moment().hour();
        if (hour < 12) return 'Good Morning';
        if (hour < 18) return 'Good Afternoon';
        return 'Good Evening';
    };

    const userName = 'David'; // Replace with actual user name

    return (
        <View style={[styles.container, { backgroundColor: colors.background }]}>
            <SafeAreaView style={styles.safeArea}>
                <ScrollView
                    showsVerticalScrollIndicator={false}
                    contentContainerStyle={styles.scrollContent}
                >
                    {/* Header - Agrilo Style */}
                    <Animated.View
                        entering={FadeInDown.delay(50).springify()}
                        style={styles.header}
                    >
                        <View style={styles.greetingContainer}>
                            <Text style={[styles.greetingSmall, { color: colors.text.secondary }]}>
                                Hello {userName},
                            </Text>
                            <Text style={[styles.greetingLarge, { color: colors.text.primary }]}>
                                {getGreeting()}!
                            </Text>
                        </View>

                        <TouchableOpacity
                            style={[styles.notificationButton, { backgroundColor: isDark ? '#252525' : '#F8FAFC' }]}
                            onPress={() => (navigation as any).navigate('Notifications')}
                        >
                            <Ionicons
                                name="notifications"
                                size={20}
                                color={isDark ? '#FBBF24' : '#F59E0B'}
                            />
                        </TouchableOpacity>
                    </Animated.View>

                    {/* Tab Pills - Agrilo Style */}
                    <Animated.View
                        entering={FadeInDown.delay(100).springify()}
                        style={styles.tabContainer}
                    >
                        <TouchableOpacity
                            style={[
                                styles.tabPill,
                                activeTab === 'overview' && styles.tabPillActive,
                                {
                                    backgroundColor: activeTab === 'overview'
                                        ? (isDark ? '#fff' : '#000')
                                        : (isDark ? '#252525' : '#E5E7EB')
                                }
                            ]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setActiveTab('overview');
                            }}
                        >
                            <Text style={[
                                styles.tabPillText,
                                {
                                    color: activeTab === 'overview'
                                        ? (isDark ? '#000' : '#fff')
                                        : (isDark ? '#9CA3AF' : '#6B7280')
                                }
                            ]}>
                                Wardrobe Overview
                            </Text>
                        </TouchableOpacity>

                        <TouchableOpacity
                            style={[
                                styles.tabPill,
                                activeTab === 'suggestions' && styles.tabPillActive,
                                {
                                    backgroundColor: activeTab === 'suggestions'
                                        ? (isDark ? '#fff' : '#000')
                                        : (isDark ? '#252525' : '#E5E7EB')
                                }
                            ]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setActiveTab('suggestions');
                            }}
                        >
                            <Text style={[
                                styles.tabPillText,
                                {
                                    color: activeTab === 'suggestions'
                                        ? (isDark ? '#000' : '#fff')
                                        : (isDark ? '#9CA3AF' : '#6B7280')
                                }
                            ]}>
                                AI Suggestions
                            </Text>
                        </TouchableOpacity>
                    </Animated.View>

                    {/* 3D Floating Hero Card - Alta Style */}
                    <Animated.View
                        entering={FadeIn.delay(150).duration(800)}
                        style={styles.heroContainer}
                    >
                        <FloatingOutfitHero isDark={isDark} />
                    </Animated.View>

                    {/* Stats Grid - Agrilo Style */}
                    <View style={styles.statsGrid}>
                        <StatsCard
                            icon="flask"
                            value={itemCount}
                            label="Total Items"
                            isDark={isDark}
                            delay={200}
                        />
                        <StatsCard
                            icon="happy"
                            value="85%"
                            label="Style Match"
                            isAccent={true}
                            isDark={isDark}
                            delay={250}
                        />
                    </View>

                    {/* Recent Activity - Agrilo Style */}
                    <Animated.View
                        entering={FadeInUp.delay(300).springify()}
                        style={styles.sectionHeader}
                    >
                        <Text style={[styles.sectionTitle, { color: colors.text.primary }]}>
                            Recent Activity
                        </Text>
                        <TouchableOpacity>
                            <Text style={[styles.seeAll, { color: colors.text.secondary }]}>
                                See all
                            </Text>
                        </TouchableOpacity>
                    </Animated.View>

                    <ActivityItem
                        title="New outfit created"
                        subtitle="3 Days ago"
                        isDark={isDark}
                        delay={350}
                    />
                    <ActivityItem
                        title="Added 2 items to closet"
                        subtitle="5 Days ago"
                        isDark={isDark}
                        delay={400}
                    />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    safeArea: {
        flex: 1,
    },
    scrollContent: {
        paddingBottom: 100,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'flex-start',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingTop: 16,
        paddingBottom: 8,
    },
    greetingContainer: {
        flex: 1,
    },
    greetingSmall: {
        fontSize: 16,
        marginBottom: 4,
    },
    greetingLarge: {
        fontSize: 28,
        fontWeight: '700',
    },
    notificationButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        alignItems: 'center',
        justifyContent: 'center',
    },
    tabContainer: {
        flexDirection: 'row',
        paddingHorizontal: 20,
        paddingVertical: 16,
        gap: 12,
    },
    tabPill: {
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
    },
    tabPillActive: {},
    tabPillText: {
        fontSize: 14,
        fontWeight: '500',
    },
    heroContainer: {
        alignItems: 'center',
        paddingVertical: 20,
    },
    heroCard: {
        width: width - 40,
        height: 280,
        borderRadius: 24,
        overflow: 'hidden',
        alignItems: 'center',
        justifyContent: 'center',
    },
    heroContent: {
        alignItems: 'center',
    },
    heroLabel: {
        fontSize: 18,
        fontWeight: '600',
        marginTop: 12,
    },
    heroShine: {
        position: 'absolute',
        top: -50,
        right: -50,
        width: 200,
        height: 200,
        borderRadius: 100,
        backgroundColor: 'rgba(255,255,255,0.15)',
    },
    statsGrid: {
        flexDirection: 'row',
        paddingHorizontal: 20,
        gap: 12,
        marginTop: 8,
    },
    statsCard: {
        flex: 1,
        padding: 16,
        borderRadius: 20,
    },
    statsIconBadge: {
        width: 36,
        height: 36,
        borderRadius: 18,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 12,
    },
    statsValue: {
        fontSize: 32,
        fontWeight: '300',
    },
    statsLabel: {
        fontSize: 14,
        marginTop: 4,
    },
    sectionHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        marginTop: 32,
        marginBottom: 16,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
    },
    seeAll: {
        fontSize: 14,
    },
    activityItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginHorizontal: 20,
        padding: 16,
        borderRadius: 16,
        marginBottom: 12,
    },
    activityContent: {
        flex: 1,
    },
    activityTitle: {
        fontSize: 16,
        fontWeight: '500',
        marginBottom: 4,
    },
    activitySubtitle: {
        fontSize: 14,
    },
});

export default DailyBriefScreen;
