import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    ScrollView,
    TouchableOpacity,
    Image,
    StyleSheet,
    Dimensions,
    RefreshControl,
    ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withRepeat,
    withSequence,
    withTiming,
    FadeInDown,
    FadeInRight,
} from 'react-native-reanimated';
import { BlurView } from 'expo-blur';

import AppColors from '../constants/AppColors';
import flashSalesService from '../src/services/flashSalesService';
import { FlashSaleEvent } from '../src/types/flashSales';
import { RootStackParamList } from '../navigation/types';

const { width } = Dimensions.get('window');

// ============================================
// COUNTDOWN TIMER COMPONENT
// ============================================
const CountdownTimer = ({
    event,
    compact = false
}: {
    event: FlashSaleEvent;
    compact?: boolean
}) => {
    const [timeRemaining, setTimeRemaining] = useState(
        flashSalesService.getTimeRemaining(event)
    );

    useEffect(() => {
        const interval = setInterval(() => {
            setTimeRemaining(flashSalesService.getTimeRemaining(event));
        }, 1000);

        return () => clearInterval(interval);
    }, [event]);

    const { hours, minutes, seconds, isEnding } = timeRemaining;
    const label = event.status === 'upcoming' ? 'Starts in' : 'Ends in';

    if (compact) {
        return (
            <View style={[styles.countdownCompact, isEnding && styles.countdownEnding]}>
                <Ionicons
                    name="time-outline"
                    size={12}
                    color={isEnding ? '#FF3B30' : '#FFF'}
                />
                <Text style={[styles.countdownCompactText, isEnding && styles.countdownEndingText]}>
                    {hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m ${seconds}s`}
                </Text>
            </View>
        );
    }

    return (
        <View style={styles.countdownContainer}>
            <Text style={styles.countdownLabel}>{label}</Text>
            <View style={styles.countdownDigits}>
                <View style={styles.countdownBlock}>
                    <Text style={styles.countdownNumber}>{String(hours).padStart(2, '0')}</Text>
                    <Text style={styles.countdownUnit}>HRS</Text>
                </View>
                <Text style={styles.countdownSeparator}>:</Text>
                <View style={styles.countdownBlock}>
                    <Text style={styles.countdownNumber}>{String(minutes).padStart(2, '0')}</Text>
                    <Text style={styles.countdownUnit}>MIN</Text>
                </View>
                <Text style={styles.countdownSeparator}>:</Text>
                <View style={styles.countdownBlock}>
                    <Text style={[styles.countdownNumber, isEnding && styles.countdownEndingText]}>
                        {String(seconds).padStart(2, '0')}
                    </Text>
                    <Text style={styles.countdownUnit}>SEC</Text>
                </View>
            </View>
        </View>
    );
};

// ============================================
// HERO EVENT CARD
// ============================================
const HeroEventCard = ({
    event,
    onPress
}: {
    event: FlashSaleEvent;
    onPress: () => void
}) => {
    const pulseAnim = useSharedValue(1);

    useEffect(() => {
        pulseAnim.value = withRepeat(
            withSequence(
                withTiming(1.05, { duration: 1000 }),
                withTiming(1, { duration: 1000 })
            ),
            -1,
            true
        );
    }, []);

    const pulseStyle = useAnimatedStyle(() => ({
        transform: [{ scale: pulseAnim.value }],
    }));

    return (
        <Animated.View entering={FadeInDown.duration(600).springify()}>
            <TouchableOpacity
                style={styles.heroCard}
                onPress={onPress}
                activeOpacity={0.9}
            >
                <Image
                    source={{ uri: event.heroImage }}
                    style={styles.heroImage}
                />
                <LinearGradient
                    colors={['transparent', 'rgba(0,0,0,0.8)']}
                    style={styles.heroGradient}
                />

                {/* Live badge */}
                {event.status === 'active' && (
                    <Animated.View style={[styles.liveBadge, pulseStyle]}>
                        <View style={styles.liveDot} />
                        <Text style={styles.liveText}>LIVE</Text>
                    </Animated.View>
                )}

                {/* Exclusive badge */}
                {event.isExclusive && (
                    <View style={styles.exclusiveBadge}>
                        <Ionicons name="diamond" size={12} color="#FFD700" />
                        <Text style={styles.exclusiveText}>EXCLUSIVE</Text>
                    </View>
                )}

                <View style={styles.heroContent}>
                    <View style={styles.heroBrandRow}>
                        {event.brandLogo && (
                            <Image
                                source={{ uri: event.brandLogo }}
                                style={styles.heroBrandLogo}
                                resizeMode="contain"
                            />
                        )}
                        <Text style={styles.heroDiscount}>Up to {event.discountPercentage}% OFF</Text>
                    </View>

                    <Text style={styles.heroTitle}>{event.title}</Text>
                    <Text style={styles.heroDescription} numberOfLines={2}>
                        {event.description}
                    </Text>

                    <CountdownTimer event={event} />

                    <View style={styles.heroFooter}>
                        <Text style={styles.heroItemCount}>
                            {event.itemCount} exclusive pieces
                        </Text>
                        <View style={styles.heroShopButton}>
                            <Text style={styles.heroShopButtonText}>Shop Now</Text>
                            <Ionicons name="arrow-forward" size={16} color="#0A1931" />
                        </View>
                    </View>
                </View>
            </TouchableOpacity>
        </Animated.View>
    );
};

// ============================================
// EVENT CARD (Grid/List)
// ============================================
const EventCard = ({
    event,
    onPress,
    index,
}: {
    event: FlashSaleEvent;
    onPress: () => void;
    index: number;
}) => {
    const [isSubscribed, setIsSubscribed] = useState(
        flashSalesService.isSubscribed(event.id)
    );

    const handleSubscribe = async () => {
        if (isSubscribed) {
            await flashSalesService.unsubscribeFromEvent(event.id);
        } else {
            await flashSalesService.subscribeToEvent(event.id);
        }
        setIsSubscribed(!isSubscribed);
    };

    return (
        <Animated.View entering={FadeInRight.delay(index * 100).duration(400)}>
            <TouchableOpacity
                style={styles.eventCard}
                onPress={onPress}
                activeOpacity={0.9}
            >
                <Image
                    source={{ uri: event.heroImage }}
                    style={styles.eventCardImage}
                />
                <LinearGradient
                    colors={['transparent', 'rgba(0,0,0,0.7)']}
                    style={styles.eventCardGradient}
                />

                {/* Discount badge */}
                <View style={styles.discountBadge}>
                    <Text style={styles.discountBadgeText}>-{event.discountPercentage}%</Text>
                </View>

                {/* Countdown */}
                <View style={styles.eventCardCountdown}>
                    <CountdownTimer event={event} compact />
                </View>

                <View style={styles.eventCardContent}>
                    <Text style={styles.eventCardBrand}>{event.brand}</Text>
                    <Text style={styles.eventCardTitle} numberOfLines={1}>
                        {event.title}
                    </Text>
                    <Text style={styles.eventCardItems}>
                        {event.itemCount} items
                    </Text>
                </View>

                {/* Notify button for upcoming */}
                {event.status === 'upcoming' && (
                    <TouchableOpacity
                        style={[
                            styles.notifyButton,
                            isSubscribed && styles.notifyButtonActive
                        ]}
                        onPress={handleSubscribe}
                    >
                        <Ionicons
                            name={isSubscribed ? "notifications" : "notifications-outline"}
                            size={14}
                            color={isSubscribed ? "#FFD700" : "#FFF"}
                        />
                    </TouchableOpacity>
                )}
            </TouchableOpacity>
        </Animated.View>
    );
};

// ============================================
// MAIN SCREEN
// ============================================
const FlashSalesScreen = () => {
    const navigation = useNavigation<NativeStackNavigationProp<RootStackParamList>>();

    const [activeEvents, setActiveEvents] = useState<FlashSaleEvent[]>([]);
    const [upcomingEvents, setUpcomingEvents] = useState<FlashSaleEvent[]>([]);
    const [featuredEvent, setFeaturedEvent] = useState<FlashSaleEvent | null>(null);
    const [loading, setLoading] = useState(true);
    const [refreshing, setRefreshing] = useState(false);

    const loadEvents = async () => {
        try {
            const [active, upcoming, featured] = await Promise.all([
                flashSalesService.getActiveEvents(),
                flashSalesService.getUpcomingEvents(),
                flashSalesService.getFeaturedEvent(),
            ]);

            setActiveEvents(active);
            setUpcomingEvents(upcoming);
            setFeaturedEvent(featured);
        } catch (error) {
            console.error('Failed to load flash sales:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadEvents();
    }, []);

    const onRefresh = useCallback(async () => {
        setRefreshing(true);
        await loadEvents();
        setRefreshing(false);
    }, []);

    const navigateToEvent = (event: FlashSaleEvent) => {
        navigation.navigate('FlashSaleEvent', { eventId: event.id });
    };

    if (loading) {
        return (
            <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color={AppColors.accent} />
                <Text style={styles.loadingText}>Loading exclusive sales...</Text>
            </View>
        );
    }

    return (
        <SafeAreaView style={styles.container} edges={['top']}>
            {/* Header */}
            <View style={styles.header}>
                <TouchableOpacity
                    style={styles.backButton}
                    onPress={() => navigation.goBack()}
                >
                    <Ionicons name="chevron-back" size={24} color={AppColors.text} />
                </TouchableOpacity>
                <View style={styles.headerTitleContainer}>
                    <Text style={styles.headerTitle}>Flash Sales</Text>
                    <View style={styles.headerBadge}>
                        <Text style={styles.headerBadgeText}>🔥 LIVE</Text>
                    </View>
                </View>
                <TouchableOpacity style={styles.filterButton}>
                    <Ionicons name="options-outline" size={22} color={AppColors.text} />
                </TouchableOpacity>
            </View>

            <ScrollView
                style={styles.scrollView}
                contentContainerStyle={styles.scrollContent}
                showsVerticalScrollIndicator={false}
                refreshControl={
                    <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
                }
            >
                {/* Hero Featured Event */}
                {featuredEvent && (
                    <View style={styles.section}>
                        <HeroEventCard
                            event={featuredEvent}
                            onPress={() => navigateToEvent(featuredEvent)}
                        />
                    </View>
                )}

                {/* Active Sales */}
                {activeEvents.length > 0 && (
                    <View style={styles.section}>
                        <View style={styles.sectionHeader}>
                            <Text style={styles.sectionTitle}>⚡ Happening Now</Text>
                            <Text style={styles.sectionSubtitle}>
                                {activeEvents.length} active {activeEvents.length === 1 ? 'sale' : 'sales'}
                            </Text>
                        </View>
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.horizontalScroll}
                        >
                            {activeEvents.map((event, index) => (
                                <EventCard
                                    key={event.id}
                                    event={event}
                                    index={index}
                                    onPress={() => navigateToEvent(event)}
                                />
                            ))}
                        </ScrollView>
                    </View>
                )}

                {/* Upcoming Sales */}
                {upcomingEvents.length > 0 && (
                    <View style={styles.section}>
                        <View style={styles.sectionHeader}>
                            <Text style={styles.sectionTitle}>🗓️ Coming Soon</Text>
                            <Text style={styles.sectionSubtitle}>
                                Don't miss out - get notified
                            </Text>
                        </View>
                        <ScrollView
                            horizontal
                            showsHorizontalScrollIndicator={false}
                            contentContainerStyle={styles.horizontalScroll}
                        >
                            {upcomingEvents.map((event, index) => (
                                <EventCard
                                    key={event.id}
                                    event={event}
                                    index={index}
                                    onPress={() => navigateToEvent(event)}
                                />
                            ))}
                        </ScrollView>
                    </View>
                )}

                {/* Info Section */}
                <View style={styles.infoSection}>
                    <LinearGradient
                        colors={['#F8F8F8', '#FFFFFF']}
                        style={styles.infoCard}
                    >
                        <View style={styles.infoIcon}>
                            <Ionicons name="diamond-outline" size={28} color={AppColors.accent} />
                        </View>
                        <Text style={styles.infoTitle}>Exclusive Access</Text>
                        <Text style={styles.infoText}>
                            Premium brands offer their excess inventory at exclusive prices.
                            Limited time, limited stock – first come, first served.
                        </Text>
                    </LinearGradient>
                </View>

                {/* Bottom spacing */}
                <View style={{ height: 100 }} />
            </ScrollView>
        </SafeAreaView>
    );
};

// ============================================
// STYLES
// ============================================
const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: AppColors.background,
    },
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: AppColors.background,
    },
    loadingText: {
        marginTop: 12,
        fontSize: 14,
        color: AppColors.textSecondary,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingVertical: 12,
        borderBottomWidth: 1,
        borderBottomColor: AppColors.border,
    },
    backButton: {
        width: 40,
        height: 40,
        justifyContent: 'center',
        alignItems: 'center',
    },
    headerTitleContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 8,
    },
    headerTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: AppColors.text,
    },
    headerBadge: {
        backgroundColor: '#FF3B30',
        paddingHorizontal: 8,
        paddingVertical: 3,
        borderRadius: 12,
    },
    headerBadgeText: {
        color: '#FFF',
        fontSize: 10,
        fontWeight: '700',
    },
    filterButton: {
        width: 40,
        height: 40,
        justifyContent: 'center',
        alignItems: 'center',
    },

    // Scroll
    scrollView: {
        flex: 1,
    },
    scrollContent: {
        paddingTop: 16,
    },

    // Sections
    section: {
        marginBottom: 28,
    },
    sectionHeader: {
        paddingHorizontal: 20,
        marginBottom: 16,
    },
    sectionTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: AppColors.text,
        marginBottom: 4,
    },
    sectionSubtitle: {
        fontSize: 13,
        color: AppColors.textSecondary,
    },
    horizontalScroll: {
        paddingHorizontal: 20,
        gap: 16,
    },

    // Hero Card
    heroCard: {
        marginHorizontal: 20,
        height: 400,
        borderRadius: 24,
        overflow: 'hidden',
        backgroundColor: '#0A1931',
    },
    heroImage: {
        width: '100%',
        height: '100%',
        position: 'absolute',
    },
    heroGradient: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '70%',
    },
    liveBadge: {
        position: 'absolute',
        top: 16,
        left: 16,
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FF3B30',
        paddingHorizontal: 10,
        paddingVertical: 5,
        borderRadius: 12,
        gap: 4,
    },
    liveDot: {
        width: 6,
        height: 6,
        borderRadius: 3,
        backgroundColor: '#FFF',
    },
    liveText: {
        color: '#FFF',
        fontSize: 11,
        fontWeight: '700',
        letterSpacing: 1,
    },
    exclusiveBadge: {
        position: 'absolute',
        top: 16,
        right: 16,
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(0,0,0,0.6)',
        paddingHorizontal: 10,
        paddingVertical: 5,
        borderRadius: 12,
        gap: 4,
    },
    exclusiveText: {
        color: '#FFD700',
        fontSize: 10,
        fontWeight: '700',
        letterSpacing: 1,
    },
    heroContent: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: 20,
    },
    heroBrandRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
        marginBottom: 8,
    },
    heroBrandLogo: {
        width: 80,
        height: 24,
        tintColor: '#FFF',
    },
    heroDiscount: {
        fontSize: 14,
        fontWeight: '700',
        color: '#FFD700',
        letterSpacing: 0.5,
    },
    heroTitle: {
        fontSize: 26,
        fontWeight: '800',
        color: '#FFF',
        marginBottom: 6,
    },
    heroDescription: {
        fontSize: 14,
        color: 'rgba(255,255,255,0.8)',
        lineHeight: 20,
        marginBottom: 16,
    },
    heroFooter: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginTop: 16,
    },
    heroItemCount: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.7)',
    },
    heroShopButton: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#FFF',
        paddingHorizontal: 20,
        paddingVertical: 12,
        borderRadius: 24,
        gap: 6,
    },
    heroShopButtonText: {
        fontSize: 14,
        fontWeight: '700',
        color: '#0A1931',
    },

    // Countdown
    countdownContainer: {
        marginTop: 8,
    },
    countdownLabel: {
        fontSize: 11,
        color: 'rgba(255,255,255,0.6)',
        marginBottom: 8,
        letterSpacing: 1,
        textTransform: 'uppercase',
    },
    countdownDigits: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
    },
    countdownBlock: {
        alignItems: 'center',
    },
    countdownNumber: {
        fontSize: 28,
        fontWeight: '700',
        color: '#FFF',
        fontVariant: ['tabular-nums'],
    },
    countdownUnit: {
        fontSize: 9,
        color: 'rgba(255,255,255,0.5)',
        letterSpacing: 1,
        marginTop: 2,
    },
    countdownSeparator: {
        fontSize: 24,
        fontWeight: '300',
        color: 'rgba(255,255,255,0.5)',
        marginHorizontal: 2,
    },
    countdownCompact: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(0,0,0,0.5)',
        paddingHorizontal: 8,
        paddingVertical: 4,
        borderRadius: 10,
        gap: 4,
    },
    countdownCompactText: {
        fontSize: 11,
        fontWeight: '600',
        color: '#FFF',
    },
    countdownEnding: {
        backgroundColor: 'rgba(255,59,48,0.2)',
    },
    countdownEndingText: {
        color: '#FF3B30',
    },

    // Event Card
    eventCard: {
        width: width * 0.55,
        height: 260,
        borderRadius: 20,
        overflow: 'hidden',
        backgroundColor: '#0A1931',
    },
    eventCardImage: {
        width: '100%',
        height: '100%',
        position: 'absolute',
    },
    eventCardGradient: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        height: '60%',
    },
    discountBadge: {
        position: 'absolute',
        top: 12,
        left: 12,
        backgroundColor: '#FF3B30',
        paddingHorizontal: 10,
        paddingVertical: 4,
        borderRadius: 10,
    },
    discountBadgeText: {
        color: '#FFF',
        fontSize: 13,
        fontWeight: '700',
    },
    eventCardCountdown: {
        position: 'absolute',
        top: 12,
        right: 12,
    },
    eventCardContent: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        padding: 16,
    },
    eventCardBrand: {
        fontSize: 11,
        fontWeight: '600',
        color: 'rgba(255,255,255,0.7)',
        letterSpacing: 1,
        textTransform: 'uppercase',
        marginBottom: 4,
    },
    eventCardTitle: {
        fontSize: 16,
        fontWeight: '700',
        color: '#FFF',
        marginBottom: 6,
    },
    eventCardItems: {
        fontSize: 12,
        color: 'rgba(255,255,255,0.6)',
    },
    notifyButton: {
        position: 'absolute',
        bottom: 16,
        right: 16,
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: 'rgba(255,255,255,0.2)',
        justifyContent: 'center',
        alignItems: 'center',
    },
    notifyButtonActive: {
        backgroundColor: 'rgba(255,215,0,0.3)',
    },

    // Info Section
    infoSection: {
        paddingHorizontal: 20,
        marginTop: 8,
    },
    infoCard: {
        padding: 24,
        borderRadius: 20,
        alignItems: 'center',
    },
    infoIcon: {
        width: 56,
        height: 56,
        borderRadius: 28,
        backgroundColor: 'rgba(0,122,255,0.1)',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 16,
    },
    infoTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: AppColors.text,
        marginBottom: 8,
    },
    infoText: {
        fontSize: 14,
        color: AppColors.textSecondary,
        textAlign: 'center',
        lineHeight: 21,
    },
});

export default FlashSalesScreen;
