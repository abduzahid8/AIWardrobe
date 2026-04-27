import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TextInput,
    TouchableOpacity,
    ScrollView,
    ActivityIndicator,
    Alert,
    StyleSheet,
    Dimensions,
    Platform,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import DateTimePicker from '@react-native-community/datetimepicker';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withRepeat,
    withSequence,
    withTiming,
    Easing,
} from 'react-native-reanimated';
import AppColors from '../constants/AppColors';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import { useTranslation } from 'react-i18next';

const { width } = Dimensions.get('window');

interface WeatherDay {
    date: string;
    tempHigh: number;
    tempLow: number;
    condition: string;
    description: string;
    icon: string;
}

interface PackingItem {
    _id: string;
    itemType: string;
    color: string;
    imageUrl?: string;
    uses: number;
}

// Outfit by day type
interface OutfitByDay {
    date: string;
    items: PackingItem[];
}

interface TripPlan {
    destination: string;
    weather: WeatherDay[];
    packingList: PackingItem[];
    outfitsByDay: OutfitByDay[];
    stats: {
        totalItems: number;
        totalOutfits: number;
        daysPlanned: number;
    };
}

// Pulsing dot for loading
const PulsingDot = ({ delay = 0 }: { delay?: number }) => {
    const opacity = useSharedValue(0.3);

    useEffect(() => {
        const timer = setTimeout(() => {
            opacity.value = withRepeat(
                withSequence(
                    withTiming(1, { duration: 500 }),
                    withTiming(0.3, { duration: 500 })
                ),
                -1,
                false
            );
        }, delay);
        return () => clearTimeout(timer);
    }, []);

    const animatedStyle = useAnimatedStyle(() => ({
        opacity: opacity.value,
    }));

    return (
        <Animated.View
            style={[styles.pulsingDot, animatedStyle]}
        />
    );
};

// Floating icon animation
const FloatingIcon = () => {
    const translateY = useSharedValue(0);

    useEffect(() => {
        translateY.value = withRepeat(
            withSequence(
                withTiming(-8, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) }),
                withTiming(0, { duration: 2000, easing: Easing.bezier(0.4, 0, 0.2, 1) })
            ),
            -1,
            true
        );
    }, []);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ translateY: translateY.value }],
    }));

    return (
        <Animated.View style={[styles.floatingIconContainer, animatedStyle]}>
            <View style={styles.floatingIcon}>
                <Ionicons name="airplane" size={36} color={AppColors.primary} />
            </View>
        </Animated.View>
    );
};

// Occasion chip component
const OccasionChip = ({
    option,
    selected,
    onPress
}: {
    option: { id: string; emoji: string; label: string };
    selected: boolean;
    onPress: () => void;
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    const handlePressIn = () => {
        scale.value = withSpring(0.95, { damping: 15 });
    };

    const handlePressOut = () => {
        scale.value = withSpring(1, { damping: 15 });
    };

    return (
        <TouchableOpacity
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={() => {
                Haptics.selectionAsync();
                onPress();
            }}
            activeOpacity={1}
        >
            <Animated.View style={[
                styles.occasionChip,
                selected && styles.occasionChipSelected,
                animatedStyle,
            ]}>
                <Text style={styles.occasionEmoji}>{option.emoji}</Text>
                <Text style={[
                    styles.occasionLabel,
                    selected && styles.occasionLabelSelected,
                ]}>
                    {option.label}
                </Text>
            </Animated.View>
        </TouchableOpacity>
    );
};

const TripPlannerScreen = () => {
    const navigation = useNavigation();
    const { user } = useAuthStore();
    const { t } = useTranslation();
    const [step, setStep] = useState<'input' | 'loading' | 'result'>('input');

    // Form inputs
    const [destination, setDestination] = useState('');
    const [startDate, setStartDate] = useState(new Date());
    const [endDate, setEndDate] = useState(new Date(Date.now() + 7 * 24 * 60 * 60 * 1000));
    const [occasions, setOccasions] = useState<string[]>(['casual']);
    const [showStartPicker, setShowStartPicker] = useState(false);
    const [showEndPicker, setShowEndPicker] = useState(false);

    // Results
    const [tripPlan, setTripPlan] = useState<TripPlan | null>(null);

    const buttonScale = useSharedValue(1);
    const buttonAnimStyle = useAnimatedStyle(() => ({
        transform: [{ scale: buttonScale.value }],
    }));

    const occasionOptions = [
        { id: 'casual', emoji: '👕', label: 'Casual' },
        { id: 'business', emoji: '💼', label: 'Business' },
        { id: 'formal', emoji: '👔', label: 'Formal' },
        { id: 'beach', emoji: '🏖️', label: 'Beach' },
        { id: 'sport', emoji: '⚽', label: 'Sport' },
        { id: 'party', emoji: '🎉', label: 'Party' }
    ];

    const toggleOccasion = (occasionId: string) => {
        setOccasions(prev =>
            prev.includes(occasionId)
                ? prev.filter(o => o !== occasionId)
                : [...prev, occasionId]
        );
    };

    const handlePressIn = () => {
        buttonScale.value = withSpring(0.96, { damping: 15 });
    };

    const handlePressOut = () => {
        buttonScale.value = withSpring(1, { damping: 15 });
    };

    const handleCreatePlan = async () => {
        if (!destination.trim()) {
            Alert.alert(t('common.error'), t('tripPlanner.enterDestination'));
            return;
        }

        if (!user) {
            Alert.alert(t('common.error'), t('tripPlanner.mustBeLoggedIn'));
            return;
        }

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setStep('loading');

        try {
            const { data, error } = await supabase.functions.invoke('create-trip-plan', {
                body: {
                    destination: destination.trim(),
                    startDate: startDate.toISOString().split('T')[0],
                    endDate: endDate.toISOString().split('T')[0],
                    occasions
                }
            });

            if (error) {
                console.error("Supabase function error:", error);
                // Fallback demo data (simplified for resilience)
                throw new Error(error.message || t('common.functionError'));
            }

            setTripPlan(data);
            setStep('result');
        } catch (error: any) {
            console.error('Trip creation error:', error);
            // Fallback demo data if function fails or network error
            setTripPlan({
                destination: destination.trim(),
                weather: [],
                packingList: [
                    { _id: '1', itemType: 'T-Shirt', color: 'White', uses: 3 },
                    { _id: '2', itemType: 'Jeans', color: 'Blue', uses: 2 },
                    { _id: '3', itemType: 'Sneakers', color: 'White', uses: 5 },
                    { _id: '4', itemType: 'Jacket', color: 'Navy', uses: 3 },
                ],
                outfitsByDay: [],
                stats: {
                    totalItems: 4,
                    totalOutfits: 5,
                    daysPlanned: Math.ceil((endDate.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24))
                }
            });
            // Show alert but still show demo result
            Alert.alert(t('tripPlanner.note'), t('tripPlanner.couldNotConnectPlanner'));
            setStep('result');
        }
    };

    const handleStartOver = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setStep('input');
        setDestination('');
        setTripPlan(null);
    };

    const formatDate = (date: Date) => {
        return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    };

    // Input Step
    const renderInputStep = () => (
        <ScrollView
            style={{ flex: 1 }}
            contentContainerStyle={{ paddingHorizontal: 24, paddingBottom: 40 }}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
        >
            {/* Hero */}
            <Animated.View entering={FadeIn.duration(600)} style={styles.heroSection}>
                <FloatingIcon />
                <Text style={styles.heroTitle}>{t('tripPlanner.planYourTrip')}</Text>
                <Text style={styles.heroSubtitle}>
                    {t('tripPlanner.personalizedPackingList')}
                </Text>
            </Animated.View>

            {/* Destination */}
            <Animated.View entering={FadeInUp.delay(100).springify()}>
                <Text style={styles.sectionLabel}>{t('tripPlanner.destination')}</Text>
                <TextInput
                    style={styles.textInput}
                    placeholder={t('tripPlanner.destinationPlaceholder')}
                    placeholderTextColor={AppColors.textMuted}
                    value={destination}
                    onChangeText={setDestination}
                    maxLength={200}
                />
            </Animated.View>

            {/* Dates */}
            <Animated.View entering={FadeInUp.delay(150).springify()} style={{ marginTop: 20 }}>
                <Text style={styles.sectionLabel}>{t('tripPlanner.travelDates')}</Text>
                <View style={styles.dateRow}>
                    <TouchableOpacity
                        onPress={() => setShowStartPicker(true)}
                        style={styles.dateButton}
                    >
                        <Text style={styles.dateLabelSmall}>{t('tripPlanner.start')}</Text>
                        <Text style={styles.dateValue}>{formatDate(startDate)}</Text>
                    </TouchableOpacity>

                    <View style={styles.dateDivider}>
                        <Ionicons name="arrow-forward" size={16} color={AppColors.textMuted} />
                    </View>

                    <TouchableOpacity
                        onPress={() => setShowEndPicker(true)}
                        style={styles.dateButton}
                    >
                        <Text style={styles.dateLabelSmall}>{t('tripPlanner.end')}</Text>
                        <Text style={styles.dateValue}>{formatDate(endDate)}</Text>
                    </TouchableOpacity>
                </View>
            </Animated.View>

            {showStartPicker && (
                <DateTimePicker
                    value={startDate}
                    mode="date"
                    display="default"
                    onChange={(event, date) => {
                        setShowStartPicker(false);
                        if (date) setStartDate(date);
                    }}
                />
            )}

            {showEndPicker && (
                <DateTimePicker
                    value={endDate}
                    mode="date"
                    display="default"
                    minimumDate={startDate}
                    onChange={(event, date) => {
                        setShowEndPicker(false);
                        if (date) setEndDate(date);
                    }}
                />
            )}

            {/* Occasions */}
            <Animated.View entering={FadeInUp.delay(200).springify()} style={{ marginTop: 20 }}>
                <Text style={styles.sectionLabel}>{t('tripPlanner.occasions')}</Text>
                <View style={styles.occasionsGrid}>
                    {occasionOptions.map(option => (
                        <OccasionChip
                            key={option.id}
                            option={option}
                            selected={occasions.includes(option.id)}
                            onPress={() => toggleOccasion(option.id)}
                        />
                    ))}
                </View>
            </Animated.View>

            {/* Create Button */}
            <Animated.View entering={FadeInUp.delay(300).springify()} style={{ marginTop: 32 }}>
                <TouchableOpacity
                    onPressIn={handlePressIn}
                    onPressOut={handlePressOut}
                    onPress={handleCreatePlan}
                    disabled={!destination.trim()}
                    activeOpacity={1}
                >
                    <Animated.View style={[
                        styles.primaryButton,
                        !destination.trim() && styles.primaryButtonDisabled,
                        buttonAnimStyle,
                    ]}>
                        <Ionicons
                            name="sparkles"
                            size={20}
                            color={AppColors.background}
                            style={{ marginRight: 10 }}
                        />
                        <Text style={styles.primaryButtonText}>{t('tripPlanner.createTripPlan')}</Text>
                    </Animated.View>
                </TouchableOpacity>
            </Animated.View>
        </ScrollView>
    );

    // Loading Step
    const renderLoadingStep = () => (
        <View style={styles.loadingContainer}>
            <Animated.View entering={FadeIn.duration(400)}>
                <View style={styles.loadingIcon}>
                    <ActivityIndicator size="large" color={AppColors.primary} />
                </View>
            </Animated.View>

            <Animated.Text entering={FadeInUp.delay(100)} style={styles.loadingTitle}>
                Planning your trip...
            </Animated.Text>

            <Animated.Text entering={FadeInUp.delay(200)} style={styles.loadingSubtitle}>
                Analyzing weather and selecting the perfect outfits for {destination}
            </Animated.Text>

            <View style={styles.dotsContainer}>
                <PulsingDot delay={0} />
                <PulsingDot delay={150} />
                <PulsingDot delay={300} />
            </View>
        </View>
    );

    // Result Step
    const renderResultStep = () => {
        if (!tripPlan) return null;

        return (
            <ScrollView
                style={{ flex: 1 }}
                contentContainerStyle={{ paddingHorizontal: 24, paddingBottom: 40 }}
                showsVerticalScrollIndicator={false}
            >
                {/* Header */}
                <Animated.View entering={FadeInDown.delay(50).springify()} style={styles.resultHeader}>
                    <Text style={styles.resultSubtitle}>{t('tripPlanner.yourTripTo')}</Text>
                    <Text style={styles.resultTitle}>{tripPlan.destination}</Text>
                    <Text style={styles.resultDates}>
                        {formatDate(startDate)} - {formatDate(endDate)}
                    </Text>
                </Animated.View>

                {/* Stats */}
                <Animated.View entering={FadeInUp.delay(100).springify()} style={styles.statsCard}>
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{tripPlan.stats.daysPlanned}</Text>
                        <Text style={styles.statLabel}>{t('tripPlanner.days')}</Text>
                    </View>
                    <View style={styles.statDivider} />
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{tripPlan.stats.totalItems}</Text>
                        <Text style={styles.statLabel}>{t('common.items')}</Text>
                    </View>
                    <View style={styles.statDivider} />
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{tripPlan.stats.totalOutfits}</Text>
                        <Text style={styles.statLabel}>{t('common.looks')}</Text>
                    </View>
                </Animated.View>

                {/* Packing List */}
                <Animated.View entering={FadeInUp.delay(200).springify()}>
                    <Text style={styles.sectionTitle}>{t('tripPlanner.packingList')}</Text>
                    <View style={styles.packingListCard}>
                        {tripPlan.packingList.map((item, index) => (
                            <View
                                key={item._id}
                                style={[
                                    styles.packingItem,
                                    index < tripPlan.packingList.length - 1 && styles.packingItemBorder
                                ]}
                            >
                                <View style={styles.packingItemIcon}>
                                    <Ionicons name="shirt-outline" size={20} color={AppColors.textSecondary} />
                                </View>
                                <View style={styles.packingItemInfo}>
                                    <Text style={styles.packingItemName}>
                                        {item.color} {item.itemType}
                                    </Text>
                                    <Text style={styles.packingItemUses}>
                                        Wear {item.uses}x during trip
                                    </Text>
                                </View>
                                <View style={styles.packingItemCheck}>
                                    <Ionicons name="checkmark-circle" size={22} color={AppColors.textMuted} />
                                </View>
                            </View>
                        ))}
                    </View>
                </Animated.View>

                {/* Action Button */}
                <Animated.View entering={FadeInUp.delay(300).springify()} style={{ marginTop: 24 }}>
                    <TouchableOpacity
                        onPress={handleStartOver}
                        style={styles.primaryButton}
                    >
                        <Ionicons
                            name="add"
                            size={22}
                            color={AppColors.background}
                            style={{ marginRight: 8 }}
                        />
                        <Text style={styles.primaryButtonText}>{t('tripPlanner.planAnotherTrip')}</Text>
                    </TouchableOpacity>
                </Animated.View>
            </ScrollView>
        );
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={{ flex: 1 }}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity
                        onPress={() => navigation.goBack()}
                        hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                    >
                        <Ionicons name="close" size={28} color={AppColors.text} />
                    </TouchableOpacity>

                    {step === 'result' && (
                        <TouchableOpacity
                            onPress={() => {/* Share */ }}
                            hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
                        >
                            <Ionicons name="share-outline" size={24} color={AppColors.text} />
                        </TouchableOpacity>
                    )}

                    {step !== 'result' && <View style={{ width: 28 }} />}
                </View>

                {/* Content */}
                {step === 'input' && renderInputStep()}
                {step === 'loading' && renderLoadingStep()}
                {step === 'result' && renderResultStep()}
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: AppColors.background,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
    },

    // Hero
    heroSection: {
        alignItems: 'center',
        paddingTop: 32,
        paddingBottom: 32,
    },
    floatingIconContainer: {
        marginBottom: 20,
    },
    floatingIcon: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: AppColors.surface,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    heroTitle: {
        fontSize: 28,
        fontWeight: '700',
        color: AppColors.text,
        textAlign: 'center',
        marginBottom: 10,
        letterSpacing: -0.5,
    },
    heroSubtitle: {
        fontSize: 16,
        color: AppColors.textSecondary,
        textAlign: 'center',
        lineHeight: 24,
        paddingHorizontal: 20,
    },

    // Section
    sectionLabel: {
        fontSize: 16,
        fontWeight: '600',
        color: AppColors.text,
        marginBottom: 12,
    },
    sectionTitle: {
        fontSize: 18,
        fontWeight: '600',
        color: AppColors.text,
        marginBottom: 12,
        marginTop: 24,
    },

    // Input
    textInput: {
        backgroundColor: AppColors.surface,
        borderRadius: 16,
        padding: 16,
        fontSize: 16,
        color: AppColors.text,
        borderWidth: 1,
        borderColor: AppColors.border,
    },

    // Date
    dateRow: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    dateButton: {
        flex: 1,
        backgroundColor: AppColors.surface,
        borderRadius: 16,
        padding: 16,
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    dateLabelSmall: {
        fontSize: 12,
        color: AppColors.textMuted,
        marginBottom: 4,
    },
    dateValue: {
        fontSize: 16,
        fontWeight: '600',
        color: AppColors.text,
    },
    dateDivider: {
        paddingHorizontal: 12,
    },

    // Occasions
    occasionsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 10,
    },
    occasionChip: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 20,
        backgroundColor: AppColors.surface,
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    occasionChipSelected: {
        backgroundColor: AppColors.primary,
        borderColor: AppColors.primary,
    },
    occasionEmoji: {
        fontSize: 16,
        marginRight: 6,
    },
    occasionLabel: {
        fontSize: 14,
        color: AppColors.text,
    },
    occasionLabelSelected: {
        color: AppColors.background,
        fontWeight: '600',
    },

    // Button
    primaryButton: {
        backgroundColor: AppColors.primary,
        paddingVertical: 18,
        borderRadius: 16,
        alignItems: 'center',
        flexDirection: 'row',
        justifyContent: 'center',
    },
    primaryButtonDisabled: {
        backgroundColor: AppColors.border,
    },
    primaryButtonText: {
        fontSize: 17,
        fontWeight: '600',
        color: AppColors.background,
    },

    // Loading
    loadingContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        paddingHorizontal: 40,
    },
    loadingIcon: {
        width: 100,
        height: 100,
        borderRadius: 50,
        backgroundColor: AppColors.surface,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 28,
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    loadingTitle: {
        fontSize: 24,
        fontWeight: '700',
        color: AppColors.text,
        textAlign: 'center',
        marginBottom: 12,
    },
    loadingSubtitle: {
        fontSize: 15,
        color: AppColors.textSecondary,
        textAlign: 'center',
        lineHeight: 22,
    },
    dotsContainer: {
        flexDirection: 'row',
        marginTop: 28,
    },
    pulsingDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: AppColors.primary,
        marginHorizontal: 3,
    },

    // Result
    resultHeader: {
        alignItems: 'center',
        paddingTop: 16,
        paddingBottom: 24,
    },
    resultSubtitle: {
        fontSize: 14,
        color: AppColors.textMuted,
        marginBottom: 4,
    },
    resultTitle: {
        fontSize: 28,
        fontWeight: '700',
        color: AppColors.text,
        letterSpacing: -0.5,
    },
    resultDates: {
        fontSize: 15,
        color: AppColors.textSecondary,
        marginTop: 8,
    },

    // Stats
    statsCard: {
        flexDirection: 'row',
        backgroundColor: AppColors.surface,
        borderRadius: 20,
        padding: 20,
        borderWidth: 1,
        borderColor: AppColors.border,
    },
    statItem: {
        flex: 1,
        alignItems: 'center',
    },
    statNumber: {
        fontSize: 28,
        fontWeight: '700',
        color: AppColors.text,
    },
    statLabel: {
        fontSize: 13,
        color: AppColors.textSecondary,
        marginTop: 4,
    },
    statDivider: {
        width: 1,
        backgroundColor: AppColors.border,
    },

    // Packing List
    packingListCard: {
        backgroundColor: AppColors.surface,
        borderRadius: 20,
        borderWidth: 1,
        borderColor: AppColors.border,
        overflow: 'hidden',
    },
    packingItem: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 16,
    },
    packingItemBorder: {
        borderBottomWidth: 1,
        borderBottomColor: AppColors.border,
    },
    packingItemIcon: {
        width: 40,
        height: 40,
        borderRadius: 12,
        backgroundColor: AppColors.background,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 14,
    },
    packingItemInfo: {
        flex: 1,
    },
    packingItemName: {
        fontSize: 15,
        fontWeight: '500',
        color: AppColors.text,
    },
    packingItemUses: {
        fontSize: 13,
        color: AppColors.textSecondary,
        marginTop: 2,
    },
    packingItemCheck: {
        marginLeft: 12,
    },
});

export default TripPlannerScreen;
