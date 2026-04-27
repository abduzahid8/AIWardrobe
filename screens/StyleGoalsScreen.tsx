import React, { useState, useEffect, useCallback } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    Dimensions,
    ScrollView,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation, useFocusEffect } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInUp,
    FadeInDown,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withTiming,
} from 'react-native-reanimated';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { TahoeIconButton } from '../components/TahoeButton';
import AppColors from '../constants/AppColors';
import { useTranslation } from 'react-i18next';

const { width } = Dimensions.get('window');

// Use unified AppColors
const COLORS = {
    background: AppColors.background,
    surface: AppColors.surface,
    surfaceLight: AppColors.surfaceSecondary,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    textMuted: AppColors.textMuted,
    border: AppColors.border,
    success: '#34C759',
    warning: '#FF9500',
    error: '#FF3B30',
};

interface StyleGoal {
    id: string;
    title: string;
    icon: string;
    target: number;
    unit: string;
    description: string;
    color: string;
}

interface UserGoal {
    goalId: string;
    progress: number;
    startedAt: string;
    completedAt?: string;
}

interface ChallengeProgress {
    challengeId: string;
    daysCompleted: number;
    startedAt: string;
    status: 'active' | 'completed' | 'failed';
}

interface Challenge {
    id: string;
    title: string;
    description: string;
    days: number;
    reward: string;
    difficulty: 'easy' | 'medium' | 'hard';
}

interface GoalCardProps {
    goal: any;
    userGoal?: UserGoal;
    isActive: boolean;
    isCompleted: boolean;
    progress: number;
    onPress: () => void;
    onUpdateProgress: (amount: number) => void;
    t: (key: string) => string;
}

// Progress Ring Component
const ProgressRing = ({
    progress,
    size = 60,
    strokeWidth = 6,
    color = COLORS.primary
}: {
    progress: number;
    size?: number;
    strokeWidth?: number;
    color?: string;
}) => {
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const progressValue = Math.min(progress, 100);
    const offset = circumference - (progressValue / 100) * circumference;

    return (
        <View style={{ width: size, height: size }}>
            <View style={[styles.ringContainer, { width: size, height: size }]}>
                {/* Background circle */}
                <View
                    style={[
                        styles.ringBg,
                        {
                            width: size,
                            height: size,
                            borderRadius: size / 2,
                            borderWidth: strokeWidth,
                            borderColor: COLORS.border,
                        }
                    ]}
                />
                {/* Progress indicator (simplified visual) */}
                <View
                    style={[
                        styles.progressIndicator,
                        {
                            width: size - strokeWidth * 2,
                            height: size - strokeWidth * 2,
                            borderRadius: (size - strokeWidth * 2) / 2,
                            backgroundColor: color + '20',
                        }
                    ]}
                >
                    <Text style={[styles.progressText, { color }]}>
                        {Math.round(progressValue)}%
                    </Text>
                </View>
            </View>
        </View>
    );
};

// Goal Card Component
const GoalCard = ({ goal, userGoal, isActive, isCompleted, progress, onPress, onUpdateProgress, t }: GoalCardProps) => {
    return (
        <TouchableOpacity
            style={[
                styles.goalCard,
                isActive && styles.goalCardActive,
                isCompleted && styles.goalCardCompleted
            ]}
            onPress={onPress}
            activeOpacity={0.8}
        >
            <View style={styles.goalHeader}>
                <View style={[styles.goalIconBg, { backgroundColor: goal.color + '20' }]}>
                    <Ionicons name={goal.icon as any} size={24} color={goal.color} />
                </View>
                <View style={styles.goalInfo}>
                    <Text style={styles.goalTitle}>{goal.title}</Text>
                    <Text style={styles.goalDescription} numberOfLines={1}>
                        {goal.description}
                    </Text>
                </View>
                {isActive && (
                    <ProgressRing progress={progress} size={50} color={goal.color} />
                )}
            </View>

            {isActive && (
                <View style={styles.goalProgress}>
                    <View style={styles.progressBarBg}>
                        <View
                            style={[
                                styles.progressBarFill,
                                { width: `${Math.min(progress, 100)}%`, backgroundColor: goal.color }
                            ]}
                        />
                    </View>
                    <Text style={styles.progressLabel}>
                        {userGoal?.progress ?? 0} / {goal.target} {goal.unit}
                    </Text>

                    {/* Quick increment buttons */}
                    <View style={styles.incrementButtons}>
                        <TouchableOpacity
                            style={styles.incrementBtn}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                onUpdateProgress(1);
                            }}
                        >
                            <Ionicons name="add" size={20} color={COLORS.primary} />
                            <Text style={styles.incrementBtnText}>+1</Text>
                        </TouchableOpacity>
                    </View>
                </View>
            )}

            {isCompleted && (
                <View style={styles.completedBadge}>
                    <Ionicons name="checkmark-circle" size={18} color={COLORS.success} />
                    <Text style={styles.completedText}>{t('styleGoals.completedText')}</Text>
                </View>
            )}

            {!isActive && !isCompleted && (
                <TouchableOpacity
                    style={[styles.startGoalBtn, { backgroundColor: goal.color }]}
                    onPress={onPress}
                >
                    <Text style={styles.startGoalBtnText}>{t('styleGoals.startGoal')}</Text>
                    <Ionicons name="arrow-forward" size={16} color="#fff" />
                </TouchableOpacity>
            )}
        </TouchableOpacity>
    );
};

// Challenge Card Component
const ChallengeCard = ({
    challenge,
    progress,
    onAccept,
    onLogDay,
    t,
}: {
    challenge: Challenge;
    progress?: ChallengeProgress;
    onAccept: () => void;
    onLogDay: () => void;
    t: (key: string) => string;
}) => {
    const isActive = progress?.status === 'active';
    const isCompleted = progress?.status === 'completed';
    const dayProgress = progress ? (progress.daysCompleted / challenge.days) * 100 : 0;

    return (
        <View style={[
            styles.challengeCard,
            isActive && styles.challengeCardActive,
            isCompleted && styles.challengeCardCompleted,
        ]}>
            <View style={styles.challengeHeader}>
                <Text style={styles.challengeReward}>{challenge.reward}</Text>
                <View style={styles.challengeInfo}>
                    <Text style={styles.challengeTitle}>{challenge.title}</Text>
                    <Text style={styles.challengeDescription}>{challenge.description}</Text>
                </View>
            </View>

            {isActive && (
                <View style={styles.challengeProgress}>
                    <View style={styles.daysRow}>
                        {Array.from({ length: challenge.days }).map((_, idx) => (
                            <View
                                key={idx}
                                style={[
                                    styles.dayDot,
                                    idx < (progress?.daysCompleted || 0) && styles.dayDotCompleted
                                ]}
                            />
                        ))}
                    </View>
                    <Text style={styles.daysText}>
                        {progress?.daysCompleted || 0} / {challenge.days} days
                    </Text>
                    <TouchableOpacity
                        style={styles.logDayBtn}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            onLogDay();
                        }}
                    >
                        <Ionicons name="checkmark" size={18} color="#fff" />
                        <Text style={styles.logDayBtnText}>{t('styleGoals.logToday')}</Text>
                    </TouchableOpacity>
                </View>
            )}

            {isCompleted && (
                <View style={styles.completedBadge}>
                    <Ionicons name="trophy" size={18} color={COLORS.warning} />
                    <Text style={styles.completedText}>{t('styleGoals.challengeCompleteText')}</Text>
                </View>
            )}

            {!isActive && !isCompleted && (
                <TouchableOpacity
                    style={styles.acceptChallengeBtn}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                        onAccept();
                    }}
                >
                    <Text style={styles.acceptChallengeBtnText}>{t('styleGoals.acceptChallenge')}</Text>
                </TouchableOpacity>
            )}
        </View>
    );
};

const StyleGoalsScreen = () => {
    const navigation = useNavigation();
    const { t } = useTranslation();
    const [userGoals, setUserGoals] = useState<UserGoal[]>([]);
    const [challenges, setChallenges] = useState<ChallengeProgress[]>([]);
    const [activeTab, setActiveTab] = useState<'goals' | 'challenges'>('goals');

    const AVAILABLE_GOALS: StyleGoal[] = [
        {
            id: 'capsule',
            title: t('styleGoals.available.0.title'),
            icon: 'grid-outline',
            target: 30,
            unit: t('styleGoals.available.0.unit'),
            description: t('styleGoals.available.0.description'),
            color: '#6B7280'
        },
        {
            id: 'sustainable',
            title: t('styleGoals.available.1.title'),
            icon: 'leaf-outline',
            target: 10,
            unit: t('styleGoals.available.1.unit'),
            description: t('styleGoals.available.1.description'),
            color: '#22C55E'
        },
        {
            id: 'colorful',
            title: t('styleGoals.available.2.title'),
            icon: 'color-palette-outline',
            target: 5,
            unit: t('styleGoals.available.2.unit'),
            description: t('styleGoals.available.2.description'),
            color: '#F59E0B'
        },
        {
            id: 'minimalist',
            title: t('styleGoals.available.3.title'),
            icon: 'remove-circle-outline',
            target: 20,
            unit: t('styleGoals.available.3.unit'),
            description: t('styleGoals.available.3.description'),
            color: '#3B82F6'
        },
        {
            id: 'professional',
            title: t('styleGoals.available.4.title'),
            icon: 'briefcase-outline',
            target: 7,
            unit: t('styleGoals.available.4.unit'),
            description: t('styleGoals.available.4.description'),
            color: '#8B5CF6'
        },
        {
            id: 'complete_outfits',
            title: t('styleGoals.available.5.title'),
            icon: 'layers-outline',
            target: 14,
            unit: t('styleGoals.available.5.unit'),
            description: t('styleGoals.available.5.description'),
            color: '#EC4899'
        },
    ];

    const WEEKLY_CHALLENGES = [
        {
            id: 'no_repeat',
            title: t('styleGoals.weeklyChallengesList.0.title'),
            description: t('styleGoals.weeklyChallengesList.0.description'),
            days: 7,
            reward: '🏆',
            difficulty: 'medium' as const,
        },
        {
            id: 'monochrome',
            title: t('styleGoals.weeklyChallengesList.1.title'),
            description: t('styleGoals.weeklyChallengesList.1.description'),
            days: 1,
            reward: '⭐',
            difficulty: 'easy' as const,
        },
        {
            id: 'accessorize',
            title: t('styleGoals.weeklyChallengesList.2.title'),
            description: t('styleGoals.weeklyChallengesList.2.description'),
            days: 5,
            reward: '💎',
            difficulty: 'easy' as const,
        },
        {
            id: 'rediscover',
            title: t('styleGoals.weeklyChallengesList.3.title'),
            description: t('styleGoals.weeklyChallengesList.3.description'),
            days: 7,
            reward: '🌟',
            difficulty: 'medium' as const,
        },
    ];

    // Load saved data
    const loadData = useCallback(async () => {
        try {
            const savedGoals = await AsyncStorage.getItem('styleGoals');
            const savedChallenges = await AsyncStorage.getItem('styleChallenges');

            if (savedGoals) {
                setUserGoals(JSON.parse(savedGoals));
            }
            if (savedChallenges) {
                setChallenges(JSON.parse(savedChallenges));
            }
        } catch (error) {
            console.error('Failed to load style goals:', error);
        }
    }, []);

    useFocusEffect(
        useCallback(() => {
            loadData();
        }, [loadData])
    );

    // Save data helper
    const saveGoals = async (goals: UserGoal[]) => {
        try {
            await AsyncStorage.setItem('styleGoals', JSON.stringify(goals));
            setUserGoals(goals);
        } catch (error) {
            console.error('Failed to save goals:', error);
        }
    };

    const saveChallenges = async (challengesList: ChallengeProgress[]) => {
        try {
            await AsyncStorage.setItem('styleChallenges', JSON.stringify(challengesList));
            setChallenges(challengesList);
        } catch (error) {
            console.error('Failed to save challenges:', error);
        }
    };

    // Start a goal
    const startGoal = (goalId: string) => {
        const existingGoal = userGoals.find(g => g.goalId === goalId);
        if (existingGoal && !existingGoal.completedAt) {
            Alert.alert(t('styleGoals.alreadyActive'), t('styleGoals.goalAlreadyProgress'));
            return;
        }

        const newGoal: UserGoal = {
            goalId,
            progress: 0,
            startedAt: new Date().toISOString(),
        };

        saveGoals([...userGoals.filter(g => g.goalId !== goalId), newGoal]);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    };

    // Update goal progress
    const updateGoalProgress = (goalId: string, increment: number) => {
        const goal = AVAILABLE_GOALS.find(g => g.id === goalId);
        const updatedGoals = userGoals.map(g => {
            if (g.goalId === goalId) {
                const newProgress = Math.min(g.progress + increment, goal?.target || 100);
                const isComplete = newProgress >= (goal?.target || 100);
                return {
                    ...g,
                    progress: newProgress,
                    completedAt: isComplete ? new Date().toISOString() : undefined,
                };
            }
            return g;
        });

        saveGoals(updatedGoals);

        // Check if completed
        const updatedGoal = updatedGoals.find(g => g.goalId === goalId);
        if (updatedGoal?.completedAt) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            Alert.alert(t('styleGoals.goalAchieved'), `${t('styleGoals.congratulationsCompleted')} "${goal?.title}"`);
        }
    };

    // Accept a challenge
    const acceptChallenge = (challengeId: string) => {
        const newChallenge: ChallengeProgress = {
            challengeId,
            daysCompleted: 0,
            startedAt: new Date().toISOString(),
            status: 'active',
        };

        saveChallenges([...challenges.filter(c => c.challengeId !== challengeId), newChallenge]);
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    };

    // Log a day for challenge
    const logChallengeDay = (challengeId: string) => {
        const challenge = WEEKLY_CHALLENGES.find(c => c.id === challengeId);
        const updatedChallenges = challenges.map(c => {
            if (c.challengeId === challengeId && c.status === 'active') {
                const newDays = c.daysCompleted + 1;
                const isComplete = newDays >= (challenge?.days || 7);
                return {
                    ...c,
                    daysCompleted: newDays,
                    status: isComplete ? 'completed' : 'active',
                } as ChallengeProgress;
            }
            return c;
        });

        saveChallenges(updatedChallenges);

        // Check if completed
        const updated = updatedChallenges.find(c => c.challengeId === challengeId);
        if (updated?.status === 'completed') {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
            Alert.alert(t('styleGoals.challengeComplete'), `${t('styleGoals.amazingCompleted')} "${challenge?.title}"`);
        }
    };

    // Calculate overall stats
    const activeGoalsCount = userGoals.filter(g => !g.completedAt).length;
    const completedGoalsCount = userGoals.filter(g => g.completedAt).length;
    const activeChallengesCount = challenges.filter(c => c.status === 'active').length;

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <Animated.View
                    entering={FadeInDown.delay(50).springify()}
                    style={styles.header}
                >
                    <TahoeIconButton
                        icon="arrow-back"
                        onPress={() => navigation.goBack()}
                        color={COLORS.text}
                    />

                    <View style={styles.headerCenter}>
                        <Text style={styles.headerTitle}>{t('styleGoals.title')}</Text>
                        <Text style={styles.headerSubtitle}>{t('styleGoals.subtitle')}</Text>
                    </View>

                    <View style={{ width: 40 }} />
                </Animated.View>

                {/* Stats Summary */}
                <Animated.View
                    entering={FadeInUp.delay(100).springify()}
                    style={styles.statsSection}
                >
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{activeGoalsCount}</Text>
                        <Text style={styles.statLabel}>{t('styleGoals.activeGoals')}</Text>
                    </View>
                    <View style={styles.statDivider} />
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{completedGoalsCount}</Text>
                        <Text style={styles.statLabel}>{t('styleGoals.completed')}</Text>
                    </View>
                    <View style={styles.statDivider} />
                    <View style={styles.statItem}>
                        <Text style={styles.statNumber}>{activeChallengesCount}</Text>
                        <Text style={styles.statLabel}>{t('styleGoals.challenges')}</Text>
                    </View>
                </Animated.View>

                {/* Tab Switcher */}
                <View style={styles.tabContainer}>
                    <TouchableOpacity
                        style={[styles.tab, activeTab === 'goals' && styles.tabActive]}
                        onPress={() => setActiveTab('goals')}
                    >
                        <Text style={[styles.tabText, activeTab === 'goals' && styles.tabTextActive]}>
                            Goals
                        </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={[styles.tab, activeTab === 'challenges' && styles.tabActive]}
                        onPress={() => setActiveTab('challenges')}
                    >
                        <Text style={[styles.tabText, activeTab === 'challenges' && styles.tabTextActive]}>
                            Challenges
                        </Text>
                    </TouchableOpacity>
                </View>

                <ScrollView
                    contentContainerStyle={styles.scrollContent}
                    showsVerticalScrollIndicator={false}
                >
                    {activeTab === 'goals' ? (
                        <Animated.View entering={FadeIn.delay(150)}>
                            <Text style={styles.sectionTitle}>{t('styleGoals.yourStyleGoals')}</Text>
                            <Text style={styles.sectionSubtitle}>
                                Set goals to level up your wardrobe
                            </Text>

                            {AVAILABLE_GOALS.map((goal) => {
                                const userGoal = userGoals.find(g => g.goalId === goal.id);
                                const isActive = !!userGoal && !userGoal.completedAt;
                                const isCompleted = !!userGoal?.completedAt;
                                const progress = userGoal?.progress || 0;
                                return (
                                    <GoalCard
                                        key={goal.id}
                                        goal={goal}
                                        userGoal={userGoal}
                                        isActive={isActive}
                                        isCompleted={isCompleted}
                                        progress={progress}
                                        onPress={() => startGoal(goal.id)}
                                        onUpdateProgress={(inc) => updateGoalProgress(goal.id, inc)}
                                        t={t}
                                    />
                                );
                            })}
                        </Animated.View>
                    ) : (
                        <Animated.View entering={FadeIn.delay(150)}>
                            <Text style={styles.sectionTitle}>{t('styleGoals.weeklyChallenges')}</Text>
                            <Text style={styles.sectionSubtitle}>
                                Push your style boundaries with fun challenges
                            </Text>

                            {WEEKLY_CHALLENGES.map((challenge) => {
                                const progress = challenges.find(c => c.challengeId === challenge.id);
                                return (
                                    <ChallengeCard
                                        key={challenge.id}
                                        challenge={challenge}
                                        progress={progress}
                                        onAccept={() => acceptChallenge(challenge.id)}
                                        onLogDay={() => logChallengeDay(challenge.id)}
                                        t={t}
                                    />
                                );
                            })}
                        </Animated.View>
                    )}

                    <View style={{ height: 100 }} />
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    safeArea: {
        flex: 1,
    },

    // Header
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderBottomWidth: 1,
        borderBottomColor: COLORS.border,
    },
    headerCenter: {
        alignItems: 'center',
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: COLORS.text,
    },
    headerSubtitle: {
        fontSize: 12,
        color: COLORS.textSecondary,
        marginTop: 2,
    },

    // Stats
    statsSection: {
        flexDirection: 'row',
        paddingVertical: 20,
        paddingHorizontal: 24,
        justifyContent: 'space-around',
        backgroundColor: COLORS.surfaceLight,
        marginHorizontal: 16,
        marginTop: 16,
        borderRadius: 16,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    statItem: {
        alignItems: 'center',
    },
    statNumber: {
        fontSize: 28,
        fontWeight: '700',
        color: COLORS.primary,
    },
    statLabel: {
        fontSize: 12,
        color: COLORS.textSecondary,
        marginTop: 4,
    },
    statDivider: {
        width: 1,
        height: 40,
        backgroundColor: COLORS.border,
    },

    // Tabs
    tabContainer: {
        flexDirection: 'row',
        marginHorizontal: 16,
        marginTop: 20,
        borderRadius: 12,
        backgroundColor: COLORS.surfaceLight,
        padding: 4,
    },
    tab: {
        flex: 1,
        paddingVertical: 12,
        alignItems: 'center',
        borderRadius: 8,
    },
    tabActive: {
        backgroundColor: COLORS.primary,
    },
    tabText: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.textSecondary,
    },
    tabTextActive: {
        color: COLORS.background,
    },

    // Scroll content
    scrollContent: {
        paddingHorizontal: 16,
        paddingTop: 24,
    },
    sectionTitle: {
        fontSize: 22,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 4,
    },
    sectionSubtitle: {
        fontSize: 14,
        color: COLORS.textSecondary,
        marginBottom: 20,
    },

    // Goal Card
    goalCard: {
        backgroundColor: COLORS.surfaceLight,
        borderRadius: 20,
        padding: 20,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    goalCardActive: {
        borderColor: COLORS.primary + '40',
    },
    goalCardCompleted: {
        opacity: 0.8,
    },
    goalHeader: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    goalIconBg: {
        width: 50,
        height: 50,
        borderRadius: 14,
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 14,
    },
    goalInfo: {
        flex: 1,
    },
    goalTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: COLORS.text,
    },
    goalDescription: {
        fontSize: 13,
        color: COLORS.textSecondary,
        marginTop: 2,
    },
    goalProgress: {
        marginTop: 16,
    },
    progressBarBg: {
        height: 8,
        backgroundColor: COLORS.border,
        borderRadius: 4,
        overflow: 'hidden',
    },
    progressBarFill: {
        height: '100%',
        borderRadius: 4,
    },
    progressLabel: {
        fontSize: 12,
        color: COLORS.textSecondary,
        marginTop: 8,
    },
    incrementButtons: {
        flexDirection: 'row',
        justifyContent: 'flex-end',
        marginTop: 12,
    },
    incrementBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.primary + '15',
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
        gap: 4,
    },
    incrementBtnText: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.primary,
    },
    startGoalBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 12,
        borderRadius: 12,
        marginTop: 16,
        gap: 8,
    },
    startGoalBtnText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
    },
    completedBadge: {
        flexDirection: 'row',
        alignItems: 'center',
        marginTop: 12,
        gap: 6,
    },
    completedText: {
        fontSize: 14,
        fontWeight: '600',
        color: COLORS.success,
    },

    // Challenge Card
    challengeCard: {
        backgroundColor: COLORS.surfaceLight,
        borderRadius: 20,
        padding: 20,
        marginBottom: 16,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    challengeCardActive: {
        borderColor: COLORS.warning + '60',
        backgroundColor: COLORS.warning + '08',
    },
    challengeCardCompleted: {
        opacity: 0.8,
    },
    challengeHeader: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    challengeReward: {
        fontSize: 32,
        marginRight: 14,
    },
    challengeInfo: {
        flex: 1,
    },
    challengeTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: COLORS.text,
    },
    challengeDescription: {
        fontSize: 13,
        color: COLORS.textSecondary,
        marginTop: 2,
    },
    challengeProgress: {
        marginTop: 16,
    },
    daysRow: {
        flexDirection: 'row',
        gap: 8,
    },
    dayDot: {
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: COLORS.border,
    },
    dayDotCompleted: {
        backgroundColor: COLORS.success,
    },
    daysText: {
        fontSize: 12,
        color: COLORS.textSecondary,
        marginTop: 10,
    },
    logDayBtn: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: COLORS.success,
        paddingVertical: 10,
        borderRadius: 12,
        marginTop: 14,
        gap: 6,
    },
    logDayBtnText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
    },
    acceptChallengeBtn: {
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: COLORS.warning,
        paddingVertical: 12,
        borderRadius: 12,
        marginTop: 16,
    },
    acceptChallengeBtnText: {
        color: '#fff',
        fontSize: 14,
        fontWeight: '600',
    },

    // Progress Ring
    ringContainer: {
        position: 'relative',
        alignItems: 'center',
        justifyContent: 'center',
    },
    ringBg: {
        position: 'absolute',
    },
    progressIndicator: {
        alignItems: 'center',
        justifyContent: 'center',
    },
    progressText: {
        fontSize: 12,
        fontWeight: '700',
    },
});

export default StyleGoalsScreen;
