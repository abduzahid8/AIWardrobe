import React, { useState, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    TouchableOpacity,
    ScrollView,
    Image,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useNavigation } from '@react-navigation/native';
import * as Haptics from 'expo-haptics';
import Animated, {
    FadeIn,
    FadeInDown,
    FadeInUp,
    FadeOut,
    SlideInRight,
    SlideOutLeft,
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withTiming,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import AppColors from '../constants/AppColors';
import { useStylePreferenceStore } from '../store/stylePreferenceStore';

const { width, height } = Dimensions.get('window');

const COLORS = {
    background: AppColors.background,
    surface: AppColors.surface,
    primary: AppColors.primary,
    accent: AppColors.accent,
    text: AppColors.text,
    textSecondary: AppColors.textSecondary,
    border: AppColors.border,
};

// ============================================
// STEP DATA
// ============================================

const STYLE_PERSONALITIES = [
    { id: 'classic', name: 'Classic', emoji: '👔', description: 'Timeless, elegant, refined' },
    { id: 'trendy', name: 'Trendy', emoji: '✨', description: 'Fashion-forward, current' },
    { id: 'minimalist', name: 'Minimalist', emoji: '⬜', description: 'Clean lines, simple' },
    { id: 'bohemian', name: 'Bohemian', emoji: '🌸', description: 'Free-spirited, artistic' },
    { id: 'edgy', name: 'Edgy', emoji: '🖤', description: 'Bold, unconventional' },
    { id: 'romantic', name: 'Romantic', emoji: '💕', description: 'Soft, feminine, flowy' },
    { id: 'sporty', name: 'Sporty', emoji: '⚡', description: 'Active, comfortable' },
];

const COLOR_OPTIONS = [
    { id: 'black', name: 'Black', color: '#000000' },
    { id: 'white', name: 'White', color: '#FFFFFF' },
    { id: 'navy', name: 'Navy', color: '#1a237e' },
    { id: 'beige', name: 'Beige', color: '#d4c4a8' },
    { id: 'gray', name: 'Gray', color: '#757575' },
    { id: 'burgundy', name: 'Burgundy', color: '#800020' },
    { id: 'olive', name: 'Olive', color: '#556b2f' },
    { id: 'brown', name: 'Brown', color: '#8b4513' },
    { id: 'pink', name: 'Pink', color: '#e91e63' },
    { id: 'blue', name: 'Blue', color: '#2196f3' },
    { id: 'red', name: 'Red', color: '#f44336' },
    { id: 'green', name: 'Green', color: '#4caf50' },
];

const OCCASIONS = [
    { id: 'work', name: 'Work/Office', icon: 'briefcase-outline' },
    { id: 'casual', name: 'Casual/Weekend', icon: 'cafe-outline' },
    { id: 'date', name: 'Date Night', icon: 'heart-outline' },
    { id: 'fitness', name: 'Fitness/Gym', icon: 'fitness-outline' },
    { id: 'formal', name: 'Formal Events', icon: 'diamond-outline' },
    { id: 'travel', name: 'Travel', icon: 'airplane-outline' },
];

const FIT_OPTIONS = [
    { id: 'loose', name: 'Relaxed Fit', description: 'Comfortable, roomy', icon: '👕' },
    { id: 'fitted', name: 'Fitted', description: 'Tailored, structured', icon: '👔' },
    { id: 'balanced', name: 'Balanced', description: 'Mix of both', icon: '⚖️' },
];

const STYLE_GOALS = [
    { id: 'organize_closet', name: 'Organize My Closet', icon: 'grid-outline' },
    { id: 'get_styled', name: 'Get AI Outfit Ideas', icon: 'sparkles-outline' },
    { id: 'shop_smarter', name: 'Shop Smarter', icon: 'cart-outline' },
    { id: 'build_capsule', name: 'Build a Capsule Wardrobe', icon: 'cube-outline' },
    { id: 'explore_trends', name: 'Explore New Trends', icon: 'trending-up-outline' },
    { id: 'sustainability', name: 'Be More Sustainable', icon: 'leaf-outline' },
];

// ============================================
// COMPONENTS
// ============================================

const ProgressBar = ({ currentStep, totalSteps }: { currentStep: number; totalSteps: number }) => {
    const progress = (currentStep + 1) / totalSteps;

    return (
        <View style={styles.progressContainer}>
            <View style={styles.progressBar}>
                <Animated.View
                    style={[styles.progressFill, { width: `${progress * 100}%` }]}
                />
            </View>
            <Text style={styles.progressText}>
                {currentStep + 1} of {totalSteps}
            </Text>
        </View>
    );
};

const SelectableChip = ({
    item,
    selected,
    onPress,
    showColor
}: {
    item: any;
    selected: boolean;
    onPress: () => void;
    showColor?: boolean;
}) => (
    <TouchableOpacity
        style={[
            styles.chip,
            selected && styles.chipSelected,
            showColor && { borderColor: item.color }
        ]}
        onPress={onPress}
        activeOpacity={0.7}
    >
        {showColor && (
            <View style={[styles.colorDot, { backgroundColor: item.color }]} />
        )}
        {item.emoji && <Text style={styles.chipEmoji}>{item.emoji}</Text>}
        {item.icon && (
            <Ionicons
                name={item.icon}
                size={20}
                color={selected ? COLORS.primary : COLORS.textSecondary}
            />
        )}
        <Text style={[styles.chipText, selected && styles.chipTextSelected]}>
            {item.name}
        </Text>
        {selected && (
            <Ionicons name="checkmark-circle" size={18} color={COLORS.primary} />
        )}
    </TouchableOpacity>
);

// ============================================
// STEP SCREENS
// ============================================

const WelcomeStep = ({ onNext }: { onNext: () => void }) => (
    <Animated.View
        style={styles.stepContainer}
        entering={FadeIn.duration(500)}
    >
        <View style={styles.welcomeContent}>
            <Text style={styles.welcomeEmoji}>👋</Text>
            <Text style={styles.welcomeTitle}>Welcome to AIWardrobe</Text>
            <Text style={styles.welcomeSubtitle}>
                Let's personalize your experience in just 5 quick steps
            </Text>

            <View style={styles.benefitsList}>
                <View style={styles.benefitItem}>
                    <Ionicons name="sparkles" size={24} color={COLORS.primary} />
                    <Text style={styles.benefitText}>AI learns your unique style</Text>
                </View>
                <View style={styles.benefitItem}>
                    <Ionicons name="thumbs-up" size={24} color={COLORS.primary} />
                    <Text style={styles.benefitText}>Better outfit recommendations</Text>
                </View>
                <View style={styles.benefitItem}>
                    <Ionicons name="time" size={24} color={COLORS.primary} />
                    <Text style={styles.benefitText}>Takes less than 2 minutes</Text>
                </View>
            </View>
        </View>

        <TouchableOpacity style={styles.primaryButton} onPress={onNext}>
            <LinearGradient
                colors={[COLORS.primary, COLORS.accent]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.gradientButton}
            >
                <Text style={styles.primaryButtonText}>Let's Get Started</Text>
                <Ionicons name="arrow-forward" size={20} color="#FFF" />
            </LinearGradient>
        </TouchableOpacity>
    </Animated.View>
);

const StylePersonalityStep = ({
    selected,
    onSelect,
    onNext,
    onBack
}: {
    selected: string | undefined;
    onSelect: (id: string) => void;
    onNext: () => void;
    onBack: () => void;
}) => (
    <Animated.View
        style={styles.stepContainer}
        entering={SlideInRight.duration(300)}
        exiting={SlideOutLeft.duration(300)}
    >
        <Text style={styles.stepTitle}>What's your style personality?</Text>
        <Text style={styles.stepSubtitle}>Select the one that resonates most with you</Text>

        <ScrollView
            style={styles.optionsScroll}
            showsVerticalScrollIndicator={false}
        >
            <View style={styles.optionsGrid}>
                {STYLE_PERSONALITIES.map((item) => (
                    <TouchableOpacity
                        key={item.id}
                        style={[
                            styles.personalityCard,
                            selected === item.id && styles.personalityCardSelected
                        ]}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            onSelect(item.id);
                        }}
                    >
                        <Text style={styles.personalityEmoji}>{item.emoji}</Text>
                        <Text style={styles.personalityName}>{item.name}</Text>
                        <Text style={styles.personalityDesc}>{item.description}</Text>
                        {selected === item.id && (
                            <View style={styles.selectedBadge}>
                                <Ionicons name="checkmark" size={16} color="#FFF" />
                            </View>
                        )}
                    </TouchableOpacity>
                ))}
            </View>
        </ScrollView>

        <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.backButton} onPress={onBack}>
                <Ionicons name="arrow-back" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
            <TouchableOpacity
                style={[styles.nextButton, !selected && styles.buttonDisabled]}
                onPress={onNext}
                disabled={!selected}
            >
                <Text style={styles.nextButtonText}>Continue</Text>
                <Ionicons name="arrow-forward" size={20} color="#FFF" />
            </TouchableOpacity>
        </View>
    </Animated.View>
);

const ColorPreferencesStep = ({
    favoriteColors,
    onToggleColor,
    onNext,
    onBack
}: {
    favoriteColors: string[];
    onToggleColor: (id: string) => void;
    onNext: () => void;
    onBack: () => void;
}) => (
    <Animated.View
        style={styles.stepContainer}
        entering={SlideInRight.duration(300)}
        exiting={SlideOutLeft.duration(300)}
    >
        <Text style={styles.stepTitle}>Which colors do you love?</Text>
        <Text style={styles.stepSubtitle}>Select 3-5 colors you wear most often</Text>

        <View style={styles.colorGrid}>
            {COLOR_OPTIONS.map((color) => (
                <TouchableOpacity
                    key={color.id}
                    style={[
                        styles.colorOption,
                        { backgroundColor: color.color },
                        favoriteColors.includes(color.id) && styles.colorOptionSelected
                    ]}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        onToggleColor(color.id);
                    }}
                >
                    {favoriteColors.includes(color.id) && (
                        <Ionicons
                            name="checkmark"
                            size={24}
                            color={color.id === 'white' || color.id === 'beige' ? '#000' : '#FFF'}
                        />
                    )}
                </TouchableOpacity>
            ))}
        </View>

        <Text style={styles.helperText}>
            Selected: {favoriteColors.length} colors
        </Text>

        <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.backButton} onPress={onBack}>
                <Ionicons name="arrow-back" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
            <TouchableOpacity
                style={[styles.nextButton, favoriteColors.length < 1 && styles.buttonDisabled]}
                onPress={onNext}
                disabled={favoriteColors.length < 1}
            >
                <Text style={styles.nextButtonText}>Continue</Text>
                <Ionicons name="arrow-forward" size={20} color="#FFF" />
            </TouchableOpacity>
        </View>
    </Animated.View>
);

const OccasionsStep = ({
    selectedOccasions,
    onToggleOccasion,
    onNext,
    onBack
}: {
    selectedOccasions: string[];
    onToggleOccasion: (id: string) => void;
    onNext: () => void;
    onBack: () => void;
}) => (
    <Animated.View
        style={styles.stepContainer}
        entering={SlideInRight.duration(300)}
        exiting={SlideOutLeft.duration(300)}
    >
        <Text style={styles.stepTitle}>What do you dress for most?</Text>
        <Text style={styles.stepSubtitle}>Select all that apply</Text>

        <View style={styles.occasionGrid}>
            {OCCASIONS.map((occasion) => (
                <TouchableOpacity
                    key={occasion.id}
                    style={[
                        styles.occasionCard,
                        selectedOccasions.includes(occasion.id) && styles.occasionCardSelected
                    ]}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        onToggleOccasion(occasion.id);
                    }}
                >
                    <View style={[
                        styles.occasionIconContainer,
                        selectedOccasions.includes(occasion.id) && styles.occasionIconSelected
                    ]}>
                        <Ionicons
                            name={occasion.icon as any}
                            size={28}
                            color={selectedOccasions.includes(occasion.id) ? '#FFF' : COLORS.textSecondary}
                        />
                    </View>
                    <Text style={[
                        styles.occasionName,
                        selectedOccasions.includes(occasion.id) && styles.occasionNameSelected
                    ]}>
                        {occasion.name}
                    </Text>
                </TouchableOpacity>
            ))}
        </View>

        <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.backButton} onPress={onBack}>
                <Ionicons name="arrow-back" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
            <TouchableOpacity
                style={[styles.nextButton, selectedOccasions.length < 1 && styles.buttonDisabled]}
                onPress={onNext}
                disabled={selectedOccasions.length < 1}
            >
                <Text style={styles.nextButtonText}>Continue</Text>
                <Ionicons name="arrow-forward" size={20} color="#FFF" />
            </TouchableOpacity>
        </View>
    </Animated.View>
);

const FitPreferenceStep = ({
    selected,
    onSelect,
    onNext,
    onBack
}: {
    selected: string;
    onSelect: (id: string) => void;
    onNext: () => void;
    onBack: () => void;
}) => (
    <Animated.View
        style={styles.stepContainer}
        entering={SlideInRight.duration(300)}
        exiting={SlideOutLeft.duration(300)}
    >
        <Text style={styles.stepTitle}>How do you like your clothes to fit?</Text>
        <Text style={styles.stepSubtitle}>This helps us suggest the right styles</Text>

        <View style={styles.fitOptions}>
            {FIT_OPTIONS.map((option) => (
                <TouchableOpacity
                    key={option.id}
                    style={[
                        styles.fitCard,
                        selected === option.id && styles.fitCardSelected
                    ]}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        onSelect(option.id);
                    }}
                >
                    <Text style={styles.fitEmoji}>{option.icon}</Text>
                    <Text style={styles.fitName}>{option.name}</Text>
                    <Text style={styles.fitDesc}>{option.description}</Text>
                    {selected === option.id && (
                        <View style={styles.selectedBadge}>
                            <Ionicons name="checkmark" size={16} color="#FFF" />
                        </View>
                    )}
                </TouchableOpacity>
            ))}
        </View>

        <View style={styles.buttonRow}>
            <TouchableOpacity style={styles.backButton} onPress={onBack}>
                <Ionicons name="arrow-back" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
            <TouchableOpacity
                style={styles.nextButton}
                onPress={onNext}
            >
                <Text style={styles.nextButtonText}>Continue</Text>
                <Ionicons name="arrow-forward" size={20} color="#FFF" />
            </TouchableOpacity>
        </View>
    </Animated.View>
);

const GoalsStep = ({
    selectedGoals,
    onToggleGoal,
    onComplete
}: {
    selectedGoals: string[];
    onToggleGoal: (id: string) => void;
    onComplete: () => void;
}) => (
    <Animated.View
        style={styles.stepContainer}
        entering={SlideInRight.duration(300)}
    >
        <Text style={styles.stepTitle}>What are your style goals?</Text>
        <Text style={styles.stepSubtitle}>Select all that apply - we'll tailor your experience</Text>

        <View style={styles.goalsGrid}>
            {STYLE_GOALS.map((goal) => (
                <TouchableOpacity
                    key={goal.id}
                    style={[
                        styles.goalCard,
                        selectedGoals.includes(goal.id) && styles.goalCardSelected
                    ]}
                    onPress={() => {
                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                        onToggleGoal(goal.id);
                    }}
                >
                    <Ionicons
                        name={goal.icon as any}
                        size={24}
                        color={selectedGoals.includes(goal.id) ? COLORS.primary : COLORS.textSecondary}
                    />
                    <Text style={[
                        styles.goalText,
                        selectedGoals.includes(goal.id) && styles.goalTextSelected
                    ]}>
                        {goal.name}
                    </Text>
                </TouchableOpacity>
            ))}
        </View>

        <TouchableOpacity style={styles.completeButton} onPress={onComplete}>
            <LinearGradient
                colors={[COLORS.primary, COLORS.accent]}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.gradientButton}
            >
                <Text style={styles.primaryButtonText}>Complete Setup</Text>
                <Ionicons name="checkmark-circle" size={20} color="#FFF" />
            </LinearGradient>
        </TouchableOpacity>
    </Animated.View>
);

// ============================================
// MAIN SCREEN
// ============================================

const StyleQuizScreen = () => {
    const navigation = useNavigation();
    const {
        setPreferences,
        completeOnboarding,
        preferences
    } = useStylePreferenceStore();

    const [step, setStep] = useState(0);
    const [stylePersonality, setStylePersonality] = useState<string | undefined>();
    const [favoriteColors, setFavoriteColors] = useState<string[]>([]);
    const [occasions, setOccasions] = useState<string[]>([]);
    const [fitPreference, setFitPreference] = useState<string>('balanced');
    const [goals, setGoals] = useState<string[]>([]);

    const TOTAL_STEPS = 6;

    const handleNext = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        setStep(step + 1);
    };

    const handleBack = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setStep(Math.max(0, step - 1));
    };

    const toggleColor = (colorId: string) => {
        setFavoriteColors(prev =>
            prev.includes(colorId)
                ? prev.filter(c => c !== colorId)
                : [...prev, colorId]
        );
    };

    const toggleOccasion = (occasionId: string) => {
        setOccasions(prev =>
            prev.includes(occasionId)
                ? prev.filter(o => o !== occasionId)
                : [...prev, occasionId]
        );
    };

    const toggleGoal = (goalId: string) => {
        setGoals(prev =>
            prev.includes(goalId)
                ? prev.filter(g => g !== goalId)
                : [...prev, goalId]
        );
    };

    const handleComplete = () => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);

        // Save all preferences
        setPreferences({
            stylePersonality: stylePersonality as any,
            favoriteColors,
            primaryOccasions: occasions,
            fitPreference: fitPreference as any,
            styleGoals: goals,
            prefersSustainable: goals.includes('sustainability'),
        });

        completeOnboarding();

        // Navigate to main app
        navigation.reset({
            index: 0,
            routes: [{ name: 'Main' as never }],
        });
    };

    const renderStep = () => {
        switch (step) {
            case 0:
                return <WelcomeStep onNext={handleNext} />;
            case 1:
                return (
                    <StylePersonalityStep
                        selected={stylePersonality}
                        onSelect={setStylePersonality}
                        onNext={handleNext}
                        onBack={handleBack}
                    />
                );
            case 2:
                return (
                    <ColorPreferencesStep
                        favoriteColors={favoriteColors}
                        onToggleColor={toggleColor}
                        onNext={handleNext}
                        onBack={handleBack}
                    />
                );
            case 3:
                return (
                    <OccasionsStep
                        selectedOccasions={occasions}
                        onToggleOccasion={toggleOccasion}
                        onNext={handleNext}
                        onBack={handleBack}
                    />
                );
            case 4:
                return (
                    <FitPreferenceStep
                        selected={fitPreference}
                        onSelect={setFitPreference}
                        onNext={handleNext}
                        onBack={handleBack}
                    />
                );
            case 5:
                return (
                    <GoalsStep
                        selectedGoals={goals}
                        onToggleGoal={toggleGoal}
                        onComplete={handleComplete}
                    />
                );
            default:
                return null;
        }
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {step > 0 && <ProgressBar currentStep={step - 1} totalSteps={TOTAL_STEPS - 1} />}

                <TouchableOpacity
                    style={styles.skipButton}
                    onPress={() => {
                        completeOnboarding();
                        navigation.reset({
                            index: 0,
                            routes: [{ name: 'Main' as never }],
                        });
                    }}
                >
                    <Text style={styles.skipText}>Skip</Text>
                </TouchableOpacity>

                {renderStep()}
            </SafeAreaView>
        </View>
    );
};

// ============================================
// STYLES
// ============================================

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,
    },
    safeArea: {
        flex: 1,
    },
    progressContainer: {
        paddingHorizontal: 24,
        paddingTop: 12,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    progressBar: {
        flex: 1,
        height: 4,
        backgroundColor: COLORS.border,
        borderRadius: 2,
        overflow: 'hidden',
    },
    progressFill: {
        height: '100%',
        backgroundColor: COLORS.primary,
        borderRadius: 2,
    },
    progressText: {
        fontSize: 13,
        color: COLORS.textSecondary,
    },
    skipButton: {
        position: 'absolute',
        right: 20,
        top: 60,
        zIndex: 10,
    },
    skipText: {
        fontSize: 16,
        color: COLORS.textSecondary,
    },
    stepContainer: {
        flex: 1,
        padding: 24,
    },
    stepTitle: {
        fontSize: 28,
        fontWeight: '700',
        color: COLORS.text,
        marginBottom: 8,
        marginTop: 20,
    },
    stepSubtitle: {
        fontSize: 16,
        color: COLORS.textSecondary,
        marginBottom: 24,
    },

    // Welcome
    welcomeContent: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
    },
    welcomeEmoji: {
        fontSize: 64,
        marginBottom: 20,
    },
    welcomeTitle: {
        fontSize: 32,
        fontWeight: '700',
        color: COLORS.text,
        textAlign: 'center',
        marginBottom: 12,
    },
    welcomeSubtitle: {
        fontSize: 18,
        color: COLORS.textSecondary,
        textAlign: 'center',
        marginBottom: 40,
    },
    benefitsList: {
        gap: 16,
    },
    benefitItem: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    benefitText: {
        fontSize: 16,
        color: COLORS.text,
    },

    // Buttons
    primaryButton: {
        marginTop: 'auto',
    },
    gradientButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 16,
        borderRadius: 16,
        gap: 8,
    },
    primaryButtonText: {
        fontSize: 18,
        fontWeight: '600',
        color: '#FFF',
    },
    buttonRow: {
        flexDirection: 'row',
        gap: 12,
        marginTop: 'auto',
    },
    backButton: {
        width: 56,
        height: 56,
        borderRadius: 16,
        backgroundColor: COLORS.surface,
        alignItems: 'center',
        justifyContent: 'center',
    },
    nextButton: {
        flex: 1,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: COLORS.primary,
        paddingVertical: 16,
        borderRadius: 16,
        gap: 8,
    },
    nextButtonText: {
        fontSize: 18,
        fontWeight: '600',
        color: '#FFF',
    },
    buttonDisabled: {
        opacity: 0.5,
    },
    completeButton: {
        marginTop: 'auto',
    },

    // Options
    optionsScroll: {
        flex: 1,
    },
    optionsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 12,
    },

    // Personality cards
    personalityCard: {
        width: (width - 60) / 2,
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 16,
        alignItems: 'center',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    personalityCardSelected: {
        borderColor: COLORS.primary,
        backgroundColor: `${COLORS.primary}10`,
    },
    personalityEmoji: {
        fontSize: 36,
        marginBottom: 8,
    },
    personalityName: {
        fontSize: 16,
        fontWeight: '600',
        color: COLORS.text,
        marginBottom: 4,
    },
    personalityDesc: {
        fontSize: 13,
        color: COLORS.textSecondary,
        textAlign: 'center',
    },
    selectedBadge: {
        position: 'absolute',
        top: 8,
        right: 8,
        width: 24,
        height: 24,
        borderRadius: 12,
        backgroundColor: COLORS.primary,
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Colors
    colorGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 16,
        justifyContent: 'center',
    },
    colorOption: {
        width: 60,
        height: 60,
        borderRadius: 30,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 3,
        borderColor: 'transparent',
    },
    colorOptionSelected: {
        borderColor: COLORS.primary,
    },
    helperText: {
        fontSize: 14,
        color: COLORS.textSecondary,
        textAlign: 'center',
        marginTop: 16,
    },

    // Occasions
    occasionGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        gap: 12,
    },
    occasionCard: {
        width: (width - 60) / 2,
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 16,
        alignItems: 'center',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    occasionCardSelected: {
        borderColor: COLORS.primary,
    },
    occasionIconContainer: {
        width: 56,
        height: 56,
        borderRadius: 28,
        backgroundColor: `${COLORS.textSecondary}20`,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
    },
    occasionIconSelected: {
        backgroundColor: COLORS.primary,
    },
    occasionName: {
        fontSize: 14,
        color: COLORS.textSecondary,
        textAlign: 'center',
    },
    occasionNameSelected: {
        color: COLORS.text,
        fontWeight: '600',
    },

    // Fit options
    fitOptions: {
        gap: 12,
    },
    fitCard: {
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 20,
        flexDirection: 'row',
        alignItems: 'center',
        borderWidth: 2,
        borderColor: 'transparent',
    },
    fitCardSelected: {
        borderColor: COLORS.primary,
    },
    fitEmoji: {
        fontSize: 32,
        marginRight: 16,
    },
    fitName: {
        fontSize: 18,
        fontWeight: '600',
        color: COLORS.text,
    },
    fitDesc: {
        fontSize: 14,
        color: COLORS.textSecondary,
        marginLeft: 'auto',
    },

    // Goals
    goalsGrid: {
        gap: 12,
    },
    goalCard: {
        backgroundColor: COLORS.surface,
        borderRadius: 16,
        padding: 16,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
        borderWidth: 2,
        borderColor: 'transparent',
    },
    goalCardSelected: {
        borderColor: COLORS.primary,
        backgroundColor: `${COLORS.primary}10`,
    },
    goalText: {
        fontSize: 16,
        color: COLORS.textSecondary,
    },
    goalTextSelected: {
        color: COLORS.text,
        fontWeight: '600',
    },

    // Chips
    chip: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: COLORS.surface,
        paddingHorizontal: 16,
        paddingVertical: 10,
        borderRadius: 24,
        gap: 8,
        borderWidth: 1,
        borderColor: COLORS.border,
    },
    chipSelected: {
        borderColor: COLORS.primary,
        backgroundColor: `${COLORS.primary}10`,
    },
    chipEmoji: {
        fontSize: 16,
    },
    chipText: {
        fontSize: 14,
        color: COLORS.textSecondary,
    },
    chipTextSelected: {
        color: COLORS.text,
        fontWeight: '600',
    },
    colorDot: {
        width: 16,
        height: 16,
        borderRadius: 8,
    },
});

export default StyleQuizScreen;
