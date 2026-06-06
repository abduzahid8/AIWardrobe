import React, { useEffect, useState } from 'react';
import { View, StyleSheet, Dimensions,  } from 'react-native'
import { ScaledText } from './ui/ScaledText';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withDelay,
    FadeInUp,
    FadeOutDown,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import { useTranslation } from 'react-i18next';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width } = Dimensions.get('window');

interface OnboardingProgressProps {
    targetItems?: number;
    currentItems: number;
    onComplete?: () => void;
}

/**
 * OnboardingProgress - Alta-style gamified toast
 * 
 * Shows progress toward unlocking personalized looks:
 * "Add X items to unlock personalized daily looks"
 * 
 * Features:
 * - Animated progress bar
 * - Auto-dismisses when goal reached
 * - Persists dismissed state
 */
const OnboardingProgress: React.FC<OnboardingProgressProps> = ({
    targetItems = 5,
    currentItems,
    onComplete,
}) => {
    const { t } = useTranslation();
    const [dismissed, setDismissed] = useState(false);
    const [visible, setVisible] = useState(true);
    const progressWidth = useSharedValue(0);

    const progress = Math.min(currentItems / targetItems, 1);
    const remaining = Math.max(targetItems - currentItems, 0);
    const isComplete = currentItems >= targetItems;

    useEffect(() => {
        // Check if already dismissed
        AsyncStorage.getItem('onboardingDismissed').then((value) => {
            if (value === 'true') {
                setDismissed(true);
                setVisible(false);
            }
        });
    }, []);

    useEffect(() => {
        // Animate progress bar
        progressWidth.value = withDelay(300, withSpring(progress * 100, {
            damping: 15,
            stiffness: 100,
        }));

        // Auto-dismiss and call onComplete when goal reached
        if (isComplete && !dismissed) {
            setTimeout(() => {
                setVisible(false);
                AsyncStorage.setItem('onboardingDismissed', 'true');
                onComplete?.();
            }, 2000);
        }
    }, [progress, isComplete]);

    const progressAnimatedStyle = useAnimatedStyle(() => ({
        width: `${progressWidth.value}%`,
    }));

    if (!visible || dismissed) return null;

    return (
        <Animated.View
            entering={FadeInUp.springify().delay(500)}
            exiting={FadeOutDown.springify()}
            style={styles.container}
        >
            <View style={styles.content}>
                <View style={styles.iconContainer}>
                    <Ionicons
                        name={isComplete ? "checkmark-circle" : "sparkles"}
                        size={20}
                        color={isComplete ? "#22C55E" : "#0A1931"}
                    />
                </View>
                <View style={styles.textContainer}>
                    <ScaledText style={styles.title}>
                        {isComplete
                            ? t('onboarding.personalizedLooksUnlocked')
                            : t('onboarding.addItemsToUnlock', { 
                                count: remaining,
                                suffix: remaining !== 1 ? 's' : ''
                            })
                        }
                    </ScaledText>
                </View>
            </View>

            <View style={styles.progressBarBg}>
                <Animated.View style={[styles.progressBarFill, progressAnimatedStyle]} />
            </View>

            <ScaledText style={styles.counter}>{t('onboarding.itemsCounter', { current: currentItems, target: targetItems })}</ScaledText>
        </Animated.View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'absolute',
        bottom: 100,
        left: 20,
        right: 20,
        backgroundColor: '#0A1931',
        borderRadius: 16,
        padding: 16,
        shadowColor: '#0A1931',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.25,
        shadowRadius: 16,
        elevation: 10,
    },
    content: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 12,
    },
    iconContainer: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: '#F5F5F5',
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 12,
    },
    textContainer: {
        flex: 1,
    },
    title: {
        fontSize: 14,
        fontWeight: '600',
        color: '#FFFFFF',
        lineHeight: 20,
    },
    progressBarBg: {
        height: 6,
        backgroundColor: '#333',
        borderRadius: 3,
        overflow: 'hidden',
    },
    progressBarFill: {
        height: '100%',
        backgroundColor: '#FFFFFF',
        borderRadius: 3,
    },
    counter: {
        fontSize: 12,
        color: '#8E8E8E',
        marginTop: 8,
        textAlign: 'center',
    },
});

export default OnboardingProgress;
