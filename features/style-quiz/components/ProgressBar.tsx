/**
 * ProgressBar — Reusable quiz progress indicator.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import Animated from 'react-native-reanimated';
import AppColors from '../../../constants/AppColors';

interface ProgressBarProps {
    currentStep: number;
    totalSteps: number;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({ currentStep, totalSteps }) => {
    const progress = (currentStep + 1) / totalSteps;

    return (
        <View style={styles.container}>
            <View style={styles.bar}>
                <Animated.View style={[styles.fill, { width: `${progress * 100}%` }]} />
            </View>
            <Text style={styles.text}>
                {currentStep + 1} of {totalSteps}
            </Text>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        paddingHorizontal: 24,
        paddingTop: 12,
        flexDirection: 'row',
        alignItems: 'center',
        gap: 12,
    },
    bar: {
        flex: 1,
        height: 4,
        backgroundColor: AppColors.border,
        borderRadius: 2,
        overflow: 'hidden',
    },
    fill: {
        height: '100%',
        backgroundColor: AppColors.primary,
        borderRadius: 2,
    },
    text: {
        fontSize: 13,
        color: AppColors.textSecondary,
    },
});

export default ProgressBar;
