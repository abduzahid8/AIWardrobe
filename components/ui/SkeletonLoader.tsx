import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet, ViewStyle, DimensionValue } from 'react-native';
import { useTranslation } from 'react-i18next';

interface SkeletonLoaderProps {
    width?: DimensionValue;
    height?: number;
    borderRadius?: number;
    style?: ViewStyle;
}

/**
 * SkeletonLoader — animated shimmer placeholder for loading states.
 * Replace ActivityIndicator with this for a polished loading experience.
 *
 * @example
 * <SkeletonLoader width="100%" height={200} borderRadius={16} />
 * <SkeletonLoader width={120} height={20} borderRadius={8} />
 */
export const SkeletonLoader: React.FC<SkeletonLoaderProps> = ({
    width = '100%',
    height = 20,
    borderRadius = 8,
    style,
}) => {
    const { t } = useTranslation();
    const shimmerAnim = useRef(new Animated.Value(0)).current;

    useEffect(() => {
        const animation = Animated.loop(
            Animated.sequence([
                Animated.timing(shimmerAnim, {
                    toValue: 1,
                    duration: 1000,
                    useNativeDriver: true,
                }),
                Animated.timing(shimmerAnim, {
                    toValue: 0,
                    duration: 1000,
                    useNativeDriver: true,
                }),
            ])
        );
        animation.start();
        return () => animation.stop();
    }, [shimmerAnim]);

    const opacity = shimmerAnim.interpolate({
        inputRange: [0, 1],
        outputRange: [0.3, 0.7],
    });

    return (
        <Animated.View
            style={[
                styles.skeleton,
                { width, height, borderRadius, opacity },
                style,
            ]}
            accessibilityLabel={t('common.loading')}
            accessibilityRole="progressbar"
        />
    );
};

/**
 * SkeletonCard — pre-built skeleton layout for card-like content
 */
export const SkeletonCard: React.FC<{ style?: ViewStyle }> = ({ style }) => (
    <View style={[styles.card, style]}>
        <SkeletonLoader width="100%" height={160} borderRadius={12} />
        <View style={styles.cardContent}>
            <SkeletonLoader width="70%" height={16} borderRadius={6} />
            <SkeletonLoader width="40%" height={12} borderRadius={6} style={{ marginTop: 8 }} />
        </View>
    </View>
);

/**
 * SkeletonList — renders multiple skeleton items for list loading states
 */
export const SkeletonList: React.FC<{ count?: number; style?: ViewStyle }> = ({
    count = 4,
    style,
}) => (
    <View style={style}>
        {Array.from({ length: count }).map((_, i) => (
            <View key={i} style={styles.listItem}>
                <SkeletonLoader width={48} height={48} borderRadius={24} />
                <View style={styles.listItemContent}>
                    <SkeletonLoader width="60%" height={14} borderRadius={6} />
                    <SkeletonLoader width="80%" height={10} borderRadius={4} style={{ marginTop: 6 }} />
                </View>
            </View>
        ))}
    </View>
);

const styles = StyleSheet.create({
    skeleton: {
        backgroundColor: '#E0E0E0',
    },
    card: {
        marginBottom: 16,
    },
    cardContent: {
        marginTop: 12,
    },
    listItem: {
        flexDirection: 'row',
        alignItems: 'center',
        paddingVertical: 12,
    },
    listItemContent: {
        flex: 1,
        marginLeft: 12,
    },
});

export default SkeletonLoader;
