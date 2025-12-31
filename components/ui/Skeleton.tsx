import React, { useEffect } from 'react';
import { View, StyleSheet, Dimensions } from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withTiming,
    interpolate,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, borderRadius, spacing } from '../../src/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

interface SkeletonProps {
    width?: number | string;
    height?: number;
    borderRadius?: number;
    style?: any;
}

// Base skeleton component with shimmer animation
export const Skeleton: React.FC<SkeletonProps> = ({
    width = '100%',
    height = 20,
    borderRadius: radius = 8,
    style,
}) => {
    const shimmer = useSharedValue(0);

    useEffect(() => {
        shimmer.value = withRepeat(
            withTiming(1, { duration: 1500 }),
            -1,
            false
        );
    }, []);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [
            {
                translateX: interpolate(
                    shimmer.value,
                    [0, 1],
                    [-SCREEN_WIDTH, SCREEN_WIDTH]
                ),
            },
        ],
    }));

    return (
        <View
            style={[
                styles.skeleton,
                {
                    width: width as any,
                    height,
                    borderRadius: radius,
                },
                style,
            ]}
        >
            <Animated.View style={[styles.shimmer, animatedStyle]}>
                <LinearGradient
                    colors={['transparent', 'rgba(255,255,255,0.3)', 'transparent']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={StyleSheet.absoluteFill}
                />
            </Animated.View>
        </View>
    );
};

// Skeleton for clothing cards
export const ClothingCardSkeleton: React.FC = () => (
    <View style={styles.cardSkeleton}>
        <Skeleton height={140} borderRadius={borderRadius.l} />
        <View style={styles.cardContent}>
            <Skeleton width="70%" height={16} style={{ marginBottom: 8 }} />
            <Skeleton width="40%" height={12} />
        </View>
    </View>
);

// Skeleton for outfit cards
export const OutfitCardSkeleton: React.FC = () => (
    <View style={styles.outfitSkeleton}>
        <Skeleton height={180} borderRadius={borderRadius.xl} />
        <View style={styles.outfitContent}>
            <Skeleton width="60%" height={18} style={{ marginBottom: 8 }} />
            <Skeleton width="80%" height={14} style={{ marginBottom: 6 }} />
            <Skeleton width="50%" height={12} />
        </View>
    </View>
);

// Skeleton for celebrity cards
export const CelebCardSkeleton: React.FC = () => (
    <View style={styles.celebSkeleton}>
        <Skeleton height={320} borderRadius={borderRadius.xl} />
    </View>
);

// Skeleton for profile header
export const ProfileHeaderSkeleton: React.FC = () => (
    <View style={styles.profileSkeleton}>
        <Skeleton width={100} height={100} borderRadius={50} />
        <View style={styles.statsRow}>
            <View style={styles.statSkeleton}>
                <Skeleton width={40} height={24} style={{ marginBottom: 4 }} />
                <Skeleton width={50} height={14} />
            </View>
            <View style={styles.statSkeleton}>
                <Skeleton width={40} height={24} style={{ marginBottom: 4 }} />
                <Skeleton width={50} height={14} />
            </View>
            <View style={styles.statSkeleton}>
                <Skeleton width={40} height={24} style={{ marginBottom: 4 }} />
                <Skeleton width={50} height={14} />
            </View>
        </View>
    </View>
);

// Grid skeleton for loading states
export const GridSkeleton: React.FC<{ count?: number }> = ({ count = 6 }) => (
    <View style={styles.gridContainer}>
        {Array.from({ length: count }).map((_, index) => (
            <ClothingCardSkeleton key={index} />
        ))}
    </View>
);

// List skeleton for loading states
export const ListSkeleton: React.FC<{ count?: number }> = ({ count = 3 }) => (
    <View style={styles.listContainer}>
        {Array.from({ length: count }).map((_, index) => (
            <View key={index} style={styles.listItem}>
                <Skeleton width={60} height={60} borderRadius={12} />
                <View style={styles.listContent}>
                    <Skeleton width="70%" height={16} style={{ marginBottom: 6 }} />
                    <Skeleton width="50%" height={12} />
                </View>
            </View>
        ))}
    </View>
);

const styles = StyleSheet.create({
    skeleton: {
        backgroundColor: colors.surfaceHighlight || '#EEECE8',
        overflow: 'hidden',
    },
    shimmer: {
        ...StyleSheet.absoluteFillObject,
        width: SCREEN_WIDTH * 2,
    },
    cardSkeleton: {
        width: (SCREEN_WIDTH - spacing.l * 3) / 2,
        marginBottom: spacing.m,
    },
    cardContent: {
        paddingTop: spacing.s,
    },
    outfitSkeleton: {
        width: SCREEN_WIDTH * 0.8,
        marginRight: spacing.m,
    },
    outfitContent: {
        paddingTop: spacing.m,
    },
    celebSkeleton: {
        width: SCREEN_WIDTH * 0.8,
        marginRight: spacing.m,
    },
    profileSkeleton: {
        alignItems: 'center',
        paddingVertical: spacing.xl,
    },
    statsRow: {
        flexDirection: 'row',
        marginTop: spacing.l,
        gap: spacing.xl,
    },
    statSkeleton: {
        alignItems: 'center',
    },
    gridContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        paddingHorizontal: spacing.l,
        gap: spacing.m,
    },
    listContainer: {
        paddingHorizontal: spacing.l,
    },
    listItem: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: spacing.m,
    },
    listContent: {
        flex: 1,
        marginLeft: spacing.m,
    },
});

export default Skeleton;
