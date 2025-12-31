import React, { useEffect } from 'react';
import {
    View,
    StyleSheet,
    ViewStyle,
    Dimensions,
    DimensionValue,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withTiming,
    withSequence,
    Easing,
    interpolate,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, spacing } from '../../src/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

type LoadingVariant = 'card' | 'list' | 'grid' | 'profile' | 'detail';

interface LoadingStateProps {
    variant?: LoadingVariant;
    count?: number;
    style?: ViewStyle;
}

interface SkeletonProps {
    width: DimensionValue;
    height: number;
    borderRadius?: number;
    style?: ViewStyle;
}

/**
 * Skeleton loading component with shimmer effect
 */
const Skeleton: React.FC<SkeletonProps> = ({
    width,
    height,
    borderRadius = 8,
    style,
}) => {
    const translateX = useSharedValue(-SCREEN_WIDTH);

    useEffect(() => {
        translateX.value = withRepeat(
            withTiming(SCREEN_WIDTH, {
                duration: 1200,
                easing: Easing.bezier(0.25, 0.1, 0.25, 1),
            }),
            -1, // Infinite
            false
        );
    }, []);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ translateX: translateX.value }],
    }));

    return (
        <View
            style={[
                styles.skeleton,
                {
                    width,
                    height,
                    borderRadius,
                },
                style,
            ]}
        >
            <Animated.View style={[styles.shimmer, animatedStyle]}>
                <LinearGradient
                    colors={['transparent', 'rgba(255,255,255,0.4)', 'transparent']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 0 }}
                    style={styles.gradient}
                />
            </Animated.View>
        </View>
    );
};

/**
 * Card skeleton for loading states
 */
const CardSkeleton: React.FC = () => (
    <View style={styles.cardSkeleton}>
        <Skeleton width="100%" height={150} borderRadius={12} />
        <View style={styles.cardContent}>
            <Skeleton width="70%" height={16} style={{ marginBottom: 8 }} />
            <Skeleton width="40%" height={14} />
        </View>
    </View>
);

/**
 * List item skeleton
 */
const ListSkeleton: React.FC = () => (
    <View style={styles.listSkeleton}>
        <Skeleton width={60} height={60} borderRadius={30} />
        <View style={styles.listContent}>
            <Skeleton width="60%" height={16} style={{ marginBottom: 8 }} />
            <Skeleton width="40%" height={14} />
        </View>
        <Skeleton width={24} height={24} borderRadius={12} />
    </View>
);

/**
 * Grid item skeleton
 */
const GridSkeleton: React.FC = () => (
    <View style={styles.gridSkeleton}>
        <Skeleton width="100%" height={120} borderRadius={12} />
        <Skeleton width="80%" height={14} style={{ marginTop: 8 }} />
    </View>
);

/**
 * Profile skeleton
 */
const ProfileSkeleton: React.FC = () => (
    <View style={styles.profileSkeleton}>
        <Skeleton width={80} height={80} borderRadius={40} />
        <Skeleton width={120} height={20} style={{ marginTop: 12 }} />
        <Skeleton width={180} height={14} style={{ marginTop: 8 }} />
        <View style={styles.profileStats}>
            <View style={styles.profileStat}>
                <Skeleton width={40} height={24} />
                <Skeleton width={60} height={14} style={{ marginTop: 4 }} />
            </View>
            <View style={styles.profileStat}>
                <Skeleton width={40} height={24} />
                <Skeleton width={60} height={14} style={{ marginTop: 4 }} />
            </View>
            <View style={styles.profileStat}>
                <Skeleton width={40} height={24} />
                <Skeleton width={60} height={14} style={{ marginTop: 4 }} />
            </View>
        </View>
    </View>
);

/**
 * Detail page skeleton
 */
const DetailSkeleton: React.FC = () => (
    <View style={styles.detailSkeleton}>
        <Skeleton width="100%" height={300} borderRadius={0} />
        <View style={styles.detailContent}>
            <Skeleton width="80%" height={24} style={{ marginBottom: 12 }} />
            <Skeleton width="50%" height={18} style={{ marginBottom: 20 }} />
            <Skeleton width="100%" height={100} borderRadius={12} />
        </View>
    </View>
);

/**
 * Loading state component with multiple variants
 * 
 * @example
 * <LoadingState variant="card" count={3} />
 * 
 * @example
 * <LoadingState variant="grid" count={6} />
 */
export const LoadingState: React.FC<LoadingStateProps> = ({
    variant = 'card',
    count = 3,
    style,
}) => {
    const renderSkeleton = () => {
        switch (variant) {
            case 'list':
                return <ListSkeleton />;
            case 'grid':
                return <GridSkeleton />;
            case 'profile':
                return <ProfileSkeleton />;
            case 'detail':
                return <DetailSkeleton />;
            case 'card':
            default:
                return <CardSkeleton />;
        }
    };

    // Profile and detail are single items
    if (variant === 'profile' || variant === 'detail') {
        return <View style={style}>{renderSkeleton()}</View>;
    }

    // Grid layout
    if (variant === 'grid') {
        return (
            <View style={[styles.gridContainer, style]}>
                {Array.from({ length: count }).map((_, index) => (
                    <View key={index} style={styles.gridItem}>
                        {renderSkeleton()}
                    </View>
                ))}
            </View>
        );
    }

    // List layout
    return (
        <View style={style}>
            {Array.from({ length: count }).map((_, index) => (
                <View key={index}>
                    {renderSkeleton()}
                </View>
            ))}
        </View>
    );
};

const styles = StyleSheet.create({
    skeleton: {
        backgroundColor: colors.surfaceHighlight,
        overflow: 'hidden',
    },
    shimmer: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        width: SCREEN_WIDTH,
    },
    gradient: {
        flex: 1,
        width: '100%',
    },
    // Card skeleton
    cardSkeleton: {
        marginBottom: spacing.m,
    },
    cardContent: {
        padding: spacing.m,
    },
    // List skeleton
    listSkeleton: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: spacing.m,
        marginBottom: spacing.s,
    },
    listContent: {
        flex: 1,
        marginLeft: spacing.m,
    },
    // Grid skeleton
    gridContainer: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        marginHorizontal: -spacing.xs,
    },
    gridItem: {
        width: '33.33%',
        padding: spacing.xs,
    },
    gridSkeleton: {
        padding: spacing.xs,
    },
    // Profile skeleton
    profileSkeleton: {
        alignItems: 'center',
        padding: spacing.l,
    },
    profileStats: {
        flexDirection: 'row',
        marginTop: spacing.l,
    },
    profileStat: {
        alignItems: 'center',
        paddingHorizontal: spacing.l,
    },
    // Detail skeleton
    detailSkeleton: {},
    detailContent: {
        padding: spacing.l,
    },
});

export default LoadingState;
