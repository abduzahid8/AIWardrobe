/**
 * TypingIndicator — Animated dots shown while AI is thinking.
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withRepeat,
    withTiming,
    withDelay,
    withSequence,
} from 'react-native-reanimated';
import { useEffect } from 'react';

interface TypingIndicatorProps {
    color?: string;
}

export const TypingIndicator: React.FC<TypingIndicatorProps> = ({ color = '#6B7280' }) => {
    const dot1 = useSharedValue(0);
    const dot2 = useSharedValue(0);
    const dot3 = useSharedValue(0);

    useEffect(() => {
        dot1.value = withRepeat(
            withSequence(withTiming(-6, { duration: 300 }), withTiming(0, { duration: 300 })),
            -1, true
        );
        dot2.value = withDelay(150,
            withRepeat(
                withSequence(withTiming(-6, { duration: 300 }), withTiming(0, { duration: 300 })),
                -1, true
            )
        );
        dot3.value = withDelay(300,
            withRepeat(
                withSequence(withTiming(-6, { duration: 300 }), withTiming(0, { duration: 300 })),
                -1, true
            )
        );
    }, []);

    const style1 = useAnimatedStyle(() => ({ transform: [{ translateY: dot1.value }] }));
    const style2 = useAnimatedStyle(() => ({ transform: [{ translateY: dot2.value }] }));
    const style3 = useAnimatedStyle(() => ({ transform: [{ translateY: dot3.value }] }));

    return (
        <View style={styles.container}>
            <Animated.View style={[styles.dot, { backgroundColor: color }, style1]} />
            <Animated.View style={[styles.dot, { backgroundColor: color }, style2]} />
            <Animated.View style={[styles.dot, { backgroundColor: color }, style3]} />
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        paddingVertical: 8,
        paddingHorizontal: 12,
    },
    dot: {
        width: 6,
        height: 6,
        borderRadius: 3,
    },
});

export default TypingIndicator;
