import React from 'react';
import { Pressable, PressableProps, ViewStyle, StyleProp, GestureResponderEvent } from 'react-native';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from 'react-native-reanimated';
import * as Haptics from 'expo-haptics';

interface PressableScaleProps extends Omit<PressableProps, 'style'> {
    children: React.ReactNode;
    style?: StyleProp<ViewStyle>;
    scaleValue?: number;
    hapticFeedback?: boolean;
    springConfig?: {
        damping?: number;
        stiffness?: number;
    };
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

/**
 * PressableScale - Alta-style tactile button component
 * 
 * Provides a native-feeling press animation with:
 * - Scale down to 0.97 on press (configurable)
 * - Spring animation for smooth feel
 * - Optional haptic feedback
 * 
 * Usage:
 * ```tsx
 * <PressableScale onPress={() => console.log('pressed!')}>
 *   <Text>Click me</Text>
 * </PressableScale>
 * ```
 */
const PressableScale: React.FC<PressableScaleProps> = ({
    children,
    style,
    scaleValue = 0.97,
    hapticFeedback = true,
    springConfig = { damping: 15, stiffness: 400 },
    onPressIn,
    onPressOut,
    onPress,
    ...props
}) => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: scale.value }],
    }));

    const handlePressIn = (event: GestureResponderEvent) => {
        scale.value = withSpring(scaleValue, springConfig);
        if (hapticFeedback) {
            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        }
        onPressIn?.(event);
    };

    const handlePressOut = (event: GestureResponderEvent) => {
        scale.value = withSpring(1, springConfig);
        onPressOut?.(event);
    };

    return (
        <AnimatedPressable
            style={[animatedStyle, style]}
            onPressIn={handlePressIn}
            onPressOut={handlePressOut}
            onPress={onPress}
            {...props}
        >
            {children}
        </AnimatedPressable>
    );
};

export default PressableScale;
