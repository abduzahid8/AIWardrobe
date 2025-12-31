import React, { ReactNode } from 'react';
import { View, StyleSheet, StatusBar, Platform } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import Animated, {
    FadeIn,
    FadeInDown,
    FadeInUp,
    SlideInRight,
    useAnimatedStyle,
    useSharedValue,
    withTiming,
} from 'react-native-reanimated';
import { useTheme } from '../../src/theme/ThemeContext';

interface ScreenWrapperProps {
    children: ReactNode;
    /** Animation type for entrance */
    animation?: 'fade' | 'slideUp' | 'slideDown' | 'slideRight' | 'none';
    /** Delay before animation starts (ms) */
    delay?: number;
    /** Whether to use SafeAreaView edges */
    safeAreaEdges?: ('top' | 'bottom' | 'left' | 'right')[];
    /** Custom background color */
    backgroundColor?: string;
    /** Custom style */
    style?: any;
    /** Whether to show status bar */
    statusBar?: boolean;
}

/**
 * ScreenWrapper - Premium screen container with entrance animations
 * 
 * Provides consistent:
 * - Entrance animations
 * - Safe area handling
 * - Theme-aware backgrounds
 * - Status bar styling
 */
export const ScreenWrapper: React.FC<ScreenWrapperProps> = ({
    children,
    animation = 'fade',
    delay = 0,
    safeAreaEdges = ['top'],
    backgroundColor,
    style,
    statusBar = true,
}) => {
    const { isDark, colors } = useTheme();

    const getEnteringAnimation = () => {
        switch (animation) {
            case 'slideUp':
                return FadeInUp.delay(delay).duration(400).springify();
            case 'slideDown':
                return FadeInDown.delay(delay).duration(400).springify();
            case 'slideRight':
                return SlideInRight.delay(delay).duration(300);
            case 'none':
                return undefined;
            case 'fade':
            default:
                return FadeIn.delay(delay).duration(350);
        }
    };

    const bgColor = backgroundColor || colors.background;

    return (
        <View style={[styles.container, { backgroundColor: bgColor }]}>
            {statusBar && (
                <StatusBar
                    barStyle={isDark ? 'light-content' : 'dark-content'}
                    backgroundColor="transparent"
                    translucent
                />
            )}

            <SafeAreaView style={styles.safeArea} edges={safeAreaEdges}>
                <Animated.View
                    entering={getEnteringAnimation()}
                    style={[styles.content, style]}
                >
                    {children}
                </Animated.View>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    safeArea: {
        flex: 1,
    },
    content: {
        flex: 1,
    },
});

export default ScreenWrapper;
