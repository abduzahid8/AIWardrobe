// Nandly-inspired Bottom Sheet Component
// Features: Gesture-driven, backdrop blur, spring animations, snap points

import React, { useEffect, useCallback } from 'react';
import {
    View,
    StyleSheet,
    Modal,
    Pressable,
    Dimensions,
    ViewStyle,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
    runOnJS,

    Easing,
} from 'react-native-reanimated';
import { PanGestureHandler, PanGestureHandlerGestureEvent, Gesture, GestureDetector } from 'react-native-gesture-handler';
import { BlurView } from 'expo-blur';
import { useTheme } from '../../src/hooks/useTheme';
import { springConfig } from '../../src/theme/animations';

const { height: SCREEN_HEIGHT } = Dimensions.get('window');

interface BottomSheetProps {
    /** Whether the bottom sheet is visible */
    visible: boolean;
    /** Callback when the bottom sheet should close */
    onClose: () => void;
    /** Bottom sheet content */
    children: React.ReactNode;
    /** Height of the bottom sheet (if null, uses snap point) */
    height?: number;
    /** Enable backdrop blur (iOS) */
    enableBlur?: boolean;
    /** Custom snap point (0-1, percentage of screen height) */
    snapPoint?: number;
    /** Custom style */
    style?: ViewStyle;
}

export const BottomSheet: React.FC<BottomSheetProps> = ({
    visible,
    onClose,
    children,
    height,
    enableBlur = true,
    snapPoint = 0.5,
    style,
}) => {
    const { colors } = useTheme();

    const translateY = useSharedValue(SCREEN_HEIGHT);
    const backdropOpacity = useSharedValue(0);

    const sheetHeight = height || SCREEN_HEIGHT * snapPoint;

    useEffect(() => {
        if (visible) {
            // Slide up
            translateY.value = withSpring(SCREEN_HEIGHT - sheetHeight, springConfig);
            backdropOpacity.value = withTiming(1, {
                duration: 250,
                easing: Easing.out(Easing.cubic),
            });
        } else {
            // Slide down
            translateY.value = withTiming(SCREEN_HEIGHT, {
                duration: 250,
                easing: Easing.in(Easing.cubic),
            });
            backdropOpacity.value = withTiming(0, {
                duration: 250,
            });
        }
    }, [visible, sheetHeight]);

    const gesture = Gesture.Pan()
        .onStart(() => {
            // Pan gesture started
        })
        .onUpdate((event) => {
            const newY = SCREEN_HEIGHT - sheetHeight + event.translationY;
            // Only allow dragging down
            if (newY > SCREEN_HEIGHT - sheetHeight) {
                translateY.value = newY;
            }
        })
        .onEnd((event) => {
            const threshold = sheetHeight * 0.3;

            if (event.translationY > threshold || event.velocityY > 500) {
                // Close the sheet
                translateY.value = withTiming(SCREEN_HEIGHT, {
                    duration: 250,
                    easing: Easing.in(Easing.cubic),
                });
                backdropOpacity.value = withTiming(0, { duration: 250 });
                runOnJS(onClose)();
            } else {
                // Snap back
                translateY.value = withSpring(SCREEN_HEIGHT - sheetHeight, springConfig);
            }
        });

    const animatedSheetStyle = useAnimatedStyle(() => ({
        transform: [{ translateY: translateY.value }],
    }));

    const animatedBackdropStyle = useAnimatedStyle(() => ({
        opacity: backdropOpacity.value,
    }));

    if (!visible) return null;

    return (
        <Modal
            visible={visible}
            transparent
            animationType="none"
            onRequestClose={onClose}
            statusBarTranslucent
        >
            <View style={styles.container}>
                {/* Backdrop */}
                <Pressable style={StyleSheet.absoluteFill} onPress={onClose}>
                    <Animated.View style={[StyleSheet.absoluteFill, animatedBackdropStyle]}>
                        {enableBlur ? (
                            <BlurView intensity={20} style={StyleSheet.absoluteFill} tint="dark" />
                        ) : (
                            <View style={[StyleSheet.absoluteFill, { backgroundColor: 'rgba(0,0,0,0.5)' }]} />
                        )}
                    </Animated.View>
                </Pressable>

                {/* Bottom Sheet */}
                <GestureDetector gesture={gesture}>
                    <Animated.View
                        style={[
                            styles.sheet,
                            {
                                height: sheetHeight,
                                backgroundColor: colors.surface,
                            },
                            animatedSheetStyle,
                            style,
                        ]}
                    >
                        {/* Handle */}
                        <View style={styles.handleContainer}>
                            <View style={[styles.handle, { backgroundColor: colors.border }]} />
                        </View>

                        {/* Content */}
                        <View style={styles.content}>{children}</View>
                    </Animated.View>
                </GestureDetector>
            </View>
        </Modal>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
    },
    sheet: {
        position: 'absolute',
        left: 0,
        right: 0,
        bottom: 0,
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.1,
        shadowRadius: 12,
        elevation: 8,
    },
    handleContainer: {
        paddingVertical: 12,
        alignItems: 'center',
    },
    handle: {
        width: 40,
        height: 4,
        borderRadius: 2,
    },
    content: {
        flex: 1,
        paddingHorizontal: 24,
        paddingBottom: 24,
    },
});
