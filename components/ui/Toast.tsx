import React, { useEffect, useCallback } from 'react';
import {
    View,
    Text,
    StyleSheet,
    Dimensions,
    TouchableOpacity,
} from 'react-native';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withSpring,
    withTiming,
    runOnJS,
    withSequence,
} from 'react-native-reanimated';
import { Gesture, GestureDetector } from 'react-native-gesture-handler';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { colors, spacing, shadows, animations } from '../../src/theme';

const { width: SCREEN_WIDTH } = Dimensions.get('window');
const TOAST_MARGIN = spacing.m;
const TOAST_WIDTH = SCREEN_WIDTH - TOAST_MARGIN * 2;

type ToastType = 'success' | 'error' | 'warning' | 'info';

interface ToastConfig {
    id: string;
    type: ToastType;
    title: string;
    message?: string;
    duration?: number;
    onPress?: () => void;
}

interface ToastProps extends ToastConfig {
    onDismiss: (id: string) => void;
}

interface ToastIconConfig {
    name: keyof typeof Ionicons.glyphMap;
    color: string;
    bgColor: string;
}

const getToastConfig = (type: ToastType): ToastIconConfig => {
    switch (type) {
        case 'success':
            return {
                name: 'checkmark-circle',
                color: '#34C759',
                bgColor: 'rgba(52, 199, 89, 0.1)',
            };
        case 'error':
            return {
                name: 'close-circle',
                color: '#FF3B30',
                bgColor: 'rgba(255, 59, 48, 0.1)',
            };
        case 'warning':
            return {
                name: 'warning',
                color: '#FF9500',
                bgColor: 'rgba(255, 149, 0, 0.1)',
            };
        case 'info':
        default:
            return {
                name: 'information-circle',
                color: '#007AFF',
                bgColor: 'rgba(0, 122, 255, 0.1)',
            };
    }
};

/**
 * Individual Toast component
 */
const Toast: React.FC<ToastProps> = ({
    id,
    type,
    title,
    message,
    duration = 4000,
    onPress,
    onDismiss,
}) => {
    const insets = useSafeAreaInsets();
    const translateY = useSharedValue(-100);
    const translateX = useSharedValue(0);
    const opacity = useSharedValue(0);

    const config = getToastConfig(type);

    const dismiss = useCallback(() => {
        translateY.value = withTiming(-100, { duration: 200 });
        opacity.value = withTiming(0, { duration: 200 }, () => {
            runOnJS(onDismiss)(id);
        });
    }, [id, onDismiss]);

    useEffect(() => {
        // Animate in
        translateY.value = withSpring(0, animations.spring);
        opacity.value = withTiming(1, { duration: 200 });

        // Haptic feedback
        Haptics.notificationAsync(
            type === 'success'
                ? Haptics.NotificationFeedbackType.Success
                : type === 'error'
                    ? Haptics.NotificationFeedbackType.Error
                    : Haptics.NotificationFeedbackType.Warning
        );

        // Auto dismiss
        const timeout = setTimeout(() => {
            dismiss();
        }, duration);

        return () => clearTimeout(timeout);
    }, []);

    const gesture = Gesture.Pan()
        .onUpdate((event) => {
            // Allow swiping up or horizontally
            if (event.translationY < 0) {
                translateY.value = event.translationY;
            }
            translateX.value = event.translationX;
        })
        .onEnd((event) => {
            // Dismiss if swiped up or far enough horizontally
            if (event.translationY < -50 || Math.abs(event.translationX) > 100) {
                runOnJS(dismiss)();
            } else {
                translateY.value = withSpring(0, animations.spring);
                translateX.value = withSpring(0, animations.spring);
            }
        });

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [
            { translateY: translateY.value },
            { translateX: translateX.value },
        ],
        opacity: opacity.value,
    }));

    return (
        <GestureDetector gesture={gesture}>
            <Animated.View
                style={[
                    styles.toast,
                    { top: insets.top + spacing.s },
                    animatedStyle,
                ]}
            >
                <TouchableOpacity
                    onPress={() => {
                        onPress?.();
                        dismiss();
                    }}
                    activeOpacity={0.9}
                    style={styles.toastContent}
                >
                    <View style={[styles.iconContainer, { backgroundColor: config.bgColor }]}>
                        <Ionicons name={config.name} size={24} color={config.color} />
                    </View>
                    <View style={styles.textContainer}>
                        <Text style={styles.title}>{title}</Text>
                        {message && <Text style={styles.message}>{message}</Text>}
                    </View>
                    <TouchableOpacity onPress={dismiss} style={styles.closeButton}>
                        <Ionicons name="close" size={20} color={colors.text.secondary} />
                    </TouchableOpacity>
                </TouchableOpacity>
            </Animated.View>
        </GestureDetector>
    );
};

// Toast Manager for showing toasts from anywhere
class ToastManager {
    private static instance: ToastManager;
    private showToastCallback: ((config: Omit<ToastConfig, 'id'>) => void) | null = null;

    static getInstance(): ToastManager {
        if (!ToastManager.instance) {
            ToastManager.instance = new ToastManager();
        }
        return ToastManager.instance;
    }

    setShowToastCallback(callback: (config: Omit<ToastConfig, 'id'>) => void) {
        this.showToastCallback = callback;
    }

    show(config: Omit<ToastConfig, 'id'>) {
        this.showToastCallback?.(config);
    }

    success(title: string, message?: string) {
        this.show({ type: 'success', title, message });
    }

    error(title: string, message?: string) {
        this.show({ type: 'error', title, message });
    }

    warning(title: string, message?: string) {
        this.show({ type: 'warning', title, message });
    }

    info(title: string, message?: string) {
        this.show({ type: 'info', title, message });
    }
}

export const toast = ToastManager.getInstance();

/**
 * Toast container component - place at root of app
 * 
 * @example
 * // In App.tsx
 * <ToastContainer />
 * 
 * // From anywhere in the app
 * toast.success('Saved!', 'Your outfit has been saved.');
 * toast.error('Error', 'Something went wrong.');
 */
export const ToastContainer: React.FC = () => {
    const [toasts, setToasts] = React.useState<ToastConfig[]>([]);

    useEffect(() => {
        toast.setShowToastCallback((config) => {
            const id = `${Date.now()}-${Math.random()}`;
            setToasts((prev) => [...prev, { ...config, id }]);
        });
    }, []);

    const handleDismiss = (id: string) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    };

    return (
        <View style={styles.container} pointerEvents="box-none">
            {toasts.map((toastConfig) => (
                <Toast key={toastConfig.id} {...toastConfig} onDismiss={handleDismiss} />
            ))}
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
    },
    toast: {
        position: 'absolute',
        left: TOAST_MARGIN,
        right: TOAST_MARGIN,
        backgroundColor: colors.surface,
        borderRadius: 16,
        ...shadows.medium,
    },
    toastContent: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: spacing.m,
    },
    iconContainer: {
        width: 40,
        height: 40,
        borderRadius: 20,
        justifyContent: 'center',
        alignItems: 'center',
        marginRight: spacing.m,
    },
    textContainer: {
        flex: 1,
    },
    title: {
        fontSize: 15,
        fontWeight: '600',
        color: colors.text.primary,
    },
    message: {
        fontSize: 13,
        color: colors.text.secondary,
        marginTop: 2,
    },
    closeButton: {
        padding: spacing.xs,
    },
});

export default ToastContainer;
