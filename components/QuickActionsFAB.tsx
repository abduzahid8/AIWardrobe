import React, { useState } from 'react';
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Dimensions,
} from 'react-native';
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
    withTiming,
    interpolate,
    Extrapolate,
} from 'react-native-reanimated';
import { Ionicons } from '@expo/vector-icons';
import * as Haptics from 'expo-haptics';
import { BlurView } from 'expo-blur';
import { useNavigation } from '@react-navigation/native';
import { colors, spacing, shadows, animations } from '../src/theme';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

interface QuickAction {
    id: string;
    icon: keyof typeof Ionicons.glyphMap;
    label: string;
    route: string;
    color: string;
    bgColor: string;
}

const ACTIONS: QuickAction[] = [
    {
        id: 'hub',
        icon: 'grid',
        label: 'AI Hub',
        route: 'AIHub',
        color: '#FFFFFF',
        bgColor: '#6366F1',
    },
    {
        id: 'scan',
        icon: 'videocam',
        label: 'Scan Wardrobe',
        route: 'WardrobeVideo',
        color: '#FFFFFF',
        bgColor: '#1A1A1A',
    },
    {
        id: 'outfit',
        icon: 'sparkles',
        label: 'AI Stylist',
        route: 'AIChat',
        color: '#FFFFFF',
        bgColor: '#8B5CF6',
    },
    {
        id: 'add',
        icon: 'add-circle',
        label: 'Add Item',
        route: 'AddOutfit',
        color: '#FFFFFF',
        bgColor: '#22C55E',
    },
];

const AnimatedTouchable = Animated.createAnimatedComponent(TouchableOpacity);

export const QuickActionsFAB: React.FC = () => {
    const navigation = useNavigation();
    const [isOpen, setIsOpen] = useState(false);

    const progress = useSharedValue(0);
    const rotation = useSharedValue(0);
    const scale = useSharedValue(1);

    const toggleMenu = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

        if (isOpen) {
            progress.value = withSpring(0, animations.springFast);
            rotation.value = withSpring(0, animations.springFast);
        } else {
            progress.value = withSpring(1, animations.springBouncy);
            rotation.value = withSpring(45, animations.springBouncy);
        }

        setIsOpen(!isOpen);
    };

    const handleAction = (route: string) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

        // Close menu first
        progress.value = withTiming(0, { duration: 200 });
        rotation.value = withTiming(0, { duration: 200 });
        setIsOpen(false);

        // Navigate after a short delay
        setTimeout(() => {
            (navigation as any).navigate(route);
        }, 150);
    };

    // Main FAB animation
    const fabStyle = useAnimatedStyle(() => ({
        transform: [
            { scale: withSpring(scale.value, animations.springFast) },
            { rotate: `${rotation.value}deg` },
        ],
    }));

    // Overlay animation
    const overlayStyle = useAnimatedStyle(() => ({
        opacity: interpolate(progress.value, [0, 1], [0, 1]),
        pointerEvents: progress.value > 0.5 ? 'auto' : 'none',
    }));

    // Action button animations
    const getActionStyle = (index: number) => {
        return useAnimatedStyle(() => {
            const angle = -90 + (index * 30); // Spread actions in an arc
            const radius = 90;
            const baseDelay = index * 50;

            const x = Math.cos((angle * Math.PI) / 180) * radius * progress.value;
            const y = Math.sin((angle * Math.PI) / 180) * radius * progress.value;

            return {
                transform: [
                    { translateX: x },
                    { translateY: y },
                    {
                        scale: interpolate(
                            progress.value,
                            [0, 0.5, 1],
                            [0, 0.5, 1],
                            Extrapolate.CLAMP
                        )
                    },
                ],
                opacity: progress.value,
            };
        });
    };

    // Label animations
    const getLabelStyle = (index: number) => {
        return useAnimatedStyle(() => ({
            opacity: interpolate(progress.value, [0.7, 1], [0, 1]),
            transform: [
                {
                    translateX: interpolate(progress.value, [0.7, 1], [20, 0])
                },
            ],
        }));
    };

    return (
        <>
            {/* Overlay */}
            <Animated.View style={[styles.overlay, overlayStyle]}>
                <TouchableOpacity
                    style={StyleSheet.absoluteFill}
                    onPress={toggleMenu}
                    activeOpacity={1}
                >
                    <BlurView intensity={20} tint="dark" style={StyleSheet.absoluteFill} />
                </TouchableOpacity>
            </Animated.View>

            {/* Action Buttons */}
            <View style={styles.container}>
                {ACTIONS.map((action, index) => (
                    <Animated.View
                        key={action.id}
                        style={[styles.actionContainer, getActionStyle(index)]}
                    >
                        <TouchableOpacity
                            style={[styles.actionButton, { backgroundColor: action.bgColor }]}
                            onPress={() => handleAction(action.route)}
                            activeOpacity={0.9}
                        >
                            <Ionicons name={action.icon} size={24} color={action.color} />
                        </TouchableOpacity>
                        <Animated.View style={[styles.labelContainer, getLabelStyle(index)]}>
                            <Text style={styles.label}>{action.label}</Text>
                        </Animated.View>
                    </Animated.View>
                ))}

                {/* Main FAB */}
                <AnimatedTouchable
                    style={[styles.fab, shadows.strong, fabStyle]}
                    onPress={toggleMenu}
                    onPressIn={() => { scale.value = 0.92; }}
                    onPressOut={() => { scale.value = 1; }}
                    activeOpacity={1}
                >
                    <Ionicons
                        name="add"
                        size={32}
                        color="#FFFFFF"
                    />
                </AnimatedTouchable>
            </View>
        </>
    );
};

const styles = StyleSheet.create({
    overlay: {
        ...StyleSheet.absoluteFillObject,
        zIndex: 998,
    },
    container: {
        position: 'absolute',
        bottom: 100,
        right: 24,
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 999,
    },
    fab: {
        width: 60,
        height: 60,
        borderRadius: 30,
        backgroundColor: colors.text?.primary || '#1A1A1A',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
    },
    actionContainer: {
        position: 'absolute',
        flexDirection: 'row',
        alignItems: 'center',
    },
    actionButton: {
        width: 50,
        height: 50,
        borderRadius: 25,
        alignItems: 'center',
        justifyContent: 'center',
        ...shadows.medium,
    },
    labelContainer: {
        position: 'absolute',
        right: 60,
        backgroundColor: 'rgba(0, 0, 0, 0.8)',
        paddingHorizontal: 12,
        paddingVertical: 6,
        borderRadius: 8,
        marginRight: 8,
    },
    label: {
        color: '#FFFFFF',
        fontSize: 13,
        fontWeight: '600',
    },
});

export default QuickActionsFAB;
