import React from "react";
import {
    Modal,
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Dimensions,
    Platform,
} from "react-native";
import { BlurView } from "expo-blur";
import { LinearGradient } from "expo-linear-gradient";
import { Ionicons } from "@expo/vector-icons";
import * as Haptics from "expo-haptics";
import Animated, {
    useAnimatedStyle,
    useSharedValue,
    withSpring,
} from "react-native-reanimated";

const { width } = Dimensions.get("window");

interface TrialLimitModalProps {
    visible: boolean;
    onSignUp: () => void;
    onSignIn: () => void;
    onSubscribe?: () => void;
}

// Tahoe press animation hook
const useTahoePress = () => {
    const scale = useSharedValue(1);

    const animatedStyle = useAnimatedStyle(() => ({
        transform: [{ scale: withSpring(scale.value, { damping: 15, stiffness: 400 }) }],
    }));

    return {
        animatedStyle,
        onPressIn: () => { scale.value = 0.97; },
        onPressOut: () => { scale.value = 1; },
    };
};

const TrialLimitModal: React.FC<TrialLimitModalProps> = ({
    visible,
    onSignUp,
    onSignIn,
    onSubscribe,
}) => {
    const premiumPress = useTahoePress();
    const vipPress = useTahoePress();

    return (
        <Modal
            visible={visible}
            transparent
            animationType="fade"
            statusBarTranslucent
        >
            <BlurView intensity={20} tint="dark" style={styles.backdrop}>
                <View style={styles.container}>
                    <View style={styles.card}>
                        <LinearGradient
                            colors={["#1A1C29", "#1A1C29"]}
                            style={styles.gradient}
                        >
                            {/* Icon */}
                            <View style={styles.iconContainer}>
                                <Ionicons name="sparkles" size={42} color="#FFD700" />
                            </View>

                            {/* Title */}
                            <Text style={styles.title}>5 Free Uses Complete!</Text>

                            {/* Message */}
                            <Text style={styles.message}>
                                Upgrade to unlock unlimited AI styling and take your wardrobe to the next level.
                            </Text>

                            {/* Subscription Options */}
                            <View style={styles.subscriptionOptions}>
                                {/* Premium Option */}
                                <TouchableOpacity
                                    onPressIn={premiumPress.onPressIn}
                                    onPressOut={premiumPress.onPressOut}
                                    onPress={() => {
                                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                        onSubscribe?.();
                                    }}
                                    activeOpacity={1}
                                    style={styles.flex1}
                                >
                                    <Animated.View style={[styles.subscriptionCard, styles.premiumCard, premiumPress.animatedStyle]}>
                                        <View style={styles.popularBadge}>
                                            <Text style={styles.popularBadgeText}>POPULAR</Text>
                                        </View>
                                    </Animated.View>
                                </TouchableOpacity>

                                {/* VIP Option */}
                                <TouchableOpacity
                                    onPressIn={vipPress.onPressIn}
                                    onPressOut={vipPress.onPressOut}
                                    onPress={() => {
                                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
                                        onSubscribe?.();
                                    }}
                                    activeOpacity={1}
                                    style={styles.flex1}
                                >
                                    <Animated.View style={[styles.subscriptionCard, styles.vipCard, vipPress.animatedStyle]}>
                                    </Animated.View>
                                </TouchableOpacity>
                            </View>

                            {/* Or continue with account */}
                            <View style={styles.dividerRow}>
                                <View style={styles.divider} />
                                <Text style={styles.dividerText}>or</Text>
                                <View style={styles.divider} />
                            </View>

                            {/* Auth Buttons */}
                            <View style={styles.buttonContainer}>
                                <TouchableOpacity
                                    style={styles.primaryButton}
                                    onPress={() => {
                                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                        onSignUp();
                                    }}
                                    activeOpacity={0.8}
                                >
                                    <Text style={styles.primaryButtonText}>Create Free Account</Text>
                                </TouchableOpacity>

                                <TouchableOpacity
                                    style={styles.secondaryButton}
                                    onPress={() => {
                                        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                        onSignIn();
                                    }}
                                    activeOpacity={0.8}
                                >
                                    <Text style={styles.secondaryButtonText}>
                                        Already have an account? Sign In
                                    </Text>
                                </TouchableOpacity>
                            </View>
                        </LinearGradient>
                    </View>
                </View>
            </BlurView>
        </Modal>
    );
};

const styles = StyleSheet.create({
    backdrop: {
        flex: 1,
        backgroundColor: "rgba(0, 0, 0, 0.4)",
        justifyContent: "center",
        alignItems: "center",
    },
    container: {
        width: width - 40,
        maxWidth: 400,
    },
    card: {
        borderRadius: 28,
        overflow: "hidden",
        backgroundColor: "#1A1C29",
        ...Platform.select({
            ios: {
                shadowColor: "#000",
                shadowOffset: { width: 0, height: 10 },
                shadowOpacity: 0.5,
                shadowRadius: 20,
            },
            android: {
                elevation: 10,
            },
        }),
    },
    gradient: {
        paddingVertical: 36,
        paddingHorizontal: 20,
        alignItems: "center",
    },
    iconContainer: {
        width: 76,
        height: 76,
        borderRadius: 38,
        backgroundColor: "rgba(92, 80, 24, 0.4)",
        justifyContent: "center",
        alignItems: "center",
        marginBottom: 20,
    },
    title: {
        fontSize: 22,
        fontWeight: "800",
        color: "#FFF",
        marginBottom: 12,
        textAlign: "center",
    },
    message: {
        fontSize: 15,
        color: "rgba(255, 255, 255, 0.8)",
        textAlign: "center",
        lineHeight: 22,
        marginBottom: 30,
        paddingHorizontal: 12,
    },

    // Subscription Options
    subscriptionOptions: {
        flexDirection: 'row',
        gap: 16,
        marginBottom: 28,
        width: '100%',
        paddingHorizontal: 8,
    },
    flex1: {
        flex: 1,
    },
    subscriptionCard: {
        height: 48,
        borderRadius: 24,
        borderWidth: 2,
        justifyContent: 'center',
        alignItems: 'center',
    },
    premiumCard: {
        backgroundColor: 'transparent',
        borderColor: '#FFD700',
    },
    vipCard: {
        backgroundColor: 'transparent',
        borderColor: '#A855F7',
    },
    popularBadge: {
        position: 'absolute',
        top: -10,
        right: 14,
        backgroundColor: '#FFD700',
        paddingHorizontal: 8,
        paddingVertical: 3,
        borderRadius: 6,
    },
    popularBadgeText: {
        fontSize: 9,
        fontWeight: '800',
        color: '#000',
    },

    // Divider
    dividerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        width: '100%',
        marginBottom: 24,
        paddingHorizontal: 20,
    },
    divider: {
        flex: 1,
        height: 1,
        backgroundColor: 'rgba(255,255,255,0.15)',
    },
    dividerText: {
        fontSize: 13,
        color: 'rgba(255,255,255,0.5)',
        marginHorizontal: 16,
    },

    buttonContainer: {
        width: "100%",
        paddingHorizontal: 8,
    },
    primaryButton: {
        backgroundColor: "#FFF",
        paddingVertical: 16,
        borderRadius: 12,
        alignItems: "center",
        marginBottom: 16,
    },
    primaryButtonText: {
        fontSize: 16,
        fontWeight: "700",
        color: "#000",
    },
    secondaryButton: {
        paddingVertical: 8,
        alignItems: "center",
    },
    secondaryButtonText: {
        fontSize: 14,
        color: "rgba(255,255,255,0.7)",
        fontWeight: "400",
    },
});

export default TrialLimitModal;
