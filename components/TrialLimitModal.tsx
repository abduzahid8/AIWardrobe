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
    FadeInUp,
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
            <BlurView intensity={40} style={styles.backdrop}>
                <View style={styles.container}>
                    <View style={styles.card}>
                        <LinearGradient
                            colors={["#1A1A2E", "#16213E"]}
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 1 }}
                            style={styles.gradient}
                        >
                            {/* Icon */}
                            <View style={styles.iconContainer}>
                                <Ionicons name="sparkles" size={48} color="#FFD700" />
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
                                >
                                    <Animated.View style={[styles.subscriptionCard, styles.premiumCard, premiumPress.animatedStyle]}>
                                        <View style={styles.popularBadge}>
                                            <Text style={styles.popularBadgeText}>POPULAR</Text>
                                        </View>
                                        <Text style={styles.tierName}>Premium</Text>
                                        <View style={styles.priceRow}>
                                            <Text style={styles.price}>$9.99</Text>
                                            <Text style={styles.period}>/month</Text>
                                        </View>
                                        <View style={styles.featureRow}>
                                            <Ionicons name="checkmark" size={14} color="#34C759" />
                                            <Text style={styles.featureText}>Unlimited AI Outfits</Text>
                                        </View>
                                        <View style={styles.featureRow}>
                                            <Ionicons name="checkmark" size={14} color="#34C759" />
                                            <Text style={styles.featureText}>50 Try-Ons/month</Text>
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
                                >
                                    <Animated.View style={[styles.subscriptionCard, styles.vipCard, vipPress.animatedStyle]}>
                                        <Text style={styles.tierName}>VIP</Text>
                                        <View style={styles.priceRow}>
                                            <Text style={styles.price}>$99.99</Text>
                                            <Text style={styles.period}>/year</Text>
                                        </View>
                                        <View style={styles.featureRow}>
                                            <Ionicons name="checkmark" size={14} color="#34C759" />
                                            <Text style={styles.featureText}>Everything + Unlimited</Text>
                                        </View>
                                        <View style={styles.featureRow}>
                                            <Ionicons name="checkmark" size={14} color="#34C759" />
                                            <Text style={styles.featureText}>Priority Support</Text>
                                        </View>
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
        backgroundColor: "rgba(0, 0, 0, 0.6)",
        justifyContent: "center",
        alignItems: "center",
    },
    container: {
        width: width - 32,
        maxWidth: 400,
    },
    card: {
        borderRadius: 24,
        overflow: "hidden",
        ...Platform.select({
            ios: {
                shadowColor: "#000",
                shadowOffset: { width: 0, height: 10 },
                shadowOpacity: 0.4,
                shadowRadius: 20,
            },
            android: {
                elevation: 10,
            },
        }),
    },
    gradient: {
        padding: 24,
        alignItems: "center",
    },
    iconContainer: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: "rgba(255, 215, 0, 0.15)",
        justifyContent: "center",
        alignItems: "center",
        marginBottom: 20,
    },
    title: {
        fontSize: 24,
        fontWeight: "700",
        color: "#FFF",
        marginBottom: 10,
        textAlign: "center",
    },
    message: {
        fontSize: 14,
        color: "rgba(255, 255, 255, 0.8)",
        textAlign: "center",
        lineHeight: 20,
        marginBottom: 20,
        paddingHorizontal: 10,
    },

    // Subscription Options
    subscriptionOptions: {
        flexDirection: 'row',
        gap: 12,
        marginBottom: 20,
        width: '100%',
    },
    subscriptionCard: {
        flex: 1,
        padding: 14,
        borderRadius: 16,
        borderWidth: 2,
    },
    premiumCard: {
        backgroundColor: 'rgba(255, 215, 0, 0.1)',
        borderColor: '#FFD700',
    },
    vipCard: {
        backgroundColor: 'rgba(168, 85, 247, 0.1)',
        borderColor: '#A855F7',
    },
    popularBadge: {
        position: 'absolute',
        top: -10,
        right: 10,
        backgroundColor: '#FFD700',
        paddingHorizontal: 6,
        paddingVertical: 2,
        borderRadius: 6,
    },
    popularBadgeText: {
        fontSize: 8,
        fontWeight: '700',
        color: '#000',
    },
    tierName: {
        fontSize: 16,
        fontWeight: '700',
        color: '#FFF',
        marginBottom: 4,
    },
    priceRow: {
        flexDirection: 'row',
        alignItems: 'flex-end',
        marginBottom: 10,
    },
    price: {
        fontSize: 22,
        fontWeight: '800',
        color: '#FFF',
    },
    period: {
        fontSize: 12,
        color: 'rgba(255,255,255,0.6)',
        marginLeft: 2,
        marginBottom: 2,
    },
    featureRow: {
        flexDirection: 'row',
        alignItems: 'center',
        gap: 4,
        marginBottom: 4,
    },
    featureText: {
        fontSize: 11,
        color: 'rgba(255,255,255,0.8)',
    },

    // Divider
    dividerRow: {
        flexDirection: 'row',
        alignItems: 'center',
        width: '100%',
        marginBottom: 16,
    },
    divider: {
        flex: 1,
        height: 1,
        backgroundColor: 'rgba(255,255,255,0.2)',
    },
    dividerText: {
        fontSize: 12,
        color: 'rgba(255,255,255,0.5)',
        marginHorizontal: 12,
    },

    buttonContainer: {
        width: "100%",
    },
    primaryButton: {
        backgroundColor: "#FFF",
        paddingVertical: 14,
        borderRadius: 12,
        alignItems: "center",
        marginBottom: 10,
    },
    primaryButtonText: {
        fontSize: 16,
        fontWeight: "700",
        color: "#1A1A2E",
    },
    secondaryButton: {
        paddingVertical: 10,
        alignItems: "center",
    },
    secondaryButtonText: {
        fontSize: 13,
        color: "rgba(255,255,255,0.7)",
        fontWeight: "500",
    },
});

export default TrialLimitModal;
