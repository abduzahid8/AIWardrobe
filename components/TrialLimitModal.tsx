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

const { width } = Dimensions.get("window");

interface TrialLimitModalProps {
    visible: boolean;
    onSignUp: () => void;
    onSignIn: () => void;
}

const TrialLimitModal: React.FC<TrialLimitModalProps> = ({
    visible,
    onSignUp,
    onSignIn,
}) => {
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
                            colors={["#667eea", "#764ba2"]}
                            start={{ x: 0, y: 0 }}
                            end={{ x: 1, y: 1 }}
                            style={styles.gradient}
                        >
                            {/* Icon */}
                            <View style={styles.iconContainer}>
                                <Ionicons name="sparkles" size={48} color="#FFF" />
                            </View>

                            {/* Title */}
                            <Text style={styles.title}>Trial Ended</Text>

                            {/* Message */}
                            <Text style={styles.message}>
                                You've used all 3 free tries! Create an account to continue
                                enjoying AI Wardrobe with unlimited access.
                            </Text>

                            {/* Features List */}
                            <View style={styles.featuresList}>
                                <View style={styles.featureItem}>
                                    <Ionicons name="checkmark-circle" size={20} color="#FFF" />
                                    <Text style={styles.featureText}>Unlimited AI Outfits</Text>
                                </View>
                                <View style={styles.featureItem}>
                                    <Ionicons name="checkmark-circle" size={20} color="#FFF" />
                                    <Text style={styles.featureText}>Unlimited Wardrobe Scans</Text>
                                </View>
                                <View style={styles.featureItem}>
                                    <Ionicons name="checkmark-circle" size={20} color="#FFF" />
                                    <Text style={styles.featureText}>Save Your Preferences</Text>
                                </View>
                            </View>

                            {/* Buttons */}
                            <View style={styles.buttonContainer}>
                                <TouchableOpacity
                                    style={styles.primaryButton}
                                    onPress={onSignUp}
                                    activeOpacity={0.8}
                                >
                                    <Text style={styles.primaryButtonText}>Create Account</Text>
                                </TouchableOpacity>

                                <TouchableOpacity
                                    style={styles.secondaryButton}
                                    onPress={onSignIn}
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
        backgroundColor: "rgba(0, 0, 0, 0.5)",
        justifyContent: "center",
        alignItems: "center",
    },
    container: {
        width: width - 48,
        maxWidth: 400,
    },
    card: {
        borderRadius: 24,
        overflow: "hidden",
        ...Platform.select({
            ios: {
                shadowColor: "#000",
                shadowOffset: { width: 0, height: 10 },
                shadowOpacity: 0.3,
                shadowRadius: 20,
            },
            android: {
                elevation: 10,
            },
        }),
    },
    gradient: {
        padding: 32,
        alignItems: "center",
    },
    iconContainer: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: "rgba(255, 255, 255, 0.2)",
        justifyContent: "center",
        alignItems: "center",
        marginBottom: 24,
    },
    title: {
        fontSize: 28,
        fontWeight: "700",
        color: "#FFF",
        marginBottom: 12,
        textAlign: "center",
    },
    message: {
        fontSize: 16,
        color: "rgba(255, 255, 255, 0.9)",
        textAlign: "center",
        lineHeight: 24,
        marginBottom: 24,
    },
    featuresList: {
        width: "100%",
        marginBottom: 32,
    },
    featureItem: {
        flexDirection: "row",
        alignItems: "center",
        marginBottom: 12,
    },
    featureText: {
        fontSize: 15,
        color: "#FFF",
        marginLeft: 12,
        fontWeight: "500",
    },
    buttonContainer: {
        width: "100%",
    },
    primaryButton: {
        backgroundColor: "#FFF",
        paddingVertical: 16,
        borderRadius: 12,
        alignItems: "center",
        marginBottom: 12,
        ...Platform.select({
            ios: {
                shadowColor: "#000",
                shadowOffset: { width: 0, height: 4 },
                shadowOpacity: 0.2,
                shadowRadius: 8,
            },
            android: {
                elevation: 4,
            },
        }),
    },
    primaryButtonText: {
        fontSize: 17,
        fontWeight: "700",
        color: "#667eea",
    },
    secondaryButton: {
        paddingVertical: 12,
        alignItems: "center",
    },
    secondaryButtonText: {
        fontSize: 14,
        color: "#FFF",
        fontWeight: "600",
    },
});

export default TrialLimitModal;
