import React, { useState, useRef, useCallback, useEffect } from "react";
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Dimensions,
    Platform,
    ScrollView,
    TextInput,
    KeyboardAvoidingView,
    Keyboard,
    TouchableWithoutFeedback,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import * as Haptics from "expo-haptics";
import { BlurView } from "expo-blur";
import { WebView } from "react-native-webview";

import AppColors from "../constants/AppColors";
import LiquidGlass2026Theme from "../constants/LiquidGlass2026Theme";

const { width, height: SCREEN_HEIGHT } = Dimensions.get("window");

import { generate3Dhtml, BODY_TYPES, BodyTypeId } from "../features/try-on/utils/mannequin3D";

export default function CreateAvatarScreen() {
    const navigation = useNavigation();
    const webViewRef = useRef<WebView>(null);

    // State for real measurements
    const [heightCm, setHeightCm] = useState("175");
    const [weightKg, setWeightKg] = useState("70");
    const [bodyType, setBodyType] = useState<BodyTypeId>("average");
    const [gender, setGender] = useState<"male" | "female">("male");

    const handleContinue = () => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        (navigation as any).navigate('AIOutfit');
    };

    // Send measurements to 3D WebView
    const sendUpdate = useCallback(() => {
        const h = parseInt(heightCm) || 175;
        const w = parseInt(weightKg) || 70;
        webViewRef.current?.postMessage(JSON.stringify({
            type: 'update',
            heightCm: h,
            weightKg: w,
            bodyType: bodyType,
        }));
    }, [heightCm, weightKg, bodyType]);

    // Trigger update on value change
    useEffect(() => {
        const timeout = setTimeout(() => {
            sendUpdate();
        }, 150); // Small debounce
        return () => clearTimeout(timeout);
    }, [heightCm, weightKg, bodyType, sendUpdate]);

    const handleHeightChange = (text: string) => {
        const clean = text.replace(/[^0-9]/g, '');
        if (clean.length <= 3) setHeightCm(clean);
    };

    const handleWeightChange = (text: string) => {
        const clean = text.replace(/[^0-9]/g, '');
        if (clean.length <= 3) setWeightKg(clean);
    };

    return (
        <KeyboardAvoidingView
            style={styles.container}
            behavior={Platform.OS === "ios" ? "padding" : "height"}
        >
            <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
                <View style={styles.container}>
                    <SafeAreaView style={styles.safeArea}>
                        {/* Header */}
                        <View style={styles.header}>
                            <TouchableOpacity
                                style={styles.backButton}
                                onPress={() => navigation.goBack()}
                                hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                            >
                                <Ionicons name="arrow-back" size={22} color={LiquidGlass2026Theme.colors.text.primary} />
                            </TouchableOpacity>
                            <View style={styles.headerCenter}>
                                <Text style={styles.headerTitle}>3D Body Model</Text>
                                <Text style={styles.headerSubtitle}>Enter your measurements</Text>
                            </View>
                            <View style={{ width: 44 }} />
                        </View>

                        {/* 3D Mannequin via WebView */}
                        <View style={styles.mannequinViewer}>
                            <WebView
                                ref={webViewRef}
                                source={{ html: generate3Dhtml() }}
                                style={styles.webview}
                                scrollEnabled={false}
                                bounces={false}
                                javaScriptEnabled={true}
                                allowsInlineMediaPlayback={true}
                                originWhitelist={['*']}
                                onMessage={() => { }}
                            />
                            {/* Rotation hint */}
                            <View style={styles.rotateHint}>
                                <BlurView intensity={30} tint="light" style={styles.rotateHintInner}>
                                    <Ionicons name="finger-print-outline" size={14} color="rgba(0,0,0,0.5)" />
                                    <Text style={styles.rotateHintText}>Drag to rotate</Text>
                                </BlurView>
                            </View>
                        </View>
                    </SafeAreaView>

                    {/* Controls Bottom Sheet */}
                    <View style={styles.bottomSheetContainer}>
                        <BlurView
                            intensity={80}
                            tint="light"
                            style={styles.glassSheet}
                        >
                            <View style={styles.sheetHandle} />

                            <ScrollView
                                style={styles.controlsScroll}
                                showsVerticalScrollIndicator={false}
                                contentContainerStyle={styles.controlsContent}
                                keyboardShouldPersistTaps="handled"
                            >
                                {/* Measurement Inputs */}
                                <Text style={styles.sectionTitle}>Measurements</Text>
                                <View style={styles.measurementRow}>
                                    {/* Height */}
                                    <View style={styles.measurementCard}>
                                        <View style={styles.measurementIconRow}>
                                            <View style={styles.measurementIcon}>
                                                <Ionicons name="resize-outline" size={18} color="#fff" />
                                            </View>
                                            <Text style={styles.measurementLabel}>Height</Text>
                                        </View>
                                        <View style={styles.inputRow}>
                                            <TextInput
                                                style={styles.measurementInput}
                                                value={heightCm}
                                                onChangeText={handleHeightChange}
                                                keyboardType="number-pad"
                                                maxLength={3}
                                                placeholder="175"
                                                placeholderTextColor="rgba(0,0,0,0.2)"
                                                returnKeyType="done"
                                                onSubmitEditing={Keyboard.dismiss}
                                            />
                                            <Text style={styles.unitText}>cm</Text>
                                        </View>
                                        <Text style={styles.rangeText}>100–230 cm</Text>
                                    </View>

                                    {/* Weight */}
                                    <View style={styles.measurementCard}>
                                        <View style={styles.measurementIconRow}>
                                            <View style={[styles.measurementIcon, { backgroundColor: '#4A5568' }]}>
                                                <Ionicons name="scale-outline" size={18} color="#fff" />
                                            </View>
                                            <Text style={styles.measurementLabel}>Weight</Text>
                                        </View>
                                        <View style={styles.inputRow}>
                                            <TextInput
                                                style={styles.measurementInput}
                                                value={weightKg}
                                                onChangeText={handleWeightChange}
                                                keyboardType="number-pad"
                                                maxLength={3}
                                                placeholder="70"
                                                placeholderTextColor="rgba(0,0,0,0.2)"
                                                returnKeyType="done"
                                                onSubmitEditing={Keyboard.dismiss}
                                            />
                                            <Text style={styles.unitText}>kg</Text>
                                        </View>
                                        <Text style={styles.rangeText}>30–200 kg</Text>
                                    </View>
                                </View>

                                {/* Body Type Selection */}
                                <Text style={[styles.sectionTitle, { marginTop: 20 }]}>Body Type</Text>
                                <View style={styles.bodyTypeGrid}>
                                    {BODY_TYPES.map((bt) => {
                                        const isActive = bodyType === bt.id;
                                        return (
                                            <TouchableOpacity
                                                key={bt.id}
                                                style={[
                                                    styles.bodyTypeChip,
                                                    isActive && styles.bodyTypeChipActive,
                                                ]}
                                                onPress={() => {
                                                    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                                    setBodyType(bt.id);
                                                }}
                                                activeOpacity={0.7}
                                            >
                                                <Ionicons
                                                    name={bt.icon as any}
                                                    size={20}
                                                    color={isActive ? "#fff" : LiquidGlass2026Theme.colors.text.secondary}
                                                />
                                                <Text style={[
                                                    styles.bodyTypeLabel,
                                                    isActive && styles.bodyTypeLabelActive,
                                                ]}>
                                                    {bt.label}
                                                </Text>
                                                <Text style={[
                                                    styles.bodyTypeDesc,
                                                    isActive && styles.bodyTypeDescActive,
                                                ]}>
                                                    {bt.desc}
                                                </Text>
                                            </TouchableOpacity>
                                        );
                                    })}
                                </View>

                                {/* BMI Info Badge */}
                                {heightCm && weightKg && parseInt(heightCm) > 0 && parseInt(weightKg) > 0 && (
                                    <View style={styles.bmiCard}>
                                        <View style={styles.bmiRow}>
                                            <Text style={styles.bmiLabel}>BMI</Text>
                                            <Text style={styles.bmiValue}>
                                                {(parseInt(weightKg) / Math.pow(parseInt(heightCm) / 100, 2)).toFixed(1)}
                                            </Text>
                                        </View>
                                        <Text style={styles.bmiDesc}>
                                            {(() => {
                                                const bmi = parseInt(weightKg) / Math.pow(parseInt(heightCm) / 100, 2);
                                                if (bmi < 18.5) return "Underweight";
                                                if (bmi < 25) return "Normal weight";
                                                if (bmi < 30) return "Overweight";
                                                return "Obese";
                                            })()}
                                        </Text>
                                    </View>
                                )}
                            </ScrollView>
                        </BlurView>
                    </View>

                    {/* Floating Save Button */}
                    <View style={styles.floatingActionContainer}>
                        <BlurView intensity={20} tint="light" style={styles.fabGlass}>
                            <TouchableOpacity
                                style={styles.continueButton}
                                onPress={handleContinue}
                                activeOpacity={0.85}
                            >
                                <Ionicons name="checkmark-circle-outline" size={22} color="#fff" style={{ marginRight: 8 }} />
                                <Text style={styles.continueButtonText}>Save Body Model</Text>
                            </TouchableOpacity>
                        </BlurView>
                    </View>
                </View>
            </TouchableWithoutFeedback>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: LiquidGlass2026Theme.colors.background.secondary,
    },
    safeArea: {
        flex: 1,
        paddingBottom: SCREEN_HEIGHT * 0.48,
    },

    // ── Header ────────────────────────────────────
    header: {
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
        paddingHorizontal: 20,
        paddingTop: 8,
        paddingBottom: 4,
        zIndex: 10,
    },
    backButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: "rgba(255,255,255,0.9)",
        alignItems: "center",
        justifyContent: "center",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.06,
        shadowRadius: 8,
        elevation: 2,
    },
    headerCenter: {
        alignItems: "center",
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: "700",
        color: LiquidGlass2026Theme.colors.text.primary,
        letterSpacing: -0.3,
    },
    headerSubtitle: {
        fontSize: 12,
        fontWeight: "500",
        color: LiquidGlass2026Theme.colors.text.tertiary,
        marginTop: 2,
    },

    // ── 3D Viewer ─────────────────────────────────
    mannequinViewer: {
        flex: 1,
        zIndex: 1,
        overflow: "hidden",
    },
    webview: {
        flex: 1,
        backgroundColor: "transparent",
    },
    rotateHint: {
        position: "absolute",
        bottom: 12,
        alignSelf: "center",
        borderRadius: 20,
        overflow: "hidden",
    },
    rotateHintInner: {
        flexDirection: "row",
        alignItems: "center",
        paddingHorizontal: 14,
        paddingVertical: 6,
        gap: 6,
        backgroundColor: "rgba(255,255,255,0.5)",
    },
    rotateHintText: {
        fontSize: 12,
        fontWeight: "500",
        color: "rgba(0,0,0,0.45)",
    },

    // ── Bottom Sheet ──────────────────────────────
    bottomSheetContainer: {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        height: SCREEN_HEIGHT * 0.48,
        borderTopLeftRadius: 32,
        borderTopRightRadius: 32,
        overflow: "hidden",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: -6 },
        shadowOpacity: 0.1,
        shadowRadius: 20,
        elevation: 12,
    },
    glassSheet: {
        flex: 1,
        borderTopLeftRadius: 32,
        borderTopRightRadius: 32,
        paddingHorizontal: 20,
        paddingTop: 14,
        backgroundColor: "rgba(255, 255, 255, 0.78)",
    },
    sheetHandle: {
        width: 44,
        height: 5,
        borderRadius: 3,
        backgroundColor: "rgba(0,0,0,0.12)",
        alignSelf: "center",
        marginBottom: 16,
    },
    controlsScroll: {
        flex: 1,
    },
    controlsContent: {
        paddingBottom: 110,
    },

    // ── Section titles ────────────────────────────
    sectionTitle: {
        fontSize: 16,
        fontWeight: "700",
        color: LiquidGlass2026Theme.colors.text.primary,
        marginBottom: 12,
        letterSpacing: -0.2,
    },

    // ── Measurement inputs ────────────────────────
    measurementRow: {
        flexDirection: "row",
        gap: 12,
    },
    measurementCard: {
        flex: 1,
        backgroundColor: "rgba(255,255,255,0.75)",
        borderRadius: 20,
        padding: 16,
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.8)",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.03,
        shadowRadius: 8,
        elevation: 1,
    },
    measurementIconRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
        marginBottom: 10,
    },
    measurementIcon: {
        width: 32,
        height: 32,
        borderRadius: 10,
        backgroundColor: "#0A1931",
        alignItems: "center",
        justifyContent: "center",
    },
    measurementLabel: {
        fontSize: 14,
        fontWeight: "600",
        color: LiquidGlass2026Theme.colors.text.secondary,
    },
    inputRow: {
        flexDirection: "row",
        alignItems: "baseline",
        gap: 4,
    },
    measurementInput: {
        fontSize: 32,
        fontWeight: "800",
        color: LiquidGlass2026Theme.colors.text.primary,
        letterSpacing: -0.5,
        minWidth: 60,
        paddingVertical: 0,
    },
    unitText: {
        fontSize: 16,
        fontWeight: "600",
        color: LiquidGlass2026Theme.colors.text.tertiary,
    },
    rangeText: {
        fontSize: 11,
        fontWeight: "500",
        color: LiquidGlass2026Theme.colors.text.tertiary,
        marginTop: 6,
    },

    // ── Body type grid ────────────────────────────
    bodyTypeGrid: {
        flexDirection: "row",
        flexWrap: "wrap",
        gap: 8,
    },
    bodyTypeChip: {
        width: (width - 40 - 16) / 3,
        backgroundColor: "rgba(255,255,255,0.7)",
        borderRadius: 16,
        padding: 12,
        alignItems: "center",
        borderWidth: 1.5,
        borderColor: "rgba(0,0,0,0.06)",
    },
    bodyTypeChipActive: {
        backgroundColor: "#0A1931",
        borderColor: "#0A1931",
        shadowColor: "#0A1931",
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.2,
        shadowRadius: 10,
        elevation: 4,
    },
    bodyTypeLabel: {
        fontSize: 12,
        fontWeight: "700",
        color: LiquidGlass2026Theme.colors.text.primary,
        marginTop: 6,
    },
    bodyTypeLabelActive: {
        color: "#fff",
    },
    bodyTypeDesc: {
        fontSize: 9,
        fontWeight: "500",
        color: LiquidGlass2026Theme.colors.text.tertiary,
        marginTop: 3,
        textAlign: "center",
    },
    bodyTypeDescActive: {
        color: "rgba(255,255,255,0.7)",
    },

    // ── BMI card ──────────────────────────────────
    bmiCard: {
        marginTop: 16,
        backgroundColor: "rgba(10,25,49,0.05)",
        borderRadius: 16,
        padding: 14,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "space-between",
    },
    bmiRow: {
        flexDirection: "row",
        alignItems: "baseline",
        gap: 8,
    },
    bmiLabel: {
        fontSize: 13,
        fontWeight: "600",
        color: LiquidGlass2026Theme.colors.text.secondary,
    },
    bmiValue: {
        fontSize: 22,
        fontWeight: "800",
        color: LiquidGlass2026Theme.colors.text.primary,
    },
    bmiDesc: {
        fontSize: 13,
        fontWeight: "600",
        color: LiquidGlass2026Theme.colors.text.tertiary,
    },

    // ── Floating action ───────────────────────────
    floatingActionContainer: {
        position: "absolute",
        bottom: Platform.OS === "ios" ? 34 : 24,
        left: 20,
        right: 20,
        borderRadius: 30,
        overflow: "hidden",
        shadowColor: "#0A1931",
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.2,
        shadowRadius: 24,
        elevation: 10,
    },
    fabGlass: {
        padding: 5,
        backgroundColor: "rgba(255,255,255,0.3)",
    },
    continueButton: {
        backgroundColor: "#0A1931",
        height: 56,
        borderRadius: 28,
        flexDirection: "row",
        alignItems: "center",
        justifyContent: "center",
    },
    continueButtonText: {
        color: "#FFFFFF",
        fontSize: 16,
        fontWeight: "700",
        letterSpacing: 0.2,
    },
});
