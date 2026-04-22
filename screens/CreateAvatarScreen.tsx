import React, { useState, useCallback } from "react";
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
    Image,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useAppNavigation } from '../hooks/useAppNavigation';
import * as Haptics from "expo-haptics";
import { BlurView } from "expo-blur";
import { LinearGradient } from "expo-linear-gradient";

import AppColors from "../constants/AppColors";
import LiquidGlass2026Theme from "../constants/LiquidGlass2026Theme";

const { width } = Dimensions.get("window");

import { BODY_TYPES, BodyTypeId } from "../features/try-on/utils/mannequin3D";
import useAvatarStore from "../store/avatarStore";

const MANNEQUIN_IMAGE = require('../assets/images/mannequin_front.png');

export default function CreateAvatarScreen() {
    const navigation = useAppNavigation();
    
    // Read from persistent store
    const storedAvatar = useAvatarStore();

    // Local State for real measurements
    const [heightCm, setHeightCm] = useState(storedAvatar.heightCm || "175");
    const [weightKg, setWeightKg] = useState(storedAvatar.weightKg || "70");
    const [bodyType, setBodyType] = useState<BodyTypeId>(storedAvatar.bodyType || "average");
    const [gender, setGender] = useState<"male" | "female">(storedAvatar.gender || "male");

    const handleContinue = () => {
        // Save back to store
        storedAvatar.setMeasurements(heightCm, weightKg, bodyType);
        storedAvatar.setGender(gender);
        
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        
        if (navigation.canGoBack()) {
            navigation.goBack();
        } else {
            navigation.navigate('AIOutfit');
        }
    };

    const handleHeightChange = (text: string) => {
        const clean = text.replace(/[^0-9]/g, '');
        if (clean.length <= 3) {
            setHeightCm(clean);
            storedAvatar.setMeasurements(clean, weightKg, bodyType);
        }
    };

    const handleWeightChange = (text: string) => {
        const clean = text.replace(/[^0-9]/g, '');
        if (clean.length <= 3) {
            setWeightKg(clean);
            storedAvatar.setMeasurements(heightCm, clean, bodyType);
        }
    };

    const handleBodyTypeChange = (bt: BodyTypeId) => {
        setBodyType(bt);
        storedAvatar.setMeasurements(heightCm, weightKg, bt);
    };

    return (
        <KeyboardAvoidingView
            style={styles.container}
            behavior={Platform.OS === "ios" ? "padding" : "height"}
        >
            <LinearGradient
                colors={['#F6FAFF', '#EEF4FF', '#FFFFFF']}
                style={StyleSheet.absoluteFill}
                pointerEvents="none"
            />
            <View pointerEvents="none" style={styles.backgroundOrbTop} />
            <View pointerEvents="none" style={styles.backgroundOrbBottom} />
            <SafeAreaView style={styles.safeArea}>
                {/* Header — fixed above scroll */}
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

                {/* Single scrollable area: mannequin + controls */}
                <ScrollView
                    style={styles.controlsScroll}
                    showsVerticalScrollIndicator={false}
                    contentContainerStyle={styles.controlsContent}
                    keyboardShouldPersistTaps="handled"
                    keyboardDismissMode="on-drag"
                >
                    {/* Static Mannequin Preview */}
                    <View style={styles.mannequinViewer}>
                        <Image
                            source={MANNEQUIN_IMAGE}
                            style={styles.mannequinImage}
                            resizeMode="contain"
                        />
                        <View style={styles.mannequinOverlay}>
                            <BlurView intensity={30} tint="light" style={styles.mannequinOverlayInner}>
                                <Ionicons name="body-outline" size={14} color="rgba(0,0,0,0.5)" />
                                <Text style={styles.mannequinOverlayText}>
                                    {heightCm}cm • {weightKg}kg • {BODY_TYPES.find(b => b.id === bodyType)?.label}
                                </Text>
                            </BlurView>
                        </View>
                    </View>

                    {/* Controls — directly below mannequin */}
                    <View style={styles.sheetHandle} />

                            <Text style={styles.sectionTitle}>Body Measurements</Text>
                                <View style={styles.slidersContainer}>
                                    {/* Height Input */}
                                    <View style={styles.sliderCard}>
                                        <View style={styles.sliderHeader}>
                                            <View style={styles.measurementIconRow}>
                                                <View style={styles.measurementIcon}>
                                                    <Ionicons name="resize-outline" size={18} color="#fff" />
                                                </View>
                                                <Text style={styles.measurementLabel}>Height</Text>
                                            </View>
                                            <View style={styles.inputWrapper}>
                                                <TextInput
                                                    style={styles.measurementInput}
                                                    value={heightCm}
                                                    onChangeText={handleHeightChange}
                                                    keyboardType="number-pad"
                                                    maxLength={3}
                                                    returnKeyType="done"
                                                    onSubmitEditing={Keyboard.dismiss}
                                                />
                                                <Text style={styles.inputUnit}>cm</Text>
                                            </View>
                                        </View>
                                        <Text style={styles.inputHint}>Range: 140 – 230 cm</Text>
                                    </View>

                                    {/* Weight Input */}
                                    <View style={styles.sliderCard}>
                                        <View style={styles.sliderHeader}>
                                            <View style={styles.measurementIconRow}>
                                                <View style={[styles.measurementIcon, { backgroundColor: '#4A5568' }]}>
                                                    <Ionicons name="scale-outline" size={18} color="#fff" />
                                                </View>
                                                <Text style={styles.measurementLabel}>Weight</Text>
                                            </View>
                                            <View style={styles.inputWrapper}>
                                                <TextInput
                                                    style={styles.measurementInput}
                                                    value={weightKg}
                                                    onChangeText={handleWeightChange}
                                                    keyboardType="number-pad"
                                                    maxLength={3}
                                                    returnKeyType="done"
                                                    onSubmitEditing={Keyboard.dismiss}
                                                />
                                                <Text style={styles.inputUnit}>kg</Text>
                                            </View>
                                        </View>
                                        <Text style={styles.inputHint}>Range: 40 – 150 kg</Text>
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
                                                    handleBodyTypeChange(bt.id);
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
            </SafeAreaView>
        </KeyboardAvoidingView>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: LiquidGlass2026Theme.colors.background.secondary,
    },
    backgroundOrbTop: {
        position: 'absolute',
        top: -100,
        right: -80,
        width: 280,
        height: 280,
        borderRadius: 140,
        backgroundColor: 'rgba(188, 210, 245, 0.42)',
    },
    backgroundOrbBottom: {
        position: 'absolute',
        left: -120,
        bottom: 140,
        width: 300,
        height: 300,
        borderRadius: 150,
        backgroundColor: 'rgba(216, 229, 252, 0.34)',
    },
    safeArea: {
        flex: 1,
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
        borderWidth: 1,
        borderColor: "rgba(24,58,103,0.08)",
        alignItems: "center",
        justifyContent: "center",
        shadowColor: "#173A65",
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.06,
        shadowRadius: 12,
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
        height: 520,
        zIndex: 1,
        overflow: "hidden",
        borderRadius: 30,
        marginHorizontal: 16,
        backgroundColor: 'rgba(255,255,255,0.88)',
        borderWidth: 1,
        borderColor: 'rgba(24,58,103,0.08)',
        shadowColor: '#173A65',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.08,
        shadowRadius: 20,
        elevation: 4,
    },
    mannequinImage: {
        width: '100%',
        height: '100%',
    },
    mannequinOverlay: {
        position: "absolute",
        bottom: 12,
        alignSelf: "center",
        borderRadius: 20,
        overflow: "hidden",
    },
    mannequinOverlayInner: {
        flexDirection: "row",
        alignItems: "center",
        paddingHorizontal: 14,
        paddingVertical: 6,
        gap: 6,
        backgroundColor: "rgba(255,255,255,0.7)",
    },
    mannequinOverlayText: {
        fontSize: 12,
        fontWeight: "500",
        color: "rgba(0,0,0,0.6)",
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
        paddingHorizontal: 20,
        paddingTop: 8,
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

    // ── Measurement inputs ─────────────────────────
    slidersContainer: {
        gap: 16,
    },
    sliderCard: {
        backgroundColor: "rgba(255,255,255,0.75)",
        borderRadius: 24,
        padding: 16,
        borderWidth: 1,
        borderColor: "rgba(24,58,103,0.08)",
        shadowColor: "#173A65",
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.05,
        shadowRadius: 14,
        elevation: 3,
    },
    sliderHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 4,
    },
    inputWrapper: {
        flexDirection: "row",
        alignItems: "center",
        backgroundColor: "rgba(10,25,49,0.06)",
        borderRadius: 12,
        paddingHorizontal: 12,
        paddingVertical: 4,
        borderWidth: 1.5,
        borderColor: "rgba(10,25,49,0.12)",
    },
    measurementInput: {
        fontSize: 22,
        fontWeight: "800",
        color: LiquidGlass2026Theme.colors.text.primary,
        letterSpacing: -0.5,
        minWidth: 50,
        textAlign: "right",
        padding: 0,
    },
    inputUnit: {
        fontSize: 14,
        fontWeight: "600",
        color: LiquidGlass2026Theme.colors.text.secondary,
        marginLeft: 4,
    },
    inputHint: {
        fontSize: 11,
        fontWeight: "500",
        color: LiquidGlass2026Theme.colors.text.tertiary,
        marginTop: 6,
    },
    measurementIconRow: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
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

    // ── Body type grid ────────────────────────────
    bodyTypeGrid: {
        flexDirection: "row",
        flexWrap: "wrap",
        gap: 8,
    },
    bodyTypeChip: {
        width: (width - 40 - 8) / 2,
        backgroundColor: "rgba(255,255,255,0.7)",
        borderRadius: 20,
        padding: 12,
        alignItems: "center",
        borderWidth: 1,
        borderColor: "rgba(24,58,103,0.08)",
    },
    bodyTypeChipActive: {
        backgroundColor: "#173A65",
        borderColor: "#173A65",
        shadowColor: "#173A65",
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
        backgroundColor: "#173A65",
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
