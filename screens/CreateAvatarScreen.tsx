import React, { useState, useRef, useCallback } from "react";
import {
    View,
    Text,
    StyleSheet,
    TouchableOpacity,
    Dimensions,
    Platform,
    ScrollView,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Ionicons } from "@expo/vector-icons";
import { useNavigation } from "@react-navigation/native";
import * as Haptics from "expo-haptics";
import Slider from "@react-native-community/slider";
import { BlurView } from "expo-blur";
import { WebView } from "react-native-webview";

import AppColors from "../constants/AppColors";
import LiquidGlass2026Theme from "../constants/LiquidGlass2026Theme";

const { width, height } = Dimensions.get("window");

// Generate the inline HTML that runs Three.js inside the WebView's Safari engine
function generate3Dhtml() {
    return `
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
  * { margin: 0; padding: 0; }
  body { overflow: hidden; background: transparent; }
  canvas { display: block; width: 100vw; height: 100vh; }
</style>
</head>
<body>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\/script>
<script>
  // Scene setup
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
  camera.position.set(0, 0.8, 5);
  
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setClearColor(0x000000, 0);
  document.body.appendChild(renderer.domElement);

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
  dirLight.position.set(5, 10, 7);
  scene.add(dirLight);
  const rimLight = new THREE.DirectionalLight(0x8899aa, 0.5);
  rimLight.position.set(-5, 3, -5);
  scene.add(rimLight);

  // Materials
  const skinMat = new THREE.MeshStandardMaterial({ color: 0x94A3B8, roughness: 0.35, metalness: 0.15 });
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x64748B, roughness: 0.4, metalness: 0.1 });
  const darkerMat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.4, metalness: 0.1 });

  // Mannequin group
  const mannequin = new THREE.Group();

  // Head
  const headGeo = new THREE.SphereGeometry(0.22, 32, 32);
  const head = new THREE.Mesh(headGeo, skinMat);
  head.position.set(0, 1.85, 0);
  mannequin.add(head);

  // Neck
  const neckGeo = new THREE.CylinderGeometry(0.08, 0.1, 0.15, 16);
  const neck = new THREE.Mesh(neckGeo, skinMat);
  neck.position.set(0, 1.58, 0);
  mannequin.add(neck);

  // Shoulders
  const shouldersGeo = new THREE.CylinderGeometry(0.32, 0.3, 0.2, 32);
  const shoulders = new THREE.Mesh(shouldersGeo, darkMat);
  shoulders.position.set(0, 1.4, 0);
  mannequin.add(shoulders);

  // Upper Torso (Chest)
  const chestGeo = new THREE.CylinderGeometry(0.3, 0.27, 0.35, 32);
  const chest = new THREE.Mesh(chestGeo, darkerMat);
  chest.position.set(0, 1.12, 0);
  mannequin.add(chest);

  // Lower Torso (Waist)
  const waistGeo = new THREE.CylinderGeometry(0.27, 0.3, 0.35, 32);
  const waist = new THREE.Mesh(waistGeo, darkMat);
  waist.position.set(0, 0.78, 0);
  mannequin.add(waist);

  // Hips
  const hipsGeo = new THREE.SphereGeometry(0.3, 32, 16, 0, Math.PI * 2, 0, Math.PI / 2);
  const hips = new THREE.Mesh(hipsGeo, darkMat);
  hips.position.set(0, 0.6, 0);
  mannequin.add(hips);

  // Left Arm
  const armGeo = new THREE.CylinderGeometry(0.07, 0.055, 0.85, 16);
  const leftArm = new THREE.Mesh(armGeo, darkMat);
  leftArm.position.set(-0.42, 1.05, 0);
  leftArm.rotation.z = 0.12;
  mannequin.add(leftArm);

  // Right Arm
  const rightArm = new THREE.Mesh(armGeo, darkMat);
  rightArm.position.set(0.42, 1.05, 0);
  rightArm.rotation.z = -0.12;
  mannequin.add(rightArm);

  // Left Leg
  const legGeo = new THREE.CylinderGeometry(0.11, 0.08, 1.0, 16);
  const leftLeg = new THREE.Mesh(legGeo, darkerMat);
  leftLeg.position.set(-0.15, 0.05, 0);
  mannequin.add(leftLeg);

  // Right Leg
  const rightLeg = new THREE.Mesh(legGeo, darkerMat);
  rightLeg.position.set(0.15, 0.05, 0);
  mannequin.add(rightLeg);

  // Left Foot
  const footGeo = new THREE.BoxGeometry(0.12, 0.06, 0.22);
  const leftFoot = new THREE.Mesh(footGeo, darkerMat);
  leftFoot.position.set(-0.15, -0.48, 0.05);
  mannequin.add(leftFoot);
  
  // Right Foot  
  const rightFoot = new THREE.Mesh(footGeo, darkerMat);
  rightFoot.position.set(0.15, -0.48, 0.05);
  mannequin.add(rightFoot);

  scene.add(mannequin);

  // State
  let targetRotation = 0;
  let currentRotation = 0;
  let heightScale = 1;
  let weightScale = 1;
  let shoulderScale = 1;
  let chestScale = 1;
  let waistScale = 1;

  // Listen for messages from React Native
  window.addEventListener('message', function(event) {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'update') {
        heightScale = 0.85 + (data.height / 100) * 0.3;
        weightScale = 0.85 + (data.weight / 100) * 0.3;
        shoulderScale = 0.8 + (data.shoulders / 100) * 0.4;
        chestScale = 0.85 + (data.chest / 100) * 0.3;
        waistScale = 0.8 + (data.waist / 100) * 0.4;
      } else if (data.type === 'view') {
        targetRotation = data.view === 'front' ? 0 : Math.PI / 2;
      }
    } catch(e) {}
  });

  // Animation loop
  function animate() {
    requestAnimationFrame(animate);

    // Smooth rotation
    currentRotation += (targetRotation - currentRotation) * 0.08;
    mannequin.rotation.y = currentRotation;

    // Apply scaling
    mannequin.scale.set(weightScale, heightScale, weightScale);
    shoulders.scale.set(shoulderScale, 1, shoulderScale);
    chest.scale.set(chestScale, 1, chestScale);
    waist.scale.set(waistScale, 1, waistScale);
    hips.scale.set(waistScale, 1, waistScale);
    leftArm.position.x = -0.42 * shoulderScale;
    rightArm.position.x = 0.42 * shoulderScale;
    leftLeg.position.x = -0.15 * waistScale;
    rightLeg.position.x = 0.15 * waistScale;
    leftFoot.position.x = -0.15 * waistScale;
    rightFoot.position.x = 0.15 * waistScale;

    // Subtle idle animation
    mannequin.position.y = Math.sin(Date.now() * 0.001) * 0.02;

    renderer.render(scene, camera);
  }
  animate();

  // Handle resize
  window.addEventListener('resize', function() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
<\/script>
</body>
</html>
`;
}

export default function CreateAvatarScreen() {
    const navigation = useNavigation();
    const webViewRef = useRef<WebView>(null);

    // State for body proportions (0 to 100)
    const [heightValue, setHeightValue] = useState(50);
    const [weightValue, setWeightValue] = useState(50);
    const [shouldersValue, setShouldersValue] = useState(50);
    const [chestValue, setChestValue] = useState(50);
    const [waistValue, setWaistValue] = useState(50);

    const [activeTab, setActiveTab] = useState<"front" | "side">("front");

    const handleContinue = () => {
        Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        (navigation as any).navigate('AIOutfit');
    };

    // Send slider updates to the WebView
    const sendUpdate = useCallback(() => {
        webViewRef.current?.postMessage(JSON.stringify({
            type: 'update',
            height: heightValue,
            weight: weightValue,
            shoulders: shouldersValue,
            chest: chestValue,
            waist: waistValue,
        }));
    }, [heightValue, weightValue, shouldersValue, chestValue, waistValue]);

    // Send view toggle to WebView
    const sendViewToggle = useCallback((view: "front" | "side") => {
        webViewRef.current?.postMessage(JSON.stringify({
            type: 'view',
            view: view,
        }));
    }, []);

    // Trigger update when sliders change
    React.useEffect(() => {
        sendUpdate();
    }, [heightValue, weightValue, shouldersValue, chestValue, waistValue, sendUpdate]);

    React.useEffect(() => {
        sendViewToggle(activeTab);
    }, [activeTab, sendViewToggle]);

    const renderSlider = (
        label: string,
        value: number,
        setValue: React.Dispatch<React.SetStateAction<number>>,
        icon: string
    ) => {
        return (
            <View style={styles.sliderContainer} key={label}>
                <View style={styles.sliderHeader}>
                    <View style={styles.sliderLabelGroup}>
                        <Ionicons name={icon as any} size={16} color={LiquidGlass2026Theme.colors.text.secondary} />
                        <Text style={styles.sliderLabel}>{label}</Text>
                    </View>
                    <Text style={styles.sliderValue}>{Math.round(value)}</Text>
                </View>
                <Slider
                    style={styles.slider}
                    minimumValue={0}
                    maximumValue={100}
                    value={value}
                    onValueChange={(val) => {
                        setValue(val);
                    }}
                    onSlidingComplete={() => Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light)}
                    minimumTrackTintColor={AppColors.primary}
                    maximumTrackTintColor="rgba(0,0,0,0.1)"
                    thumbTintColor={AppColors.primary}
                />
            </View>
        );
    };

    return (
        <View style={styles.container}>
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity
                        style={styles.backButton}
                        onPress={() => navigation.goBack()}
                        hitSlop={{ top: 12, bottom: 12, left: 12, right: 12 }}
                    >
                        <Ionicons name="arrow-back" size={24} color={LiquidGlass2026Theme.colors.text.primary} />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>Customize Body</Text>
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
                </View>

                {/* Toggle View Type */}
                <View style={styles.viewToggleWrap}>
                    <BlurView intensity={30} tint="light" style={StyleSheet.absoluteFill} />
                    <TouchableOpacity
                        style={[styles.viewToggleOption, activeTab === "front" && styles.viewToggleActive]}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            setActiveTab("front");
                        }}
                    >
                        <Text style={[styles.viewToggleText, activeTab === "front" && styles.viewToggleTextActive]}>
                            Front View
                        </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                        style={[styles.viewToggleOption, activeTab === "side" && styles.viewToggleActive]}
                        onPress={() => {
                            Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                            setActiveTab("side");
                        }}
                    >
                        <Text style={[styles.viewToggleText, activeTab === "side" && styles.viewToggleTextActive]}>
                            Side View
                        </Text>
                    </TouchableOpacity>
                </View>

            </SafeAreaView>

            {/* Controls Liquid Glass Bottom Sheet */}
            <View style={styles.bottomSheetContainer}>
                <BlurView
                    intensity={80}
                    tint="light"
                    style={styles.glassSheet}
                >
                    <View style={styles.sheetHandle} />
                    <Text style={styles.controlsTitle}>Adjust Proportions</Text>

                    <ScrollView
                        style={styles.slidersScroll}
                        showsVerticalScrollIndicator={false}
                        contentContainerStyle={styles.slidersContentContainer}
                    >
                        {renderSlider("Height", heightValue, setHeightValue, "resize")}
                        {renderSlider("Weight", weightValue, setWeightValue, "barbell-outline")}
                        {renderSlider("Shoulder Width", shouldersValue, setShouldersValue, "body-outline")}
                        {renderSlider("Chest/Bust", chestValue, setChestValue, "shirt-outline")}
                        {renderSlider("Waist", waistValue, setWaistValue, "contract-outline")}
                    </ScrollView>
                </BlurView>
            </View>

            {/* Floating Action Button */}
            <View style={styles.floatingActionContainer}>
                <BlurView intensity={20} tint="light" style={styles.fabGlass}>
                    <TouchableOpacity
                        style={styles.continueButton}
                        onPress={handleContinue}
                        activeOpacity={0.85}
                    >
                        <Text style={styles.continueButtonText}>Save & Finalize</Text>
                    </TouchableOpacity>
                </BlurView>
            </View>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: LiquidGlass2026Theme.colors.background.secondary,
    },
    safeArea: {
        flex: 1,
        paddingBottom: height * 0.52,
    },
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
        backgroundColor: "#fff",
        alignItems: "center",
        justifyContent: "center",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 5,
        elevation: 2,
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: "700",
        color: LiquidGlass2026Theme.colors.text.primary,
    },

    // Toggle
    viewToggleWrap: {
        flexDirection: "row",
        alignSelf: "center",
        borderRadius: 24,
        padding: 6,
        marginTop: 16,
        marginBottom: 8,
        overflow: "hidden",
        backgroundColor: "rgba(255,255,255,0.6)",
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.8)",
        zIndex: 10,
    },
    viewToggleOption: {
        paddingVertical: 10,
        paddingHorizontal: 20,
        borderRadius: 18,
    },
    viewToggleActive: {
        backgroundColor: "#fff",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.06,
        shadowRadius: 6,
        elevation: 2,
    },
    viewToggleText: {
        fontSize: 14,
        fontWeight: "600",
        color: LiquidGlass2026Theme.colors.text.secondary,
    },
    viewToggleTextActive: {
        color: LiquidGlass2026Theme.colors.text.primary,
    },

    // 3D Mannequin WebView
    mannequinViewer: {
        flex: 1,
        zIndex: 1,
        overflow: "hidden",
    },
    webview: {
        flex: 1,
        backgroundColor: "transparent",
    },

    // Liquid Glass Bottom Sheet
    bottomSheetContainer: {
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        height: height * 0.52,
        borderTopLeftRadius: 30,
        borderTopRightRadius: 30,
        overflow: "hidden",
        shadowColor: "#000",
        shadowOffset: { width: 0, height: -4 },
        shadowOpacity: 0.08,
        shadowRadius: 16,
        elevation: 10,
    },
    glassSheet: {
        flex: 1,
        borderTopLeftRadius: 30,
        borderTopRightRadius: 30,
        paddingHorizontal: 24,
        paddingTop: 16,
        backgroundColor: "rgba(255, 255, 255, 0.75)",
    },
    sheetHandle: {
        width: 48,
        height: 5,
        borderRadius: 3,
        backgroundColor: "rgba(0,0,0,0.15)",
        alignSelf: "center",
        marginBottom: 20,
    },
    controlsTitle: {
        fontSize: 20,
        fontWeight: "700",
        color: LiquidGlass2026Theme.colors.text.primary,
        marginBottom: 24,
    },
    slidersScroll: {
        flex: 1,
    },
    slidersContentContainer: {
        paddingBottom: 110,
    },
    sliderContainer: {
        marginBottom: 20,
        backgroundColor: "rgba(255,255,255,0.6)",
        padding: 16,
        borderRadius: 24,
        borderWidth: 1,
        borderColor: "rgba(255,255,255,0.7)",
    },
    sliderHeader: {
        flexDirection: "row",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: 12,
    },
    sliderLabelGroup: {
        flexDirection: "row",
        alignItems: "center",
        gap: 8,
    },
    sliderLabel: {
        fontSize: 15,
        fontWeight: "600",
        color: LiquidGlass2026Theme.colors.text.primary,
    },
    sliderValue: {
        fontSize: 15,
        fontWeight: "800",
        color: AppColors.primary,
    },
    slider: {
        width: "100%",
        height: 30,
    },

    // Floating Finish Action
    floatingActionContainer: {
        position: "absolute",
        bottom: Platform.OS === "ios" ? 34 : 24,
        left: 20,
        right: 20,
        borderRadius: 30,
        overflow: "hidden",
        shadowColor: "#0A1931",
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.15,
        shadowRadius: 20,
        elevation: 8,
    },
    fabGlass: {
        padding: 6,
        backgroundColor: "rgba(255,255,255,0.4)",
    },
    continueButton: {
        backgroundColor: "#0A1931",
        height: 56,
        borderRadius: 28,
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
