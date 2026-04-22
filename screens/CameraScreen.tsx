/**
 * CameraScreen - Native-like camera with Photo/Video toggle
 * Opens actual camera with mode switch at bottom
 */

import React, { useState, useRef, useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    StatusBar,
    Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { CameraView, CameraType, useCameraPermissions, useMicrophonePermissions } from 'expo-camera';
import { Ionicons } from '@expo/vector-icons';
import { useAppNavigation } from '../hooks/useAppNavigation';
import * as Haptics from 'expo-haptics';
import * as MediaLibrary from 'expo-media-library';
import * as ImagePicker from 'expo-image-picker';
import Animated, { FadeIn, useAnimatedStyle, useSharedValue, withSpring } from 'react-native-reanimated';



type CameraMode = 'photo' | 'video';

const CameraScreen = () => {
    const navigation = useAppNavigation();
    const cameraRef = useRef<CameraView>(null);

    const [cameraPermission, requestCameraPermission] = useCameraPermissions();
    const [, requestMicPermission] = useMicrophonePermissions();
    const [mediaPermission, requestMediaPermission] = MediaLibrary.usePermissions();

    const [facing, setFacing] = useState<CameraType>('back');
    const [mode, setMode] = useState<CameraMode>('photo');
    const [isRecording, setIsRecording] = useState(false);
    const [flash, setFlash] = useState(false);

    const recordScale = useSharedValue(1);

    // Request permissions on mount
    useEffect(() => {
        requestCameraPermission();
        requestMicPermission();
        requestMediaPermission();
    }, []);

    const toggleCameraFacing = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setFacing(current => (current === 'back' ? 'front' : 'back'));
    };

    const toggleFlash = () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        setFlash(!flash);
    };

    const takePhoto = async () => {
        if (!cameraRef.current) return;

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);

        try {
            const photo = await cameraRef.current.takePictureAsync({
                quality: 0.9,
                skipProcessing: false,
            });

            if (photo?.uri && mediaPermission?.granted) {
                await MediaLibrary.saveToLibraryAsync(photo.uri);
                Alert.alert('Photo Saved', 'Photo saved to your gallery!');
                // Navigate to WardrobeVideo to analyze
                navigation.navigate('WardrobeVideo', { imageUri: photo.uri });
            }
        } catch (error) {
            console.error('Error taking photo:', error);
            Alert.alert('Error', 'Failed to take photo');
        }
    };

    const startRecording = async () => {
        if (!cameraRef.current || isRecording) return;

        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Heavy);
        setIsRecording(true);
        recordScale.value = withSpring(1.2);

        try {
            const video = await cameraRef.current.recordAsync({
                maxDuration: 60,
            });

            if (video?.uri && mediaPermission?.granted) {
                await MediaLibrary.saveToLibraryAsync(video.uri);
                Alert.alert('Video Saved', 'Video saved to your gallery!');
                // Navigate to WardrobeVideo to analyze
                navigation.navigate('WardrobeVideo', { videoUri: video.uri });
            }
        } catch (error) {
            console.error('Error recording video:', error);
        } finally {
            setIsRecording(false);
            recordScale.value = withSpring(1);
        }
    };

    const stopRecording = () => {
        if (!cameraRef.current || !isRecording) return;
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
        cameraRef.current.stopRecording();
    };

    const handleCapture = () => {
        if (mode === 'photo') {
            takePhoto();
        } else {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }
    };

    const pickFromGallery = async () => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

        try {
            // Always allow both images and videos
            const result = await ImagePicker.launchImageLibraryAsync({
                mediaTypes: ImagePicker.MediaTypeOptions.All,
                allowsEditing: false,
                quality: 0.9,
            });

            if (!result.canceled && result.assets[0]) {
                const asset = result.assets[0];
                if (asset.type === 'video') {
                    navigation.navigate('WardrobeVideo', { videoUri: asset.uri });
                } else {
                    navigation.navigate('WardrobeVideo', { imageUri: asset.uri });
                }
            }
        } catch (error) {
            console.error('Error picking from gallery:', error);
            Alert.alert('Error', 'Failed to pick from gallery');
        }
    };

    const recordButtonStyle = useAnimatedStyle(() => ({
        transform: [{ scale: recordScale.value }],
    }));

    // Permission screen
    if (!cameraPermission?.granted) {
        return (
            <View style={styles.permissionContainer}>
                <Ionicons name="camera-outline" size={64} color="#666" />
                <Text style={styles.permissionText}>Camera permission needed</Text>
                <TouchableOpacity style={styles.permissionButton} onPress={requestCameraPermission}>
                    <Text style={styles.permissionButtonText}>Grant Access</Text>
                </TouchableOpacity>
            </View>
        );
    }

    return (
        <View style={styles.container}>
            <StatusBar barStyle="light-content" />

            {/* Camera View */}
            <CameraView
                ref={cameraRef}
                style={styles.camera}
                facing={facing}
                flash={flash ? 'on' : 'off'}
            >
                {/* Top Controls */}
                <SafeAreaView style={styles.topControls}>
                    <TouchableOpacity style={styles.controlButton} onPress={() => navigation.goBack()}>
                        <Ionicons name="close" size={28} color="#fff" />
                    </TouchableOpacity>

                    <TouchableOpacity style={styles.controlButton} onPress={toggleFlash}>
                        <Ionicons name={flash ? "flash" : "flash-off"} size={24} color="#fff" />
                    </TouchableOpacity>
                </SafeAreaView>

                {/* Recording Indicator */}
                {isRecording && (
                    <Animated.View entering={FadeIn} style={styles.recordingIndicator}>
                        <View style={styles.recordingDot} />
                        <Text style={styles.recordingText}>Recording</Text>
                    </Animated.View>
                )}

                {/* Bottom Controls */}
                <View style={styles.bottomControls}>
                    {/* Mode Toggle - Photo / Video */}
                    <View style={styles.modeToggle}>
                        <TouchableOpacity
                            style={[styles.modeButton, mode === 'photo' && styles.modeButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setMode('photo');
                            }}
                        >
                            <Text style={[styles.modeText, mode === 'photo' && styles.modeTextActive]}>Photo</Text>
                        </TouchableOpacity>
                        <TouchableOpacity
                            style={[styles.modeButton, mode === 'video' && styles.modeButtonActive]}
                            onPress={() => {
                                Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
                                setMode('video');
                            }}
                        >
                            <Text style={[styles.modeText, mode === 'video' && styles.modeTextActive]}>Video</Text>
                        </TouchableOpacity>
                    </View>

                    {/* Capture Controls */}
                    <View style={styles.captureRow}>
                        {/* Gallery Button */}
                        <TouchableOpacity style={styles.galleryButton} onPress={pickFromGallery}>
                            <Ionicons name="images-outline" size={28} color="#fff" />
                        </TouchableOpacity>

                        {/* Capture Button */}
                        <TouchableOpacity
                            style={styles.captureButtonOuter}
                            onPress={handleCapture}
                            activeOpacity={0.7}
                        >
                            <Animated.View style={[
                                styles.captureButtonInner,
                                mode === 'video' && styles.captureButtonVideo,
                                isRecording && styles.captureButtonRecording,
                                recordButtonStyle,
                            ]} />
                        </TouchableOpacity>

                        {/* Flip Camera */}
                        <TouchableOpacity style={styles.flipButton} onPress={toggleCameraFacing}>
                            <Ionicons name="camera-reverse-outline" size={28} color="#fff" />
                        </TouchableOpacity>
                    </View>
                </View>
            </CameraView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#0A1931',
    },
    camera: {
        flex: 1,
    },

    // Permission
    permissionContainer: {
        flex: 1,
        backgroundColor: '#0A1931',
        alignItems: 'center',
        justifyContent: 'center',
    },
    permissionText: {
        color: '#fff',
        fontSize: 18,
        marginTop: 16,
        marginBottom: 24,
    },
    permissionButton: {
        backgroundColor: '#fff',
        paddingHorizontal: 32,
        paddingVertical: 14,
        borderRadius: 28,
    },
    permissionButtonText: {
        fontSize: 16,
        fontWeight: '600',
        color: '#0A1931',
    },

    // Top Controls
    topControls: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingTop: 50, // Increased to clear status bar/dynamic island
    },
    controlButton: {
        width: 44,
        height: 44,
        borderRadius: 22,
        backgroundColor: 'rgba(0,0,0,0.3)',
        alignItems: 'center',
        justifyContent: 'center',
    },

    // Recording Indicator
    recordingIndicator: {
        position: 'absolute',
        top: 100,
        alignSelf: 'center',
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: 'rgba(255,0,0,0.8)',
        paddingHorizontal: 16,
        paddingVertical: 8,
        borderRadius: 20,
    },
    recordingDot: {
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: '#fff',
        marginRight: 8,
    },
    recordingText: {
        color: '#fff',
        fontWeight: '600',
        fontSize: 14,
    },

    // Bottom Controls
    bottomControls: {
        position: 'absolute',
        bottom: 0,
        left: 0,
        right: 0,
        paddingBottom: 50,
    },

    // Mode Toggle
    modeToggle: {
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 32,
        marginBottom: 32,
    },
    modeButton: {
        paddingVertical: 8,
        paddingHorizontal: 4,
    },
    modeButtonActive: {
        borderBottomWidth: 2,
        borderBottomColor: '#FFD700',
    },
    modeText: {
        color: 'rgba(255,255,255,0.6)',
        fontSize: 16,
        fontWeight: '500',
    },
    modeTextActive: {
        color: '#FFD700',
    },

    // Capture Row
    captureRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 40,
    },
    captureButtonOuter: {
        width: 80,
        height: 80,
        borderRadius: 40,
        borderWidth: 4,
        borderColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    captureButtonInner: {
        width: 64,
        height: 64,
        borderRadius: 32,
        backgroundColor: '#fff',
    },
    captureButtonVideo: {
        backgroundColor: '#FF3B30',
    },
    captureButtonRecording: {
        width: 32,
        height: 32,
        borderRadius: 8,
    },
    flipButton: {
        width: 50,
        height: 50,
        borderRadius: 25,
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
    },
    galleryButton: {
        width: 50,
        height: 50,
        borderRadius: 25,
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
    },
});

export default CameraScreen;
