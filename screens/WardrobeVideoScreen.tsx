/**
 * WardrobeVideoScreen — iOS 26 Tahoe "Liquid Glass" redesign
 *
 * All analysis logic lives in:
 *   src/features/wardrobe/useVideoAnalysis.ts  (hook)
 *   src/features/wardrobe/wardrobeUtils.ts     (pure utils)
 *   src/features/wardrobe/types.ts             (shared types)
 */

import React, { useState, useEffect } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    StyleSheet,
    ScrollView,
    Alert,
    Platform,
    Dimensions,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { BlurView } from 'expo-blur';
import Animated, {
    useSharedValue,
    useAnimatedStyle,
    withRepeat,
    withSequence,
    withTiming,
    withSpring,
    withDelay,
    FadeIn,
    FadeInDown,
    FadeInUp,
    Easing,
    interpolate,
    useAnimatedProps,
} from 'react-native-reanimated';
import CorrectionModal from '../src/components/CorrectionModal';
import { RootStackParamList } from '../navigation/types';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import useWardrobeStore from '../store/wardrobeStore';
import { useVideoAnalysis } from '../src/features/wardrobe/useVideoAnalysis';
import { DetectedItem } from '../src/features/wardrobe/types';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

const AnimatedBlurView = Animated.createAnimatedComponent(BlurView);

// ─── iOS 26 Tahoe Color Palette ───
const GLASS = {
    bg: 'rgba(255, 255, 255, 0.55)',
    bgLight: 'rgba(255, 255, 255, 0.35)',
    bgDark: 'rgba(0, 0, 0, 0.06)',
    border: 'rgba(255, 255, 255, 0.7)',
    borderSubtle: 'rgba(0, 0, 0, 0.06)',
    innerShadow: 'rgba(255, 255, 255, 0.9)',
    accent: '#007AFF',       // iOS system blue
    accentLight: 'rgba(0, 122, 255, 0.12)',
    accentGlow: 'rgba(0, 122, 255, 0.25)',
    textPrimary: '#1C1C1E',
    textSecondary: 'rgba(60, 60, 67, 0.6)',
    success: '#30D158',      // iOS system green
    successGlow: 'rgba(48, 209, 88, 0.2)',
    cardBg: 'rgba(255, 255, 255, 0.65)',
    overlayGradient: ['#F2F1F6', '#E8E7ED', '#DDDCE2'] as const,
};

/** Map YOLOv8 classifier categories to closet filter categories */
const mapToClosetCategory = (yoloCategory: string): string => {
    const cat = (yoloCategory || '').toLowerCase();
    if (['shirt', 't-shirt_top', 't-shirt', 'pullover', 'coat', 'dress', 'jacket', 'hoodie', 'sweater', 'blouse', 'top'].some(k => cat.includes(k))) return 'tops';
    if (['trouser', 'pants', 'jeans', 'shorts', 'skirt', 'leggings'].some(k => cat.includes(k))) return 'bottoms';
    if (['sneaker', 'ankle boot', 'sandal', 'shoe', 'boot', 'heel', 'loafer', 'slipper'].some(k => cat.includes(k))) return 'shoes';
    if (['bag', 'hat', 'scarf', 'belt', 'sunglasses', 'watch', 'jewelry', 'cap', 'beanie'].some(k => cat.includes(k))) return 'accessories';
    return 'tops';
};

// ─── Animated Liquid Glass Spinner ───
const LiquidGlassSpinner = () => {
    const rotation = useSharedValue(0);
    const pulse = useSharedValue(1);
    const innerPulse = useSharedValue(0.6);

    useEffect(() => {
        rotation.value = withRepeat(
            withTiming(360, { duration: 2400, easing: Easing.linear }),
            -1, false
        );
        pulse.value = withRepeat(
            withSequence(
                withTiming(1.08, { duration: 1200, easing: Easing.inOut(Easing.sin) }),
                withTiming(1, { duration: 1200, easing: Easing.inOut(Easing.sin) })
            ),
            -1, true
        );
        innerPulse.value = withRepeat(
            withSequence(
                withTiming(1, { duration: 1800, easing: Easing.inOut(Easing.sin) }),
                withTiming(0.6, { duration: 1800, easing: Easing.inOut(Easing.sin) })
            ),
            -1, true
        );
    }, []);

    const outerStyle = useAnimatedStyle(() => ({
        transform: [{ scale: pulse.value }],
    }));

    const ringStyle = useAnimatedStyle(() => ({
        transform: [{ rotate: `${rotation.value}deg` }],
    }));

    const glowStyle = useAnimatedStyle(() => ({
        opacity: innerPulse.value,
    }));

    return (
        <Animated.View style={[styles.spinnerContainer, outerStyle]}>
            {/* Outer glow ring */}
            <Animated.View style={[styles.spinnerGlow, glowStyle]} />
            {/* Rotating ring */}
            <Animated.View style={[styles.spinnerRing, ringStyle]}>
                <LinearGradient
                    colors={['rgba(0,122,255,0.6)', 'rgba(0,122,255,0)', 'rgba(0,122,255,0.3)']}
                    start={{ x: 0, y: 0 }}
                    end={{ x: 1, y: 1 }}
                    style={styles.spinnerRingGradient}
                />
            </Animated.View>
            {/* Inner glass orb */}
            <BlurView intensity={40} tint="light" style={styles.spinnerInner}>
                <Ionicons name="sparkles" size={28} color={GLASS.accent} />
            </BlurView>
        </Animated.View>
    );
};

// ─── Glass Step Indicator ───
const GlassStep = ({ icon, label, index }: { icon: string; label: string; index: number }) => (
    <Animated.View entering={FadeInDown.delay(200 + index * 120).springify().damping(18)} style={styles.stepItem}>
        <BlurView intensity={30} tint="light" style={styles.stepGlass}>
            <View style={styles.stepIconGlow}>
                <Ionicons name={icon as any} size={22} color={GLASS.accent} />
            </View>
        </BlurView>
        <Text style={styles.stepLabel}>{label}</Text>
    </Animated.View>
);

// ─── Main Screen Component ───
const WardrobeVideoScreen = () => {
    const navigation = useNavigation();
    const route = useRoute<RouteProp<RootStackParamList, 'WardrobeVideo'>>();

    const { analyzing, progress, results, analyzeVideo, analyzeImage, reset } = useVideoAnalysis();

    const [correctionModal, setCorrectionModal] = useState<{
        visible: boolean;
        item: DetectedItem | null;
        index: number;
    }>({ visible: false, item: null, index: -1 });

    // Auto-start if params provided
    React.useEffect(() => {
        if (route.params?.videoUri) {
            analyzeVideo(route.params.videoUri);
        } else if (route.params?.imageUri) {
            analyzeImage(route.params.imageUri);
        }
    }, [route.params]);

    const requestPermissions = async (): Promise<boolean> => {
        if (Platform.OS === 'web') {
            // Web doesn't need permissions for file input
            return true;
        }
        
        try {
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            console.log('Media library permission status:', status);
            
            // Handle Expo Go limitations
            if (status === 'denied') {
                Alert.alert(
                    'Permission Required', 
                    'Please grant photo library permissions. Note: Expo Go has limited media access. For full functionality, use a development build.',
                    [
                        { text: 'Cancel', style: 'cancel' },
                        { text: 'Try Anyway', onPress: () => console.log('User wants to try anyway') }
                    ]
                );
                return false;
            }
            
            if (status !== 'granted') {
                console.log('Permission not granted, status:', status);
                return false;
            }
            
            return true;
        } catch (error) {
            console.error('Permission request error:', error);
            // In Expo Go, sometimes permissions work differently
            return true; // Try anyway
        }
    };

    const pickVideoWeb = () => {
        console.log('Web file picker activated');
        
        // Create a file input element
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*'; // Only accept images on web
        input.multiple = false;
        
        input.onchange = async (event) => {
            const file = (event.target as HTMLInputElement).files?.[0];
            if (file) {
                console.log('Web file selected:', file);
                
                try {
                    // Convert file to base64
                    const base64 = await fileToBase64(file);
                    console.log('File converted to base64, length:', base64.length);
                    
                    // Process as image
                    console.log('Web: Processing image file');
                    analyzeImage(base64);
                } catch (error) {
                    console.error('Web file processing error:', error);
                    Alert.alert('Error', 'Failed to process the selected file.');
                }
            }
        };
        
        // Trigger the file picker
        input.click();
    };

    const fileToBase64 = (file: File): Promise<string> => {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                const result = reader.result as string;
                // Remove data URL prefix to get pure base64
                const base64 = result.split(',')[1];
                resolve(base64);
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    };

    const pickVideo = async () => {
        console.log('pickVideo called, platform:', Platform.OS);
        const hasPermission = await requestPermissions();
        console.log('Permission granted:', hasPermission);
        if (!hasPermission) return;
        
        if (Platform.OS === 'web') {
            // Web: Use HTML file input
            console.log('Web: Using web file picker');
            pickVideoWeb();
            return;
        }
        
        try {
            console.log('Native: Attempting to launch image library...');
            
            // Try different approaches for Expo Go vs development builds
            let result;
            try {
                // First try with All media types
                result = await ImagePicker.launchImageLibraryAsync({
                    mediaTypes: ['images', 'videos'],
                    allowsEditing: false,
                    quality: 1,
                });
            } catch (error) {
                console.warn('All media types failed, trying images only:', error);
                // Fallback to images only for Expo Go
                result = await ImagePicker.launchImageLibraryAsync({
                    mediaTypes: ['images'],
                    allowsEditing: false,
                    quality: 1,
                });
            }
            
            console.log('Native: ImagePicker result received:', JSON.stringify(result, null, 2));
            
            if (!result.canceled && result.assets && result.assets.length > 0) {
                const asset = result.assets[0];
                console.log('Native: Selected asset details:', {
                    uri: asset.uri,
                    fileName: asset.fileName,
                    mimeType: asset.mimeType,
                    duration: asset.duration,
                    type: asset.type
                });
                
                if (asset.uri) {
                    // Determine if video or image based on mimeType or duration
                    const isVideo = asset.mimeType?.includes('video') || asset.duration || asset.type === 'video';
                    
                    if (isVideo) {
                        console.log('Native: Detected video file, analyzing as video');
                        analyzeVideo(asset.uri);
                    } else {
                        console.log('Native: Detected image file, analyzing as image');
                        analyzeImage(asset.uri);
                    }
                }
            } else {
                console.log('Native: User cancelled selection or no assets found');
                
                // Provide helpful message for Expo Go users
                if (__DEV__) {
                    Alert.alert(
                        'No Media Selected', 
                        'If you\'re using Expo Go, media access may be limited. Try:\n\n1. Selecting images\n2. Using a development build for full access\n3. Using a physical device',
                        [{ text: 'OK' }]
                    );
                } else {
                    Alert.alert('Info', 'No media selected or selection was cancelled.');
                }
            }
        } catch (error: any) {
            console.error('Native: Media picker error:', error);
            
            // Provide specific error message for Expo Go limitations
            if (error.message?.includes('media library') || error.message?.includes('permission')) {
                Alert.alert(
                    'Expo Go Limitation', 
                    'Media library access is limited in Expo Go. For full functionality, create a development build.\n\nYou can still try uploading images.',
                    [{ text: 'OK' }]
                );
            } else {
                Alert.alert('Error', `Failed to pick media: ${error?.message || 'Unknown error'}. Check console for details.`);
            }
        }
    };

    // Navigate to detail editor for the first detected item
    const handleReviewAndSave = () => {
        if (!results || results.detectedItems.length === 0) return;
        
        const firstItem = results.detectedItems[0];
        const imgUrl = firstItem.frameImage || firstItem.cutoutImage || '';
        
        // Navigate to detail editor with detected data
        (navigation as any).navigate('ClothingDetailEditor', {
            imageUri: imgUrl,
            detectedType: mapToClosetCategory(firstItem.itemType),
            detectedColor: firstItem.color?.toLowerCase() || 'beige',
            detectedItem: firstItem,
        });
    };

    const saveToWardrobe = async () => {
        if (!results || results.detectedItems.length === 0) return;
        const { user } = useAuthStore.getState();
        if (!user) {
            Alert.alert('Login Required', 'Please login to save items.');
            return;
        }
        try {
            const { addItem } = useWardrobeStore.getState();

            // Optionally upsert profile metadata if needed, though auth usually handles it
            await supabase
                .from('profiles')
                .upsert(
                    { id: user.id, email: user.email, username: user.username || '' },
                    { onConflict: 'id' }
                );

            for (const item of results.detectedItems) {
                const imgUrl = item.frameImage || item.cutoutImage || '';
                console.log('[Save] Adding Item to Queue:', item.itemType);

                await addItem({
                    userId: user.id,
                    imageUrl: imgUrl,
                    category: mapToClosetCategory(item.itemType) as any,
                    subCategory: item.itemType,
                    primaryColor: item.color,
                    colorHex: '#000000',
                    pattern: 'solid',
                    material: item.material || '',
                    brand: '',
                    name: item.description || item.productDescription || `${item.color} ${item.itemType}`,
                    seasons: [],
                    occasions: [],
                    detectionConfidence: item.confidence || 0.8,
                });
            }

            Alert.alert(
                'Saved Locally!',
                `${results.detectedItems.length} item(s) saved to your wardrobe. They will sync automatically.`,
                [
                    {
                        text: 'View Wardrobe', onPress: () => {
                            reset();
                            (navigation as any).navigate('Main', { screen: 'Closet' });
                        }
                    },
                    { text: 'OK', onPress: () => reset() },
                ]
            );
        } catch (error: any) {
            Alert.alert('Error', 'Failed to save. ' + (error.message || ''));
        }
    };

    // Group items by outfitId for display
    const outfitGroups = React.useMemo(() => {
        if (!results) return {};
        const groups: Record<number, DetectedItem[]> = {};
        results.detectedItems.forEach(item => {
            const id = item.outfitId || 1;
            if (!groups[id]) groups[id] = [];
            groups[id].push(item);
        });
        return groups;
    }, [results]);

    const sortedOutfitIds = Object.keys(outfitGroups).map(Number).sort((a, b) => a - b);

    return (
        <View style={styles.container}>
            {/* Tahoe-style layered background */}
            <LinearGradient colors={GLASS.overlayGradient} style={StyleSheet.absoluteFill} />

            <SafeAreaView style={styles.safeArea}>
                {/* ─── Glass Header ─── */}
                <Animated.View entering={FadeIn.duration(500)} style={styles.headerWrap}>
                    <BlurView intensity={60} tint="light" style={styles.headerBlur}>
                        <TouchableOpacity
                            onPress={() => navigation.goBack()}
                            style={styles.backPill}
                            activeOpacity={0.7}
                        >
                            <Ionicons name="chevron-back" size={20} color={GLASS.textPrimary} />
                        </TouchableOpacity>
                        <Text style={styles.headerTitle}>AI Wardrobe Scan</Text>
                        <View style={{ width: 36 }} />
                    </BlurView>
                </Animated.View>

                <ScrollView
                    showsVerticalScrollIndicator={false}
                    contentContainerStyle={styles.scrollContent}
                >
                    {/* ─── Hero / Instructions ─── */}
                    {!results && !analyzing && (
                        <Animated.View entering={FadeInDown.springify().damping(20)} style={styles.heroSection}>
                            <Animated.Text entering={FadeIn.delay(100)} style={styles.heroTitle}>
                                Digitize Your Closet
                            </Animated.Text>
                            <Animated.Text entering={FadeIn.delay(200)} style={styles.heroSubtitle}>
                                Upload a quick video of your clothes, and our AI will automatically detect and catalog them.
                            </Animated.Text>

                            {/* Glass steps */}
                            <View style={styles.stepsRow}>
                                <GlassStep icon="videocam-outline" label="Record" index={0} />
                                <View style={styles.stepConnector}>
                                    <View style={styles.stepConnectorLine} />
                                </View>
                                <GlassStep icon="sparkles-outline" label="Analyze" index={1} />
                                <View style={styles.stepConnector}>
                                    <View style={styles.stepConnectorLine} />
                                </View>
                                <GlassStep icon="shirt-outline" label="Get Items" index={2} />
                            </View>
                        </Animated.View>
                    )}

                    {/* ─── Upload CTA ─── */}
                    {!analyzing && !results && (
                        <Animated.View entering={FadeInUp.delay(500).springify().damping(16)}>
                            <TouchableOpacity
                                style={styles.uploadCard}
                                onPress={pickVideo}
                                activeOpacity={0.85}
                            >
                                <BlurView intensity={25} tint="light" style={styles.uploadBlur}>
                                    <LinearGradient
                                        colors={['rgba(0,122,255,0.08)', 'rgba(0,122,255,0.02)']}
                                        style={StyleSheet.absoluteFill}
                                    />
                                    <View style={styles.uploadIconContainer}>
                                        <LinearGradient
                                            colors={[GLASS.accent, '#5856D6']}
                                            start={{ x: 0, y: 0 }}
                                            end={{ x: 1, y: 1 }}
                                            style={styles.uploadIconGradient}
                                        >
                                            <Ionicons name="images-outline" size={32} color="#fff" />
                                        </LinearGradient>
                                    </View>
                                    <Text style={styles.uploadTitle}>Select from Gallery</Text>
                                    <Text style={styles.uploadSubtitle}>
                                        {Platform.OS === 'web' 
                                            ? 'Choose a photo from your device' 
                                            : 'Choose a video or photo from your device'
                                        }
                                    </Text>
                                </BlurView>
                            </TouchableOpacity>
                        </Animated.View>
                    )}

                    {/* ─── Analyzing State ─── */}
                    {analyzing && (
                        <Animated.View entering={FadeIn.duration(600)} style={styles.analyzingContainer}>
                            <LiquidGlassSpinner />
                            <Animated.Text entering={FadeIn.delay(300)} style={styles.analyzingTitle}>
                                {progress}
                            </Animated.Text>
                            <Animated.Text entering={FadeIn.delay(500)} style={styles.analyzingSubtext}>
                                Our AI is analyzing every frame of your video...
                            </Animated.Text>
                        </Animated.View>
                    )}

                    {/* ─── Results ─── */}
                    {results && !analyzing && (
                        <Animated.View entering={FadeIn.duration(400)} style={styles.resultsContainer}>
                            {/* Results header */}
                            <View style={styles.resultsHeader}>
                                <View>
                                    <Text style={styles.resultsTitle}>Analysis Complete</Text>
                                    <Text style={styles.resultsSubtitle}>
                                        Found {results.detectedItems.length} items
                                    </Text>
                                </View>
                                <TouchableOpacity
                                    style={styles.retryPill}
                                    onPress={() => { reset(); pickVideo(); }}
                                    activeOpacity={0.7}
                                >
                                    <BlurView intensity={30} tint="light" style={styles.retryBlur}>
                                        <Ionicons name="refresh" size={18} color={GLASS.accent} />
                                    </BlurView>
                                </TouchableOpacity>
                            </View>

                            {/* Outfit groups */}
                            {sortedOutfitIds.map((outfitId, outfitIndex) => (
                                <Animated.View
                                    key={outfitId}
                                    entering={FadeInDown.delay(outfitIndex * 100).springify().damping(18)}
                                    style={styles.outfitGroup}
                                >
                                    {/* Outfit badge row */}
                                    <View style={styles.outfitHeader}>
                                        <View style={styles.outfitBadge}>
                                            <LinearGradient
                                                colors={[GLASS.accent, '#5856D6']}
                                                style={styles.outfitBadgeGradient}
                                            >
                                                <Text style={styles.outfitBadgeText}>{outfitIndex + 1}</Text>
                                            </LinearGradient>
                                        </View>
                                        <Text style={styles.outfitTitle}>Outfit {outfitIndex + 1}</Text>
                                        <Text style={styles.outfitCount}>
                                            ({outfitGroups[outfitId].length} items)
                                        </Text>
                                    </View>

                                    {/* Item cards */}
                                    {outfitGroups[outfitId].map((item, itemIdx) => (
                                        <Animated.View
                                            key={itemIdx}
                                            entering={FadeInDown.delay(150 + itemIdx * 80).springify().damping(20)}
                                        >
                                            <View style={styles.resultCard}>
                                                <BlurView intensity={20} tint="light" style={styles.resultCardBlur}>
                                                    {/* Item icon */}
                                                    <View style={styles.resultIconWrap}>
                                                        <View style={[
                                                            styles.resultIconBg,
                                                            { backgroundColor: item.colorHex || GLASS.accentLight }
                                                        ]}>
                                                            <Ionicons
                                                                name={
                                                                    item.position === 'upper' ? 'shirt' :
                                                                        item.position === 'lower' ? 'layers' :
                                                                            item.position === 'feet' ? 'footsteps' : 'shirt'
                                                                }
                                                                size={18}
                                                                color="#fff"
                                                            />
                                                        </View>
                                                    </View>

                                                    {/* Item info */}
                                                    <View style={styles.resultInfo}>
                                                        <Text style={styles.resultType}>{item.itemType}</Text>
                                                        <Text style={styles.resultDetails}>
                                                            {item.color}{item.material ? ` · ${item.material}` : ''}
                                                        </Text>
                                                    </View>

                                                    {/* Check badge */}
                                                    <View style={styles.checkBadge}>
                                                        <Ionicons name="checkmark-circle" size={22} color={GLASS.success} />
                                                    </View>
                                                </BlurView>
                                            </View>
                                        </Animated.View>
                                    ))}
                                </Animated.View>
                            ))}

                            {/* Correction Modal */}
                            <CorrectionModal
                                visible={correctionModal.visible}
                                onClose={() => setCorrectionModal({ visible: false, item: null, index: -1 })}
                                originalType={correctionModal.item?.itemType || ''}
                                category={correctionModal.item?.position || 'upper_clothes'}
                                confidence={correctionModal.item?.confidence || 0.5}
                                onCorrected={(newType: string) => {
                                    if (results && correctionModal.index >= 0) {
                                        const updated = [...results.detectedItems];
                                        updated[correctionModal.index] = { ...updated[correctionModal.index], itemType: newType };
                                    }
                                }}
                            />

                            {/* Save Button — Liquid Glass Dark */}
                            <Animated.View entering={FadeInUp.delay(400).springify().damping(16)}>
                                <TouchableOpacity
                                    style={styles.saveButton}
                                    onPress={handleReviewAndSave}
                                    activeOpacity={0.85}
                                >
                                    <LinearGradient
                                        colors={['#1C1C1E', '#2C2C2E']}
                                        start={{ x: 0, y: 0 }}
                                        end={{ x: 1, y: 1 }}
                                        style={styles.saveGradient}
                                    >
                                        <Text style={styles.saveText}>Review & Save</Text>
                                        <View style={styles.saveArrow}>
                                            <Ionicons name="arrow-forward" size={18} color="#fff" />
                                        </View>
                                    </LinearGradient>
                                </TouchableOpacity>
                            </Animated.View>
                        </Animated.View>
                    )}
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

// ─── iOS 26 Tahoe Liquid Glass Styles ───
const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#F2F1F6' },
    safeArea: { flex: 1 },

    // Header
    headerWrap: {
        marginHorizontal: 16,
        marginTop: 4,
        borderRadius: 22,
        overflow: 'hidden',
    },
    headerBlur: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 14,
        paddingVertical: 12,
        overflow: 'hidden',
        borderRadius: 22,
        borderWidth: 0.5,
        borderColor: GLASS.border,
    },
    backPill: {
        width: 36,
        height: 36,
        borderRadius: 18,
        backgroundColor: GLASS.bgLight,
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 0.5,
        borderColor: GLASS.border,
    },
    headerTitle: {
        fontSize: 17,
        fontWeight: '600',
        color: GLASS.textPrimary,
        letterSpacing: -0.2,
    },

    scrollContent: { padding: 20, paddingBottom: 48 },

    // Hero
    heroSection: { marginBottom: 28, alignItems: 'center' },
    heroTitle: {
        fontSize: 30,
        fontWeight: '700',
        color: GLASS.textPrimary,
        marginBottom: 10,
        textAlign: 'center',
        letterSpacing: -0.5,
    },
    heroSubtitle: {
        fontSize: 16,
        color: GLASS.textSecondary,
        textAlign: 'center',
        lineHeight: 23,
        marginBottom: 32,
        paddingHorizontal: 12,
    },

    // Steps
    stepsRow: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
    },
    stepItem: { alignItems: 'center' },
    stepGlass: {
        width: 56,
        height: 56,
        borderRadius: 28,
        overflow: 'hidden',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 0.5,
        borderColor: GLASS.border,
        marginBottom: 8,
    },
    stepIconGlow: {
        width: 56,
        height: 56,
        borderRadius: 28,
        backgroundColor: GLASS.accentLight,
        alignItems: 'center',
        justifyContent: 'center',
    },
    stepLabel: {
        fontSize: 12,
        fontWeight: '600',
        color: GLASS.accent,
        letterSpacing: 0.2,
    },
    stepConnector: {
        width: 28,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 22,
    },
    stepConnectorLine: {
        width: 28,
        height: 1.5,
        backgroundColor: 'rgba(0,122,255,0.2)',
        borderRadius: 1,
    },

    // Upload CTA
    uploadCard: {
        width: '100%',
        borderRadius: 24,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.08,
        shadowRadius: 24,
        elevation: 6,
    },
    uploadBlur: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: 32,
        overflow: 'hidden',
        borderRadius: 24,
        borderWidth: 0.5,
        borderColor: GLASS.border,
    },
    uploadIconContainer: { marginBottom: 18 },
    uploadIconGradient: {
        width: 68,
        height: 68,
        borderRadius: 34,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: GLASS.accent,
        shadowOffset: { width: 0, height: 6 },
        shadowOpacity: 0.3,
        shadowRadius: 16,
    },
    uploadTitle: {
        fontSize: 19,
        fontWeight: '600',
        color: GLASS.textPrimary,
        marginBottom: 4,
        letterSpacing: -0.2,
    },
    uploadSubtitle: {
        fontSize: 14,
        color: GLASS.textSecondary,
    },

    // Analyzing state
    analyzingContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 60,
    },
    spinnerContainer: {
        width: 110,
        height: 110,
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 28,
    },
    spinnerGlow: {
        position: 'absolute',
        width: 110,
        height: 110,
        borderRadius: 55,
        backgroundColor: GLASS.accentGlow,
    },
    spinnerRing: {
        position: 'absolute',
        width: 100,
        height: 100,
        borderRadius: 50,
        overflow: 'hidden',
    },
    spinnerRingGradient: {
        flex: 1,
        borderRadius: 50,
        borderWidth: 2.5,
        borderColor: 'transparent',
    },
    spinnerInner: {
        width: 72,
        height: 72,
        borderRadius: 36,
        overflow: 'hidden',
        alignItems: 'center',
        justifyContent: 'center',
        borderWidth: 0.5,
        borderColor: GLASS.border,
        backgroundColor: GLASS.bg,
    },
    analyzingTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: GLASS.textPrimary,
        marginBottom: 8,
        letterSpacing: -0.3,
    },
    analyzingSubtext: {
        fontSize: 14,
        color: GLASS.textSecondary,
        textAlign: 'center',
        lineHeight: 20,
        paddingHorizontal: 32,
    },

    // Results
    resultsContainer: { width: '100%' },
    resultsHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 24,
    },
    resultsTitle: {
        fontSize: 22,
        fontWeight: '700',
        color: GLASS.textPrimary,
        letterSpacing: -0.3,
    },
    resultsSubtitle: {
        fontSize: 14,
        color: GLASS.textSecondary,
        marginTop: 2,
    },
    retryPill: {
        width: 40,
        height: 40,
        borderRadius: 20,
        overflow: 'hidden',
    },
    retryBlur: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        borderRadius: 20,
        borderWidth: 0.5,
        borderColor: GLASS.border,
    },

    // Outfit groups
    outfitGroup: { marginBottom: 20 },
    outfitHeader: {
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 12,
        paddingHorizontal: 2,
    },
    outfitBadge: {
        width: 30,
        height: 30,
        borderRadius: 15,
        overflow: 'hidden',
        marginRight: 10,
    },
    outfitBadgeGradient: {
        flex: 1,
        alignItems: 'center',
        justifyContent: 'center',
        borderRadius: 15,
    },
    outfitBadgeText: {
        color: '#fff',
        fontWeight: '700',
        fontSize: 14,
    },
    outfitTitle: {
        fontSize: 16,
        fontWeight: '600',
        color: GLASS.textPrimary,
    },
    outfitCount: {
        fontSize: 13,
        color: GLASS.textSecondary,
        marginLeft: 6,
    },

    // Result card
    resultCard: {
        marginBottom: 10,
        borderRadius: 18,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.04,
        shadowRadius: 12,
        elevation: 2,
    },
    resultCardBlur: {
        flexDirection: 'row',
        alignItems: 'center',
        padding: 14,
        overflow: 'hidden',
        borderRadius: 18,
        borderWidth: 0.5,
        borderColor: GLASS.border,
        backgroundColor: GLASS.cardBg,
    },
    resultIconWrap: {
        marginRight: 14,
    },
    resultIconBg: {
        width: 42,
        height: 42,
        borderRadius: 12,
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.1,
        shadowRadius: 6,
    },
    resultInfo: { flex: 1 },
    resultType: {
        fontSize: 16,
        fontWeight: '600',
        color: GLASS.textPrimary,
        marginBottom: 2,
        textTransform: 'capitalize',
        letterSpacing: -0.1,
    },
    resultDetails: {
        fontSize: 13,
        color: GLASS.textSecondary,
    },
    checkBadge: {
        width: 32,
        height: 32,
        borderRadius: 16,
        backgroundColor: GLASS.successGlow,
        alignItems: 'center',
        justifyContent: 'center',
        marginLeft: 8,
    },

    // Save button
    saveButton: {
        marginTop: 24,
        borderRadius: 20,
        overflow: 'hidden',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 8 },
        shadowOpacity: 0.18,
        shadowRadius: 20,
        elevation: 10,
    },
    saveGradient: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 18,
        borderRadius: 20,
    },
    saveText: {
        fontSize: 16,
        fontWeight: '700',
        color: '#fff',
        marginRight: 10,
        letterSpacing: -0.1,
    },
    saveArrow: {
        width: 28,
        height: 28,
        borderRadius: 14,
        backgroundColor: 'rgba(255,255,255,0.15)',
        alignItems: 'center',
        justifyContent: 'center',
    },
});

export default WardrobeVideoScreen;
