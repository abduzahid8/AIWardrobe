/**
 * WardrobeVideoScreen — thin orchestrator
 *
 * All analysis logic lives in:
 *   src/features/wardrobe/useVideoAnalysis.ts  (hook)
 *   src/features/wardrobe/wardrobeUtils.ts     (pure utils)
 *   src/features/wardrobe/types.ts             (shared types)
 */

import React, { useState } from 'react';
import {
    View,
    Text,
    TouchableOpacity,
    ActivityIndicator,
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
import CorrectionModal from '../src/components/CorrectionModal';
import { RootStackParamList } from '../navigation/types';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import { useVideoAnalysis } from '../src/features/wardrobe/useVideoAnalysis';
import { DetectedItem } from '../src/features/wardrobe/types';

const { width } = Dimensions.get('window');

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
        if (Platform.OS !== 'web') {
            const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
            if (status !== 'granted') {
                Alert.alert('Permission Required', 'Please grant camera roll permissions to upload videos.');
                return false;
            }
        }
        return true;
    };

    const pickVideo = async () => {
        const hasPermission = await requestPermissions();
        if (!hasPermission) return;
        try {
            const result = await ImagePicker.launchImageLibraryAsync({
                // @ts-ignore — mediaTypes accepts string array in newer Expo versions
                mediaTypes: ['videos'],
                allowsEditing: false,
                quality: 1,
            });
            if (!result.canceled && result.assets[0]) {
                analyzeVideo(result.assets[0].uri);
            }
        } catch (error) {
            Alert.alert('Error', 'Failed to pick video. Please try again.');
        }
    };

    const saveToWardrobe = async () => {
        if (!results || results.detectedItems.length === 0) return;
        const { user } = useAuthStore.getState();
        if (!user) {
            Alert.alert('Login Required', 'Please login to save items.');
            return;
        }
        try {
            const itemsToSave = results.detectedItems.map((item: DetectedItem) => ({
                type: item.itemType,
                category: item.itemType,
                color: item.color,
                style: item.style,
                description: item.description || item.productDescription || `${item.color} ${item.itemType}`,
                material: item.material,
                image: item.frameImage,
                outfitId: item.outfitId || 1,
            }));
            const { error } = await supabase.functions.invoke('save-wardrobe-items', {
                body: { items: itemsToSave },
            });
            if (error) throw error;
            Alert.alert(
                'Saved!',
                `${results.detectedItems.length} item(s) saved to your wardrobe!`,
                [
                    { text: 'View Wardrobe', onPress: () => (navigation as any).navigate('Home', { screen: 'Profile' }) },
                    { text: 'OK' },
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
            <LinearGradient colors={['#ffffff', '#f0f4ff', '#e6eeff']} style={StyleSheet.absoluteFill} />
            <SafeAreaView style={styles.safeArea}>
                {/* Header */}
                <View style={styles.header}>
                    <TouchableOpacity onPress={() => navigation.goBack()} style={styles.backButton}>
                        <Ionicons name="chevron-back" size={28} color="#1a1a1a" />
                    </TouchableOpacity>
                    <Text style={styles.headerTitle}>AI Wardrobe Scan</Text>
                    <View style={{ width: 28 }} />
                </View>

                <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>

                    {/* Hero / Instructions */}
                    {!results && !analyzing && (
                        <View style={styles.heroSection}>
                            <Text style={styles.heroTitle}>Digitize Your Closet</Text>
                            <Text style={styles.heroSubtitle}>
                                Upload a quick video of your clothes, and our AI will automatically detect and catalog them.
                            </Text>
                            <View style={styles.stepsContainer}>
                                {[
                                    { icon: 'videocam-outline', label: 'Record Video' },
                                    { icon: 'sparkles-outline', label: 'AI Analysis' },
                                    { icon: 'shirt-outline', label: 'Get Items' },
                                ].map((step, i) => (
                                    <React.Fragment key={step.label}>
                                        <View style={styles.stepItem}>
                                            <View style={styles.stepIconBg}>
                                                <Ionicons name={step.icon as any} size={24} color="#4f46e5" />
                                            </View>
                                            <Text style={styles.stepText}>{step.label}</Text>
                                        </View>
                                        {i < 2 && <View style={styles.stepLine} />}
                                    </React.Fragment>
                                ))}
                            </View>
                        </View>
                    )}

                    {/* Upload CTA */}
                    {!analyzing && !results && (
                        <TouchableOpacity style={styles.uploadCard} onPress={pickVideo} activeOpacity={0.9}>
                            <LinearGradient
                                colors={['#4f46e5', '#3730a3']}
                                start={{ x: 0, y: 0 }}
                                end={{ x: 1, y: 1 }}
                                style={styles.uploadGradient}
                            >
                                <View style={styles.uploadIconContainer}>
                                    <Ionicons name="images-outline" size={40} color="#fff" />
                                </View>
                                <Text style={styles.uploadTitle}>Select from Gallery</Text>
                                <Text style={styles.uploadSubtitle}>Choose a video from your device</Text>
                            </LinearGradient>
                        </TouchableOpacity>
                    )}

                    {/* Loading State */}
                    {analyzing && (
                        <View style={styles.loadingContainer}>
                            <View style={styles.loadingCircle}>
                                <ActivityIndicator size="large" color="#4f46e5" />
                            </View>
                            <Text style={styles.loadingText}>{progress}</Text>
                            <Text style={styles.loadingSubtext}>
                                Our AI is analyzing every frame of your video...
                            </Text>
                        </View>
                    )}

                    {/* Results */}
                    {results && !analyzing && (
                        <View style={styles.resultsContainer}>
                            <View style={styles.resultsHeader}>
                                <View>
                                    <Text style={styles.resultsTitle}>Analysis Complete</Text>
                                    <Text style={styles.resultsSubtitle}>
                                        Found {results.detectedItems.length} items
                                    </Text>
                                </View>
                                <TouchableOpacity
                                    style={styles.retryButton}
                                    onPress={() => { reset(); pickVideo(); }}
                                >
                                    <Ionicons name="refresh" size={20} color="#4f46e5" />
                                </TouchableOpacity>
                            </View>

                            {/* Items grouped by outfit */}
                            {sortedOutfitIds.map((outfitId, outfitIndex) => (
                                <View key={outfitId} style={{ marginBottom: 16 }}>
                                    <View style={styles.outfitHeader}>
                                        <View style={styles.outfitBadge}>
                                            <Text style={styles.outfitBadgeText}>{outfitIndex + 1}</Text>
                                        </View>
                                        <Text style={styles.outfitTitle}>Outfit {outfitIndex + 1}</Text>
                                        <Text style={styles.outfitCount}>({outfitGroups[outfitId].length} items)</Text>
                                    </View>

                                    {outfitGroups[outfitId].map((item, itemIdx) => (
                                        <View key={itemIdx} style={[styles.resultCard, { marginBottom: 8 }]}>
                                            <View style={styles.resultIcon}>
                                                <View style={[styles.colorDot, { backgroundColor: item.colorHex || '#eee' }]}>
                                                    <Ionicons
                                                        name={
                                                            item.position === 'upper' ? 'shirt' :
                                                            item.position === 'lower' ? 'layers' :
                                                            item.position === 'feet' ? 'footsteps' : 'shirt'
                                                        }
                                                        size={20}
                                                        color="#fff"
                                                    />
                                                </View>
                                            </View>
                                            <View style={styles.resultInfo}>
                                                <Text style={styles.resultType}>{item.itemType}</Text>
                                                <Text style={styles.resultDetails}>
                                                    {item.color}{item.material ? ` • ${item.material}` : ''}
                                                </Text>
                                            </View>
                                            <View style={styles.checkIcon}>
                                                <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                                            </View>
                                        </View>
                                    ))}
                                </View>
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

                            <TouchableOpacity style={styles.saveButton} onPress={saveToWardrobe}>
                                <LinearGradient
                                    colors={['#1a1a1a', '#0A1931']}
                                    style={styles.saveButtonGradient}
                                >
                                    <Text style={styles.saveButtonText}>Save All to Wardrobe</Text>
                                    <Ionicons name="arrow-forward" size={20} color="#fff" />
                                </LinearGradient>
                            </TouchableOpacity>
                        </View>
                    )}
                </ScrollView>
            </SafeAreaView>
        </View>
    );
};

const styles = StyleSheet.create({
    container: { flex: 1, backgroundColor: '#fff' },
    safeArea: { flex: 1 },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
    },
    backButton: {
        width: 40, height: 40, borderRadius: 20,
        backgroundColor: '#fff', alignItems: 'center', justifyContent: 'center',
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05, shadowRadius: 8, elevation: 2,
    },
    headerTitle: { fontSize: 18, fontWeight: '700', color: '#1a1a1a', letterSpacing: 0.5 },
    scrollContent: { padding: 24, paddingBottom: 40 },
    heroSection: { marginBottom: 32, alignItems: 'center' },
    heroTitle: { fontSize: 28, fontWeight: '800', color: '#1a1a1a', marginBottom: 8, textAlign: 'center' },
    heroSubtitle: { fontSize: 16, color: '#666', textAlign: 'center', lineHeight: 24, marginBottom: 32 },
    stepsContainer: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', width: '100%' },
    stepItem: { alignItems: 'center' },
    stepIconBg: {
        width: 48, height: 48, borderRadius: 24,
        backgroundColor: '#eef2ff', alignItems: 'center', justifyContent: 'center', marginBottom: 8,
    },
    stepText: { fontSize: 12, fontWeight: '600', color: '#4f46e5' },
    stepLine: { width: 30, height: 2, backgroundColor: '#e0e7ff', marginHorizontal: 8, marginBottom: 20 },
    uploadCard: {
        width: '100%', height: 200, borderRadius: 24,
        shadowColor: '#4f46e5', shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.2, shadowRadius: 20, elevation: 10,
    },
    uploadGradient: { flex: 1, borderRadius: 24, alignItems: 'center', justifyContent: 'center', padding: 20 },
    uploadIconContainer: {
        width: 80, height: 80, borderRadius: 40,
        backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center',
        marginBottom: 16, borderWidth: 1, borderColor: 'rgba(255,255,255,0.3)',
    },
    uploadTitle: { fontSize: 20, fontWeight: '700', color: '#fff', marginBottom: 4 },
    uploadSubtitle: { fontSize: 14, color: 'rgba(255,255,255,0.8)' },
    loadingContainer: { alignItems: 'center', justifyContent: 'center', paddingVertical: 40 },
    loadingCircle: {
        width: 80, height: 80, borderRadius: 40, backgroundColor: '#fff',
        alignItems: 'center', justifyContent: 'center', marginBottom: 24,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1, shadowRadius: 12, elevation: 5,
    },
    loadingText: { fontSize: 18, fontWeight: '700', color: '#1a1a1a', marginBottom: 8 },
    loadingSubtext: { fontSize: 14, color: '#666', textAlign: 'center' },
    resultsContainer: { width: '100%' },
    resultsHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 },
    resultsTitle: { fontSize: 20, fontWeight: '700', color: '#1a1a1a' },
    resultsSubtitle: { fontSize: 14, color: '#666' },
    retryButton: { width: 40, height: 40, borderRadius: 20, backgroundColor: '#eef2ff', alignItems: 'center', justifyContent: 'center' },
    outfitHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 10, paddingHorizontal: 4 },
    outfitBadge: { width: 32, height: 32, borderRadius: 16, backgroundColor: '#4f46e5', alignItems: 'center', justifyContent: 'center', marginRight: 10 },
    outfitBadgeText: { color: '#fff', fontWeight: '700', fontSize: 14 },
    outfitTitle: { fontSize: 16, fontWeight: '600', color: '#1a1a1a' },
    outfitCount: { fontSize: 12, color: '#666', marginLeft: 8 },
    resultCard: {
        flexDirection: 'row', alignItems: 'center', backgroundColor: '#fff',
        padding: 16, borderRadius: 16, marginBottom: 12,
        shadowColor: '#0A1931', shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05, shadowRadius: 8, elevation: 2,
        borderWidth: 1, borderColor: '#f0f0f0',
    },
    resultIcon: { width: 48, height: 48, borderRadius: 24, backgroundColor: '#eef2ff', alignItems: 'center', justifyContent: 'center', marginRight: 16 },
    colorDot: { width: 40, height: 40, borderRadius: 8, alignItems: 'center', justifyContent: 'center' },
    resultInfo: { flex: 1 },
    resultType: { fontSize: 16, fontWeight: '600', color: '#1a1a1a', marginBottom: 2, textTransform: 'capitalize' },
    resultDetails: { fontSize: 12, color: '#666' },
    checkIcon: { marginLeft: 8 },
    saveButton: { marginTop: 24, shadowColor: '#0A1931', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.2, shadowRadius: 12, elevation: 8 },
    saveButtonGradient: { flexDirection: 'row', alignItems: 'center', justifyContent: 'center', paddingVertical: 18, borderRadius: 16 },
    saveButtonText: { fontSize: 16, fontWeight: '700', color: '#fff', marginRight: 8 },
});

export default WardrobeVideoScreen;
