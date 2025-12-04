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
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as FileSystem from 'expo-file-system/legacy';
import axios from 'axios';
import { useNavigation } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, shadows, spacing } from '../src/theme';

const { width } = Dimensions.get('window');

interface DetectedItem {
    itemType: string;
    color: string;
    style: string;
    description: string;
    material?: string;
    details?: string;
    productDescription?: string;
    frameImage?: string;
}

interface AnalysisResult {
    detectedItems: DetectedItem[];
    frameImage?: string; // The frame used for detection
}

const WardrobeVideoScreen = () => {
    const navigation = useNavigation();
    const [analyzing, setAnalyzing] = useState(false);
    const [results, setResults] = useState<AnalysisResult | null>(null);
    const [progress, setProgress] = useState('');

    const API_URL = 'https://aiwardrobe-ivh4.onrender.com';

    const requestPermissions = async () => {
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
                // @ts-ignore
                mediaTypes: ['videos'],
                allowsEditing: false,
                quality: 1,
            });

            if (!result.canceled && result.assets[0]) {
                analyzeVideo(result.assets[0].uri);
            }
        } catch (error) {
            console.error('Error picking video:', error);
            Alert.alert('Error', 'Failed to pick video. Please try again.');
        }
    };

    const extractFrames = async (videoUri: string): Promise<string[]> => {
        const frames: string[] = [];
        const timePoints = [0, 2000, 4000, 6000, 8000]; // Extract at 0, 2, 4, 6, 8 seconds

        for (const time of timePoints) {
            try {
                setProgress(`Extracting frame ${frames.length + 1}/5...`);
                const { uri } = await VideoThumbnails.getThumbnailAsync(videoUri, {
                    time,
                    quality: 0.7,
                });

                // Convert to base64
                const base64 = await FileSystem.readAsStringAsync(uri, {
                    encoding: 'base64',
                });
                frames.push(base64);
            } catch (error) {
                console.log(`Failed to extract frame at ${time}ms:`, error);
            }
        }

        return frames;
    };

    // Real AI clothing detection via backend Gemini
    const analyzeClothingWithAI = async (frameBase64: string): Promise<DetectedItem[]> => {
        try {
            setProgress('🔍 AI analyzing clothing...');

            const response = await axios.post(
                `${API_URL}/api/analyze-frames`,
                { frames: [frameBase64] },
                { timeout: 60000 }
            );

            console.log('AI Response:', response.data);
            return response.data.detectedItems || [];
        } catch (error: any) {
            console.error('AI Analysis error:', error.message);
            throw error;
        }
    };


    // STEP 3: Generate AI product image via Replicate SDXL
    const generateProductImage = async (item: DetectedItem): Promise<string> => {
        try {
            setProgress(`🎨 AI generating image for ${item.itemType}...`);

            const response = await axios.post(
                `${API_URL}/api/generate-product-image`,
                {
                    itemType: item.itemType,
                    color: item.color,
                    description: item.description
                },
                { timeout: 90000 }  // 90 seconds for image generation
            );

            if (response.data.imageUrl) {
                return response.data.imageUrl;
            }
            throw new Error('No image URL returned');
        } catch (error: any) {
            console.log('AI image generation failed, using stock:', error.message);
            return getClothingImage(item.itemType, item.color);
        }
    };

    // Fallback stock images
    const getClothingImage = (itemType: string, color: string): string => {
        const type = itemType.toLowerCase();
        const clothingImages: { [key: string]: string } = {
            'jacket': 'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=500&fit=crop',
            'denim': 'https://images.unsplash.com/photo-1551028719-00167b16eac5?w=400&h=500&fit=crop',
            'shirt': 'https://images.unsplash.com/photo-1596755094514-f87e34085b2c?w=400&h=500&fit=crop',
            't-shirt': 'https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?w=400&h=500&fit=crop',
            'jeans': 'https://images.unsplash.com/photo-1542272454315-4c01d7abdf4a?w=400&h=500&fit=crop',
            'pants': 'https://images.unsplash.com/photo-1624378439575-d8705ad7ae80?w=400&h=500&fit=crop',
            'dress': 'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=400&h=500&fit=crop',
            'sweater': 'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=400&h=500&fit=crop',
            'hoodie': 'https://images.unsplash.com/photo-1556821840-3a63f95609a7?w=400&h=500&fit=crop',
            'coat': 'https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=400&h=500&fit=crop',
            'shoes': 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400&h=500&fit=crop',
            'sneakers': 'https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=400&h=500&fit=crop',
        };

        for (const [key, url] of Object.entries(clothingImages)) {
            if (type.includes(key)) return url;
        }
        return 'https://images.unsplash.com/photo-1489987707025-afc232f7ea0f?w=400&h=500&fit=crop';
    };

    const analyzeVideo = async (videoUri: string) => {
        setAnalyzing(true);
        setResults(null);
        setProgress('📹 Extracting frames from video...');

        try {
            // STEP 1: Extract frames from video
            const frames = await extractFrames(videoUri);

            if (frames.length === 0) {
                throw new Error('Could not extract any frames from video');
            }

            // STEP 2: AI analyzes frame and detects clothing with detailed descriptions
            let detectedItems = await analyzeClothingWithAI(frames[0]);

            if (detectedItems.length === 0) {
                // Try second frame
                if (frames.length > 1) {
                    setProgress('Trying another frame...');
                    detectedItems = await analyzeClothingWithAI(frames[1]);
                }
            }

            if (detectedItems.length === 0) {
                Alert.alert('No Clothing Found', 'AI could not detect clothing items in this video. Try a video with clear clothing visible.');
                setAnalyzing(false);
                return;
            }

            // STEP 3: Generate product images for each detected item
            setProgress(`🎨 Generating ${detectedItems.length} product image(s)...`);

            const itemsWithImages: DetectedItem[] = [];
            for (const item of detectedItems) {
                try {
                    const imageUrl = await generateProductImage(item);
                    itemsWithImages.push({
                        ...item,
                        frameImage: imageUrl,
                        description: item.productDescription || `${item.color} ${item.itemType}`
                    });
                } catch (e) {
                    // Use fallback image
                    itemsWithImages.push({
                        ...item,
                        frameImage: getClothingImage(item.itemType, item.color),
                        description: item.productDescription || `${item.color} ${item.itemType}`
                    });
                }
            }

            setResults({
                detectedItems: itemsWithImages,
                frameImage: `data:image/jpeg;base64,${frames[0]}`
            });
            setProgress('');

        } catch (error: any) {
            console.error('Analysis failed:', error);
            setProgress('');
            Alert.alert('Analysis Failed', error.message || 'Please try again');
        } finally {
            setAnalyzing(false);
        }
    };

    const saveToWardrobe = async () => {
        if (!results || results.detectedItems.length === 0) return;

        setProgress('💾 Saving to wardrobe...');

        try {
            const AsyncStorage = require('@react-native-async-storage/async-storage').default;

            // Get existing saved items from local storage
            const existingData = await AsyncStorage.getItem('myWardrobeItems');
            const existingItems = existingData ? JSON.parse(existingData) : [];

            // Create new items with AI-generated or fallback images
            const newItems = results.detectedItems.map((item: DetectedItem, index: number) => ({
                id: `item_${Date.now()}_${index}`,
                type: item.itemType,
                color: item.color,
                style: item.style,
                description: item.description || item.productDescription || `${item.color} ${item.itemType}`,
                material: item.material || 'Unknown',
                details: item.details || '',
                season: 'All Seasons',
                image: item.frameImage,  // Use AI-generated or fallback image
                source: 'video_scan',
                createdAt: new Date().toISOString()
            }));

            // Save all items
            const allItems = [...newItems, ...existingItems];
            await AsyncStorage.setItem('myWardrobeItems', JSON.stringify(allItems));

            console.log('✅ Saved', newItems.length, 'items locally!');

            Alert.alert(
                'Saved! 🎉',
                `${results.detectedItems.length} item(s) saved to your wardrobe!`,
                [{
                    text: 'View Wardrobe',
                    onPress: () => (navigation as any).navigate('Home', { screen: 'Profile' })
                },
                { text: 'OK' }]
            );
        } catch (error: any) {
            console.error('Save error:', error);
            Alert.alert('Error', 'Failed to save. Please try again.');
        } finally {
            setProgress('');
        }
    };

    return (
        <View style={styles.container}>
            <LinearGradient
                colors={['#ffffff', '#f0f4ff', '#e6eeff']}
                style={StyleSheet.absoluteFill}
            />
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
                                <View style={styles.stepItem}>
                                    <View style={styles.stepIconBg}>
                                        <Ionicons name="videocam-outline" size={24} color="#4f46e5" />
                                    </View>
                                    <Text style={styles.stepText}>Record Video</Text>
                                </View>
                                <View style={styles.stepLine} />
                                <View style={styles.stepItem}>
                                    <View style={styles.stepIconBg}>
                                        <Ionicons name="sparkles-outline" size={24} color="#4f46e5" />
                                    </View>
                                    <Text style={styles.stepText}>AI Analysis</Text>
                                </View>
                                <View style={styles.stepLine} />
                                <View style={styles.stepItem}>
                                    <View style={styles.stepIconBg}>
                                        <Ionicons name="shirt-outline" size={24} color="#4f46e5" />
                                    </View>
                                    <Text style={styles.stepText}>Get Items</Text>
                                </View>
                            </View>
                        </View>
                    )}

                    {/* Main Action Area */}
                    {!analyzing && !results && (
                        <TouchableOpacity
                            style={styles.uploadCard}
                            onPress={pickVideo}
                            activeOpacity={0.9}
                        >
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
                                    onPress={() => {
                                        setResults(null);
                                        pickVideo();
                                    }}
                                >
                                    <Ionicons name="refresh" size={20} color="#4f46e5" />
                                </TouchableOpacity>
                            </View>

                            {results.detectedItems.map((item: DetectedItem, index: number) => (
                                <View key={index} style={styles.resultCard}>
                                    <View style={styles.resultIcon}>
                                        <Ionicons name="shirt" size={24} color="#4f46e5" />
                                    </View>
                                    <View style={styles.resultInfo}>
                                        <Text style={styles.resultType}>{item.itemType}</Text>
                                        <Text style={styles.resultDetails}>
                                            {item.color} • {item.style}
                                        </Text>
                                    </View>
                                    <View style={styles.checkIcon}>
                                        <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                                    </View>
                                </View>
                            ))}

                            <TouchableOpacity
                                style={styles.saveButton}
                                onPress={saveToWardrobe}
                            >
                                <LinearGradient
                                    colors={['#1a1a1a', '#000000']}
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
    container: {
        flex: 1,
        backgroundColor: '#fff',
    },
    safeArea: {
        flex: 1,
    },
    header: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        paddingHorizontal: 20,
        paddingVertical: 16,
    },
    backButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 8,
        elevation: 2,
    },
    headerTitle: {
        fontSize: 18,
        fontWeight: '700',
        color: '#1a1a1a',
        letterSpacing: 0.5,
    },
    scrollContent: {
        padding: 24,
        paddingBottom: 40,
    },
    heroSection: {
        marginBottom: 32,
        alignItems: 'center',
    },
    heroTitle: {
        fontSize: 28,
        fontWeight: '800',
        color: '#1a1a1a',
        marginBottom: 8,
        textAlign: 'center',
    },
    heroSubtitle: {
        fontSize: 16,
        color: '#666',
        textAlign: 'center',
        lineHeight: 24,
        marginBottom: 32,
    },
    stepsContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        width: '100%',
    },
    stepItem: {
        alignItems: 'center',
    },
    stepIconBg: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: '#eef2ff',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 8,
    },
    stepText: {
        fontSize: 12,
        fontWeight: '600',
        color: '#4f46e5',
    },
    stepLine: {
        width: 30,
        height: 2,
        backgroundColor: '#e0e7ff',
        marginHorizontal: 8,
        marginBottom: 20,
    },
    uploadCard: {
        width: '100%',
        height: 200,
        borderRadius: 24,
        shadowColor: '#4f46e5',
        shadowOffset: { width: 0, height: 10 },
        shadowOpacity: 0.2,
        shadowRadius: 20,
        elevation: 10,
    },
    uploadGradient: {
        flex: 1,
        borderRadius: 24,
        alignItems: 'center',
        justifyContent: 'center',
        padding: 20,
    },
    uploadIconContainer: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: 'rgba(255,255,255,0.2)',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 16,
        borderWidth: 1,
        borderColor: 'rgba(255,255,255,0.3)',
    },
    uploadTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: '#fff',
        marginBottom: 4,
    },
    uploadSubtitle: {
        fontSize: 14,
        color: 'rgba(255,255,255,0.8)',
    },
    loadingContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 40,
    },
    loadingCircle: {
        width: 80,
        height: 80,
        borderRadius: 40,
        backgroundColor: '#fff',
        alignItems: 'center',
        justifyContent: 'center',
        marginBottom: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.1,
        shadowRadius: 12,
        elevation: 5,
    },
    loadingText: {
        fontSize: 18,
        fontWeight: '700',
        color: '#1a1a1a',
        marginBottom: 8,
    },
    loadingSubtext: {
        fontSize: 14,
        color: '#666',
        textAlign: 'center',
    },
    resultsContainer: {
        width: '100%',
    },
    resultsHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: 20,
    },
    resultsTitle: {
        fontSize: 20,
        fontWeight: '700',
        color: '#1a1a1a',
    },
    resultsSubtitle: {
        fontSize: 14,
        color: '#666',
    },
    retryButton: {
        width: 40,
        height: 40,
        borderRadius: 20,
        backgroundColor: '#eef2ff',
        alignItems: 'center',
        justifyContent: 'center',
    },
    resultCard: {
        flexDirection: 'row',
        alignItems: 'center',
        backgroundColor: '#fff',
        padding: 16,
        borderRadius: 16,
        marginBottom: 12,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.05,
        shadowRadius: 8,
        elevation: 2,
        borderWidth: 1,
        borderColor: '#f0f0f0',
    },
    resultIcon: {
        width: 48,
        height: 48,
        borderRadius: 24,
        backgroundColor: '#eef2ff',
        alignItems: 'center',
        justifyContent: 'center',
        marginRight: 16,
    },
    resultInfo: {
        flex: 1,
    },
    resultType: {
        fontSize: 16,
        fontWeight: '600',
        color: '#1a1a1a',
        marginBottom: 2,
        textTransform: 'capitalize',
    },
    resultDetails: {
        fontSize: 12,
        color: '#666',
    },
    checkIcon: {
        marginLeft: 12,
    },
    saveButton: {
        marginTop: 24,
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.2,
        shadowRadius: 12,
        elevation: 8,
    },
    saveButtonGradient: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        paddingVertical: 18,
        borderRadius: 16,
    },
    saveButtonText: {
        fontSize: 16,
        fontWeight: '700',
        color: '#fff',
        marginRight: 8,
    },
});

export default WardrobeVideoScreen;
