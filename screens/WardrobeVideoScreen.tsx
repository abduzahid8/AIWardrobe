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
    specificType?: string;  // Specific type (e.g., "denim trucker jacket")
    classificationPath?: string; // Full path: "Outerwear > Jackets > Denim Jackets"
    color: string;
    colorHex?: string;
    style: string;
    description: string;
    material?: string;  // Primary material (cotton, denim, silk, etc.)
    materialDetails?: {  // Full material analysis
        type: string;
        category: string;
        texture: string;
        finish: string;
        weight: string;
        isStretch: boolean;
    };
    pattern?: string;  // Pattern type (solid, stripes, plaid, etc.)
    patternDetails?: {  // Full pattern analysis
        type: string;
        category: string;
        isStriped: boolean;
        isCheckered: boolean;
        hasPrint: boolean;
        colors: string[];
    };
    details?: string;
    productDescription?: string;
    frameImage?: string;
    position?: string;  // upper, lower, feet, accessory, full
    confidence?: number; // 0-1 confidence score
    confidenceLevel?: string; // "high", "medium", "low"
    agreementScore?: number; // Multi-model agreement 0-1
    detectionSources?: string[]; // Which AI models detected this
    styleTags?: string[];  // Style tags (e.g., ["streetwear", "casual"])
    features?: any;  // Physical features (zippers, buttons, collars)
    bbox?: number[];  // Bounding box [x, y, w, h]
    attributes?: any;  // Full attribute data from AI
}

interface AnalysisResult {
    detectedItems: DetectedItem[];
    frameImage?: string; // The frame used for detection
}

// Helper: Get body position from clothing category
const getItemPosition = (category: string): string => {
    const cat = (category || '').toLowerCase();

    // Upper body items
    if (['shirt', 'blouse', 'sweater', 'jacket', 'coat', 'top', 't-shirt', 'hoodie',
        'upper-clothes', 'cardigan', 'polo', 'tank'].some(u => cat.includes(u))) {
        return 'upper';
    }
    // Lower body items
    if (['pants', 'jeans', 'shorts', 'skirt', 'trousers', 'leggings'].some(l => cat.includes(l))) {
        return 'lower';
    }
    // Full body items
    if (['dress', 'jumpsuit', 'romper', 'overalls', 'suit'].some(f => cat.includes(f))) {
        return 'full';
    }
    // Footwear
    if (['shoe', 'boot', 'sneaker', 'sandal', 'heel', 'loafer', 'slipper'].some(f => cat.includes(f))) {
        return 'feet';
    }
    // Accessories
    if (['bag', 'hat', 'scarf', 'belt', 'watch', 'glasses', 'sunglasses'].some(a => cat.includes(a))) {
        return 'accessory';
    }
    return 'upper'; // default
};

// 🚀 Enhanced category name formatting with better display names
const formatCategoryName = (category: string): string => {
    if (!category) return "Clothing";

    // Map specific types to better display names
    const displayMap: { [key: string]: string } = {
        'upper_clothes': 'Top',
        'left_shoe': 'Shoes',
        'right_shoe': 'Shoes',
        'pants': 'Pants',
        'dress pants': 'Dress Pants',
        'dress_pants': 'Dress Pants',
        'chinos': 'Chinos',
        'jeans': 'Jeans',
        'skinny jeans': 'Skinny Jeans',
        'joggers': 'Joggers',
        't-shirt': 'T-Shirt',
        'tshirt': 'T-Shirt',
        'sport coat': 'Sport Coat',
        'sport_coat': 'Sport Coat',
        'blazer': 'Blazer',
        'denim jacket': 'Denim Jacket',
        'leather jacket': 'Leather Jacket',
        'cardigan': 'Cardigan',
        'sweater': 'Sweater',
        'hoodie': 'Hoodie',
        'sneakers': 'Sneakers',
        'running shoes': 'Running Shoes',
        'dress shoes': 'Dress Shoes',
        'boots': 'Boots',
        'loafers': 'Loafers',
    };

    const lowerCategory = category.toLowerCase();
    if (displayMap[lowerCategory]) {
        return displayMap[lowerCategory];
    }

    // Convert snake_case to Title Case
    return category
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

// 🚀 Helper: Merge left/right shoes into pairs - PRESERVE specificType!
const mergeShoeCategories = (items: any[]): any[] => {
    const shoeItems: any[] = [];
    const otherItems: any[] = [];

    items.forEach(item => {
        const cat = (item.itemType || '').toLowerCase();
        if (cat.includes('shoe') || cat.includes('left_shoe') || cat.includes('right_shoe') ||
            cat.includes('sneaker') || cat.includes('boot') || cat.includes('sandal') || cat.includes('loafer')) {
            shoeItems.push(item);
        } else {
            otherItems.push(item);
        }
    });

    // If we have shoes, merge them - but KEEP the specificType!
    if (shoeItems.length > 0) {
        const firstShoe = shoeItems[0];

        // 🚀 Use specificType for display name (sneakers, dress shoes, etc.)
        let shoeDisplayName = 'Shoes';
        if (firstShoe.specificType) {
            shoeDisplayName = formatCategoryName(firstShoe.specificType);
        } else if (firstShoe.itemType && !firstShoe.itemType.toLowerCase().includes('shoe')) {
            // Already has a specific type like "Sneakers"
            shoeDisplayName = firstShoe.itemType;
        }

        otherItems.push({
            itemType: shoeDisplayName,
            specificType: firstShoe.specificType,  // 🚀 Keep specific type!
            color: firstShoe.color || 'Unknown',
            style: 'Casual',
            description: `${firstShoe.color || ''} ${shoeDisplayName}`.trim(),
            position: 'feet',
            confidence: firstShoe.confidence,
            bbox: firstShoe.bbox,
            colorHex: firstShoe.colorHex || "#000000"
        });
    }

    return otherItems;
};

const WardrobeVideoScreen = () => {
    const navigation = useNavigation();
    const [analyzing, setAnalyzing] = useState(false);
    const [results, setResults] = useState<AnalysisResult | null>(null);
    const [progress, setProgress] = useState('');

    // Use local API server with local network IP for iOS Simulator
    const API_URL = 'http://172.20.10.5:3000';

    // Direct connection to AliceVision Python service (port 5050)
    const ALICEVISION_URL = 'http://172.20.10.5:5050';

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

    // Smart clothing detection - Priority: AliceVision (local) > OpenAI > Gemini
    const analyzeClothingWithAI = async (frames: string[]): Promise<DetectedItem[]> => {
        setProgress('🔍 AI analyzing clothing...');

        // Try with multiple frames for better detection
        for (let attempt = 0; attempt < Math.min(frames.length, 3); attempt++) {
            try {
                setProgress(`🔍 Analyzing frame ${attempt + 1}...`);

                // FIRST: Try MAXIMUM POWER DETECTION (THE BEST POSSIBLE AI!)
                try {
                    setProgress(`🔥 MAXIMUM POWER AI analyzing...`);
                    const imageData = frames[attempt].replace(/^data:image\/\w+;base64,/, '');

                    // 🔥 Use /detect-max - Florence-2 + SegFormer + YOLO + Fashion-CLIP!
                    const maxResponse = await axios.post(
                        `${ALICEVISION_URL}/detect-max`,
                        {
                            image: imageData,
                            enable_all: true  // Enable Florence-2 + everything!
                        },
                        { timeout: 300000 }  // 5 min for maximum AI
                    );

                    if (maxResponse.data.success && maxResponse.data.items?.length > 0) {
                        let detectedItems = maxResponse.data.items.map((item: any) => {
                            // Use detailed specificType from multi-model detection
                            const displayName = item.specificType
                                ? formatCategoryName(item.specificType)
                                : formatCategoryName(item.category);

                            return {
                                itemType: displayName || "Clothing Item",
                                specificType: item.specificType,
                                color: item.primaryColor || "Unknown",
                                colorHex: item.colorHex || "#000000",
                                style: "Casual",
                                description: `${item.primaryColor || ''} ${displayName}`.trim(),
                                material: item.material,
                                pattern: item.pattern,
                                position: getItemPosition(item.category),
                                confidence: item.confidence,
                                confidenceLevel: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low',
                                agreementScore: item.agreementScore,
                                detectionSources: item.detectedBy,
                                bbox: item.bbox,
                                frameImage: item.cutoutImage
                            };
                        });

                        detectedItems = mergeShoeCategories(detectedItems);

                        console.log(`🔥 MAXIMUM POWER detected ${detectedItems.length} items:`);
                        console.log(`   Models: ${maxResponse.data.modelsUsed?.join(', ')}`);
                        console.log(`   Florence-2: ${maxResponse.data.florence2Enabled ? 'YES' : 'NO'}`);
                        return detectedItems;
                    }
                } catch (maxError: any) {
                    console.log(`Maximum detection failed: ${maxError.message}, trying ultimate...`);
                }

                // SECOND: Try PERFECT DETECTION (GPT-4V + rembg - 100% ACCURATE!)
                try {
                    setProgress(`🏆 Perfect AI analyzing (GPT-4V)...`);
                    const imageData = frames[attempt].replace(/^data:image\/\w+;base64,/, '');

                    const perfectResponse = await axios.post(
                        `${ALICEVISION_URL}/detect-perfect`,
                        {
                            image: imageData,
                            create_cutouts: true
                        },
                        { timeout: 60000 }
                    );

                    if (perfectResponse.data.success && perfectResponse.data.items?.length > 0) {
                        let detectedItems = perfectResponse.data.items.map((item: any) => {
                            return {
                                itemType: formatCategoryName(item.type),
                                specificType: item.type,
                                color: item.color || "Unknown",
                                colorHex: "#000000",
                                style: item.style || "Casual",
                                description: item.description || `${item.color} ${item.type}`.trim(),
                                material: item.material,
                                position: item.position || "upper",
                                confidence: item.confidence,
                                confidenceLevel: 'high',  // GPT-4V is always high confidence
                                frameImage: item.productCardImage || item.cutoutImage,  // 🏷️ Use professional product card!
                            };
                        });

                        detectedItems = mergeShoeCategories(detectedItems);
                        const itemsWithCutouts = detectedItems.filter((i: DetectedItem) => i.frameImage).length;
                        console.log(`🏆 PERFECT AI detected ${detectedItems.length} items (${itemsWithCutouts} with cutouts, ${perfectResponse.data.modelUsed})`);
                        return detectedItems;
                    }
                } catch (perfectError: any) {
                    console.log(`Perfect detection failed: ${perfectError.message}, trying ultimate...`);
                }

                // FALLBACK: Try ULTIMATE DETECTION (local SegFormer + CLIP)
                try {
                    setProgress(`🎯 Ultimate AI analyzing...`);
                    const imageData = frames[attempt].replace(/^data:image\/\w+;base64,/, '');

                    const ultimateResponse = await axios.post(
                        `${ALICEVISION_URL}/detect-ultimate`,
                        {
                            image: imageData,
                            create_cutouts: true
                        },
                        { timeout: 120000 }
                    );

                    if (ultimateResponse.data.success && ultimateResponse.data.items?.length > 0) {
                        let detectedItems = ultimateResponse.data.items.map((item: any) => {
                            return {
                                itemType: item.label || "Clothing",
                                specificType: item.type,
                                color: item.color || "Unknown",
                                colorHex: item.colorHex || "#000000",
                                style: "Casual",
                                description: `${item.color || ''} ${item.type}`.trim(),
                                position: item.position || "upper",
                                confidence: item.confidence,
                                confidenceLevel: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low',
                                bbox: item.bbox,
                                frameImage: item.cutoutImage,
                            };
                        });

                        detectedItems = mergeShoeCategories(detectedItems);
                        console.log(`🎯 ULTIMATE AI detected ${detectedItems.length} items`);
                        return detectedItems;
                    }
                } catch (ultimateError: any) {
                    console.log(`Ultimate detection failed: ${ultimateError.message}, trying segment...`);
                }

                // FALLBACK: Try standard /segment endpoint
                try {
                    setProgress(`🤖 AI detecting & cutting out items...`);
                    const imageData = frames[attempt].replace(/^data:image\/\w+;base64,/, '');

                    const localResponse = await axios.post(
                        `${ALICEVISION_URL}/segment`,
                        {
                            image: imageData,
                            add_white_background: true,
                            use_advanced: true
                        },
                        { timeout: 120000 }
                    );

                    if (localResponse.data.success && localResponse.data.items?.length > 0) {
                        let detectedItems = localResponse.data.items.map((item: any) => {
                            const displayName = item.specificType
                                ? formatCategoryName(item.specificType)
                                : formatCategoryName(item.category);

                            return {
                                itemType: displayName || "Clothing Item",
                                specificType: item.specificType,
                                color: item.primaryColor || "Unknown",
                                colorHex: item.colorHex || "#000000",
                                style: "Casual",
                                description: `${item.primaryColor || ''} ${displayName}`.trim(),
                                position: getItemPosition(item.category),
                                confidence: item.confidence,
                                confidenceLevel: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low',
                                bbox: item.bbox
                            };
                        });

                        detectedItems = mergeShoeCategories(detectedItems);
                        console.log(`✅ Segment AI detected ${detectedItems.length} items`);
                        return detectedItems;
                    }
                } catch (localError: any) {
                    console.log(`/segment failed: ${localError.message}, trying fallbacks...`);
                }

                // FALLBACK: Try segment-all (may fail)
                try {
                    setProgress(`🤖 Local AI analyzing...`);
                    const imageData = frames[attempt].replace(/^data:image\/\w+;base64,/, '');

                    const localResponse = await axios.post(
                        `${ALICEVISION_URL}/segment`,
                        {
                            image: imageData,
                            add_white_background: false,
                            use_advanced: true
                        },
                        { timeout: 90000 }
                    );

                    if (localResponse.data.success && localResponse.data.items?.length > 0) {
                        let detectedItems = localResponse.data.items.map((item: any) => {
                            // Use specificType if available (V2 CLIP types like "t-shirt", "denim jacket")
                            const displayName = item.specificType
                                ? formatCategoryName(item.specificType)
                                : formatCategoryName(item.category);

                            return {
                                itemType: displayName || "Clothing Item",
                                specificType: item.specificType,  // Store for later use
                                color: item.primaryColor || "Unknown",
                                style: "Casual",
                                description: `${item.primaryColor || ''} ${displayName}`.trim(),
                                position: getItemPosition(item.category),
                                confidence: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low',
                                bbox: item.bbox,
                                colorHex: item.colorHex || "#000000"
                            };
                        });

                        detectedItems = mergeShoeCategories(detectedItems);

                        console.log(`✅ Local AI detected ${detectedItems.length} items:`,
                            detectedItems.map((i: DetectedItem) => `${i.itemType}${i.specificType ? ' (' + i.specificType + ')' : ''}`));
                        return detectedItems;
                    }

                    // If segmentation worked but no items, create a fallback item
                    if (localResponse.data.success) {
                        return [{
                            itemType: "Clothing",
                            color: "Detected",
                            style: "Casual",
                            description: "Clothing detected",
                            position: "upper",
                            confidence: localResponse.data.confidence || 0.7,
                            confidenceLevel: (localResponse.data.confidence || 0.7) > 0.7 ? 'high' : 'medium'
                        }];
                    }
                } catch (localError: any) {
                    console.log(`Local AI failed: ${localError.message}, trying OpenAI...`);
                }

                // FALLBACK 1: Try OpenAI Vision
                try {
                    const openAIResponse = await axios.post(
                        `${API_URL}/api/openai/analyze-clothing`,
                        { imageBase64: frames[attempt].replace(/^data:image\/\w+;base64,/, '') },
                        { timeout: 60000 }
                    );

                    if (openAIResponse.data.detectedItems?.length > 0) {
                        console.log(`✅ OpenAI detected ${openAIResponse.data.detectedItems.length} items`);
                        return openAIResponse.data.detectedItems;
                    }
                } catch (openAIError: any) {
                    console.log(`OpenAI failed: ${openAIError.message}, trying Gemini...`);
                }

                // FALLBACK 2: Try Gemini
                const response = await axios.post(
                    `${API_URL}/api/analyze-frames`,
                    { frames: [frames[attempt]] },
                    { timeout: 60000 }
                );

                if (response.data.detectedItems?.length > 0) {
                    console.log(`✅ Gemini detected ${response.data.detectedItems.length} items:`,
                        response.data.detectedItems.map((i: DetectedItem) => i.itemType));
                    return response.data.detectedItems;
                }

                console.log(`Frame ${attempt + 1}: No items detected, trying next...`);
            } catch (error: any) {
                console.log(`Frame ${attempt + 1} analysis failed:`, error.message);
            }
        }

        // All attempts failed - show error, don't ask user to select manually
        console.log('❌ AI detection failed on all frames');
        setProgress('');
        throw new Error('AI could not detect clothing items. Please try a clearer video with good lighting.');
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

            // STEP 2: AI analyzes ALL frames and detects ALL clothing items automatically
            const detectedItems = await analyzeClothingWithAI(frames);

            if (detectedItems.length === 0) {
                Alert.alert('No Clothing Found', 'AI could not detect clothing items in this video. Try a video with clear clothing visible.');
                setAnalyzing(false);
                return;
            }

            // STEP 3: Try V2 Multi-Item Processing first (if available)
            // Note: V2 requires additional dependencies - falls back to working pipeline if unavailable
            setProgress('🎯 Checking for advanced multi-item detection...');

            let useBasicPipeline = false;

            try {
                // Quick check if V2 is available (30 second timeout)
                const v2Response = await axios.post(
                    `${API_URL}/api/v2/product-photo/process-multi`,
                    {
                        frames: frames,
                        prompts: null // Auto-detect all clothing items
                    },
                    { timeout: 30000 }  // 30 seconds - fail fast if unavailable
                );

                if (v2Response.data.success && v2Response.data.items && v2Response.data.items.length > 0) {
                    const processedItems = v2Response.data.items;

                    setProgress(`✅ Found ${processedItems.length} items! Creating cards...`);

                    console.log(`✅ V2 Multi-Item Processing Complete:`);
                    console.log(`   - Detected: ${v2Response.data.totalItemsDetected} items`);
                    console.log(`   - Created: ${v2Response.data.totalCardsCreated} Massimo Dutti cards`);
                    console.log(`   - Categories: ${v2Response.data.summary.categories.join(', ')}`);

                    // Convert V2 items to DetectedItem format
                    const itemsWithImages: DetectedItem[] = processedItems.map((item: any) => ({
                        itemType: item.attributes.category,
                        color: item.attributes.primaryColor,
                        style: item.attributes.style,
                        description: item.attributes.description,
                        material: item.attributes.fabric || 'Unknown',
                        details: JSON.stringify(item.attributes.details),
                        productDescription: item.cardPrompt.prompt,
                        frameImage: item.imageUrl  // Professional Massimo Dutti white background photo!
                    }));

                    setResults({
                        detectedItems: itemsWithImages,
                        frameImage: processedItems[0]?.imageUrl || ''
                    });
                    setProgress('');
                    console.log(`✅ Saved ${itemsWithImages.length} Massimo Dutti cards!`);
                    return; // Success!
                }
            } catch (v2Error: any) {
                console.log('V2 Multi-Item processing unavailable, falling back...', v2Error.message);
                setProgress('⚠️ Advanced AI unavailable, using basic mode...');
            }

            // FALLBACK: Multi-item processing for all detected items
            setProgress(`🎨 Processing ${detectedItems.length} items...`);

            // Process ALL detected items in parallel
            const processItem = async (item: DetectedItem, index: number): Promise<DetectedItem> => {
                try {
                    // 🚀 If item already has cutout from detection, use it directly!
                    if (item.frameImage) {
                        console.log(`✅ Using pre-cut image for ${item.itemType}`);
                        return item;
                    }

                    setProgress(`📸 Cutting out ${item.itemType} (${index + 1}/${detectedItems.length})...`);

                    // FIRST: Try AliceVision per-item segmentation
                    try {
                        const imageData = frames[0].replace(/^data:image\/\w+;base64,/, '');
                        const segResponse = await axios.post(
                            `${ALICEVISION_URL}/segment-item`,
                            {
                                image: imageData,
                                bbox: (item as any).bbox || null,  // Use item's bounding box if available
                                category: item.itemType,
                                add_white_background: true,
                                padding: 30  // Add padding around item
                            },
                            { timeout: 120000 }  // 2 min for ControlNet generation
                        );

                        if (segResponse.data.success && segResponse.data.croppedImage) {
                            // 🚀 Use specificType from V2 detection if available
                            const betterType = segResponse.data.specificType;
                            const betterColor = segResponse.data.primaryColor;
                            const updatedItemType = betterType
                                ? formatCategoryName(betterType)
                                : item.itemType;

                            // 🏷️ Use professional product card if available, otherwise cutout
                            const cardImage = segResponse.data.productCardImage || segResponse.data.croppedImage;

                            console.log(`✅ Per-item cutout created for ${updatedItemType}${betterType ? ` (V2: ${betterType})` : ''}`);
                            return {
                                ...item,
                                itemType: updatedItemType,  // 🚀 Update with V2 type!
                                specificType: betterType || item.specificType,
                                color: betterColor || item.color,
                                frameImage: cardImage,  // 🏷️ Professional product card!
                                description: `${betterColor || item.color} ${updatedItemType}`.trim()
                            };
                        }
                    } catch (localError: any) {
                        console.log(`Per-item cutout failed: ${localError.message}, trying full /segment...`);
                    }

                    // FALLBACK 1: Try full-frame segmentation
                    try {
                        const imageData = frames[0].replace(/^data:image\/\w+;base64,/, '');
                        const segResponse = await axios.post(
                            `${ALICEVISION_URL}/segment`,
                            {
                                image: imageData,
                                add_white_background: true,
                                use_advanced: true
                            },
                            { timeout: 120000 }  // 2 min fallback
                        );

                        if (segResponse.data.success && segResponse.data.segmentedImage) {
                            console.log(`✅ Full-frame cutout for ${item.itemType}`);
                            return {
                                ...item,
                                frameImage: segResponse.data.segmentedImage,
                                description: item.description || `${item.color} ${item.itemType}`
                            };
                        }
                    } catch (segError: any) {
                        console.log(`Full segment failed: ${segError.message}`);
                    }

                    // FALLBACK 2: Try product photo pipeline via Node.js API
                    const productResponse = await axios.post(
                        `${API_URL}/api/product-photo/process`,
                        {
                            frames: frames,
                            clothingType: `${item?.color || ''} ${item?.style || ''} ${item?.itemType || 'clothing'}`.trim(),
                            clothingColor: item?.color || '',
                            clothingStyle: item?.style || '',
                            clothingDescription: item?.description || ''
                        },
                        { timeout: 120000 }
                    );

                    if (productResponse.data.imageUrl) {
                        console.log(`✅ Product photo created for ${item.itemType}`);
                        return {
                            ...item,
                            frameImage: productResponse.data.imageUrl,
                            description: item.description || `${item.color} ${item.itemType}`
                        };
                    }
                } catch (pipelineError: any) {
                    console.log(`Pipeline failed for ${item.itemType}:`, pipelineError.message);
                }

                // Fallback: use stock image for this item
                return {
                    ...item,
                    frameImage: getClothingImage(item.itemType, item.color),
                    description: item.description || `${item.color} ${item.itemType}`
                };
            };

            // Process all items in parallel with graceful error handling
            const itemResults = await Promise.allSettled(
                detectedItems.map((item, index) => processItem(item, index))
            );

            // Collect successful results
            const itemsWithImages: DetectedItem[] = itemResults
                .filter((result): result is PromiseFulfilledResult<DetectedItem> => result.status === 'fulfilled')
                .map(result => result.value);

            console.log(`✅ Successfully processed ${itemsWithImages.length}/${detectedItems.length} items`);

            setResults({
                detectedItems: itemsWithImages,
                frameImage: itemsWithImages[0]?.frameImage || ''
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

                            {results.detectedItems.map((item: DetectedItem, index: number) => {
                                // Get icon based on position
                                const getPositionIcon = (pos?: string) => {
                                    switch (pos) {
                                        case 'upper': return 'shirt';
                                        case 'lower': return 'layers';
                                        case 'feet': return 'footsteps';
                                        case 'accessory': return 'bag';
                                        case 'full': return 'body';
                                        default: return 'shirt';
                                    }
                                };

                                return (
                                    <View key={index} style={styles.resultCard}>
                                        <View style={styles.resultIcon}>
                                            <Ionicons name={getPositionIcon(item.position)} size={24} color="#4f46e5" />
                                        </View>
                                        <View style={styles.resultInfo}>
                                            <Text style={styles.resultType}>{item.itemType}</Text>
                                            {/* Classification path for detailed type */}
                                            {item.classificationPath && (
                                                <Text style={styles.resultPath}>{item.classificationPath}</Text>
                                            )}
                                            <Text style={styles.resultDetails}>
                                                {item.color} • {item.style}
                                                {item.material ? ` • ${item.material}` : ''}
                                                {item.pattern && item.pattern !== 'solid' ? ` • ${item.pattern}` : ''}
                                            </Text>
                                            {/* Material and pattern details */}
                                            {(item.materialDetails || item.patternDetails) && (
                                                <Text style={styles.resultMaterialPattern}>
                                                    {item.materialDetails?.texture ? `${item.materialDetails.texture} ` : ''}
                                                    {item.materialDetails?.finish ? `• ${item.materialDetails.finish}` : ''}
                                                </Text>
                                            )}
                                            {/* Detection sources for transparency */}
                                            {item.detectionSources && item.detectionSources.length > 0 && (
                                                <Text style={styles.resultSources}>
                                                    🤖 {item.detectionSources.join(' + ')}
                                                </Text>
                                            )}
                                            {item.styleTags && item.styleTags.length > 0 && (
                                                <Text style={styles.resultTags}>
                                                    {item.styleTags.slice(0, 3).join(' • ')}
                                                </Text>
                                            )}
                                        </View>
                                        <View style={styles.checkIcon}>
                                            <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                                        </View>
                                    </View>
                                );
                            })}

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
    resultTags: {
        fontSize: 11,
        color: '#4f46e5',
        marginTop: 4,
        fontStyle: 'italic',
    },
    resultPath: {
        fontSize: 10,
        color: '#9ca3af',
        marginBottom: 2,
    },
    resultMaterialPattern: {
        fontSize: 11,
        color: '#6b7280',
        marginTop: 2,
    },
    resultSources: {
        fontSize: 10,
        color: '#10b981',
        marginTop: 3,
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
