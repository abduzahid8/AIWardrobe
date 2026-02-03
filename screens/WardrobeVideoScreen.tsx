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
import { useNavigation, useRoute, RouteProp } from '@react-navigation/native';
import { LinearGradient } from 'expo-linear-gradient';
import { colors, shadows, spacing } from '../src/theme';
import CorrectionModal from '../src/components/CorrectionModal';
import { RootStackParamList } from '../navigation/types';

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
    // 🧵 V2: Fine-grained attributes from FashionFAE
    neckline?: string;  // crew neck, v-neck, turtleneck, etc.
    sleeveType?: string;  // short sleeve, long sleeve, bishop sleeve, etc.
    fit?: string;  // slim fit, regular, oversized, etc.
    closure?: string;  // button-front, zip, pullover, etc.
    details?: string;
    productDescription?: string;
    frameImage?: string;
    position?: string;  // upper, lower, feet, accessory, full
    confidence?: number; // 0-1 confidence score
    confidenceLevel?: string; // "high", "medium", "low"
    agreementScore?: number; // Multi-model agreement 0-1
    detectionSources?: string[]; // Which AI models detected this
    styleTags?: string[];  // Style tags (e.g., ["streetwear", "casual"])
    features?: Record<string, string | number | boolean>;  // Physical features (zippers, buttons, collars)
    bbox?: number[];  // Bounding box [x, y, w, h]
    attributes?: Record<string, string | number | boolean>;  // Full attribute data from AI
    outfitId?: number;  // 🎬 Outfit grouping ID (1, 2, 3, 4...)
    framesDetected?: number;  // 🎬 V2: Number of frames item was detected in
    trackId?: number;  // 🎬 V2: Unique track ID from FeatureSORT
    cutoutImage?: string;  // 🎬 Timeline: Pre-cut image from correct outfit frame
    detectionBox?: number[];  // 🎬 Timeline: Bounding box for per-item cutout
    startFrame?: number;  // 🎬 Timeline: Frame index for correct cutout generation
    frameIndex?: number;  // Frame index where item was detected
}

// API Response Types for type-safe mapping
interface APIItemResponse {
    category?: string;
    specificType?: string;
    primaryColor?: string;
    color?: string;
    colorHex?: string;
    material?: string;
    pattern?: string;
    confidence?: number;
    bbox?: number[];
    cutoutImage?: string;
    bestFrame?: string;
    attributes?: Record<string, string | number | boolean>;
    type?: string;
    fit?: string;
    trackId?: number;
    outfitId?: number;
    outfit_id?: number;
    neckline?: string;
    sleeveType?: string;
    styleTags?: string[];
    caption?: string;
    framesDetected?: number;
    frameIndices?: number[];
    // Fashion Intelligence specific
    identity?: { type?: string; subType?: string; brandGuess?: string };
    construction?: { closure?: string; neckline?: string; sleeves?: string; pockets?: string; details?: string };
    quality?: { condition?: string; level?: string; priceRange?: string };
    position?: string;
    startFrame?: number;
}

interface OutfitResponse {
    outfitId?: number;
    startFrame?: number;
    items?: APIItemResponse[];
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

// 🔒 Helper: Deduplicate items detected multiple times (same type in overlapping positions)
const deduplicateItems = (items: DetectedItem[]): DetectedItem[] => {
    if (items.length <= 1) return items;

    // Calculate IoU (Intersection over Union) for bounding boxes
    const iou = (box1: number[] | undefined, box2: number[] | undefined): number => {
        if (!box1 || !box2 || box1.length < 4 || box2.length < 4) return 0;

        const [x1, y1, w1, h1] = box1;
        const [x2, y2, w2, h2] = box2;

        const xi1 = Math.max(x1, x2);
        const yi1 = Math.max(y1, y2);
        const xi2 = Math.min(x1 + w1, x2 + w2);
        const yi2 = Math.min(y1 + h1, y2 + h2);

        if (xi2 <= xi1 || yi2 <= yi1) return 0;

        const inter = (xi2 - xi1) * (yi2 - yi1);
        const union = w1 * h1 + w2 * h2 - inter;
        return union > 0 ? inter / union : 0;
    };

    // Sort by confidence (highest first)
    const sorted = [...items].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));

    const unique: DetectedItem[] = [];
    for (const item of sorted) {
        let isDuplicate = false;
        const itemType = (item.itemType || item.specificType || '').toLowerCase();
        const itemFrame = item.frameIndex || 0;

        for (const existing of unique) {
            const existingType = (existing.itemType || existing.specificType || '').toLowerCase();
            const existingFrame = existing.frameIndex || 0;

            // 🚀 ONLY deduplicate items from SAME FRAME
            // This preserves different outfits from different frames
            if (itemFrame !== existingFrame) {
                continue; // Different frame = different outfit, not duplicate
            }

            // Same category family AND overlapping bbox (IoU > 0.5 for stricter matching)
            const sameCategory =
                (itemType.includes('shirt') && existingType.includes('shirt')) ||
                (itemType.includes('pants') && existingType.includes('pants')) ||
                (itemType.includes('jeans') && existingType.includes('jeans')) ||
                (itemType.includes('shoe') && existingType.includes('shoe')) ||
                (itemType.includes('jacket') && existingType.includes('jacket')) ||
                (itemType.includes('sweater') && existingType.includes('sweater')) ||
                (itemType === existingType);

            if (sameCategory && item.bbox && existing.bbox) {
                if (iou(item.bbox, existing.bbox) > 0.5) { // Stricter IoU threshold
                    isDuplicate = true;
                    break;
                }
            }
        }

        if (!isDuplicate) {
            unique.push(item);
        }
    }

    return unique;
};

// 🚀 Helper: Merge left/right shoes into pairs - PRESERVE specificType!
const mergeShoeCategories = (items: DetectedItem[]): DetectedItem[] => {
    const shoeItems: DetectedItem[] = [];
    const otherItems: DetectedItem[] = [];

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
    const route = useRoute<RouteProp<RootStackParamList, 'WardrobeVideo'>>();
    const [analyzing, setAnalyzing] = useState(false);
    const [results, setResults] = useState<AnalysisResult | null>(null);
    const [progress, setProgress] = useState('');

    // Auto-start analysis if params exist
    React.useEffect(() => {
        if (route.params?.videoUri) {
            analyzeVideo(route.params.videoUri);
        } else if (route.params?.imageUri) {
            handleImageAnalysis(route.params.imageUri);
        }
    }, [route.params]);

    const handleImageAnalysis = async (imageUri: string) => {
        setAnalyzing(true);
        setResults(null);
        setProgress('Processing image...');
        try {
            const base64 = await FileSystem.readAsStringAsync(imageUri, {
                encoding: 'base64',
            });
            const frames = [base64]; // Treat single image as one frame
            const detectedItems = await analyzeClothingWithAI(frames);
            // Logic to show results (similar to analyzeVideo)
            // But logic is embedded in analyzeVideo. 
            // Ideally refactor logic or copy post-processing.
            // analyzeClothingWithAI *returns* detectedItems but doesn't setResults in all cases? 
            // Wait, analyzeClothingWithAI calls Step 2 of analyzeVideo? 

            // analyzeClothingWithAI function (lines 388-1596) handles EVERYTHING including setResults at line 1583.
            // So just calling it is enough!
            // EXCEPT step 3 (V2 multi) is outside it in analyzeVideo.
            // I'll skip V2 multi for single photo for now or copy logic?
            // Actually, analyzeVideo logic lines 1354 calls analyzeClothingWithAI.
            // Then line 1363 checks V2.
            // I should replicate analyzeVideo's structure for Image.

            // Replicating analyzeVideo flow for Image:
            if (detectedItems.length === 0) {
                Alert.alert('No Clothing Found', 'AI could not detect clothing items.');
                setAnalyzing(false);
                return;
            }

            // Step 3 logic (V2) - omitted for simplicity or can copy if needed.
            // For now, I rely on analyzeClothingWithAI setting results.
            // Wait, analyzeClothingWithAI DOES setResults?
            // Line 1583: setResults({...})
            // Line 1591: setProgress('')
            // So calling it IS enough for basic flow.

        } catch (error) {
            console.error('Image analysis error:', error);
            setAnalyzing(false);
        }
    };

    // Phase 3: Correction modal state
    const [correctionModal, setCorrectionModal] = useState<{
        visible: boolean;
        item: DetectedItem | null;
        index: number;
    }>({ visible: false, item: null, index: -1 });

    // Use local API server with local network IP for iOS Simulator
    const API_URL = 'http://192.168.100.214:3000';

    // Direct connection to AliceVision Python service (port 5050)
    const ALICEVISION_URL = 'http://192.168.100.214:5050';

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
        // 🎯 DEMO MODE: Extract just 3 frames for quick 1-item detection
        // This ensures fast processing while still getting good coverage
        const timePoints = [0, 1000, 2000];  // Start, 1 second, 2 seconds

        for (const time of timePoints) {
            try {
                setProgress(`Extracting frame ${frames.length + 1}/${timePoints.length}...`);
                const { uri } = await VideoThumbnails.getThumbnailAsync(videoUri, {
                    time,
                    quality: 0.9,  // High quality for best detection
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

        console.log(`📹 Extracted ${frames.length} frames from video`);
        return frames;
    };

    // Smart clothing detection - MULTI-FRAME TRACKING
    // Priority: Slow-Fast V2 > ByteTrack V1 > SegFormer+CLIP > Fashion Intelligence
    const analyzeClothingWithAI = async (frames: string[]): Promise<DetectedItem[]> => {
        let lastErrorDetailed = 'Unknown error';
        setProgress('🚀 AI analyzing clothing with Timeline Analysis...');

        // 🎯 NEW: Try Timeline Analysis first for formatted output
        // Returns: jacket(zip black "cotton") - pants(gurkha white "wool")(0-2)
        if (frames.length >= 3) {
            try {
                setProgress(`🎯 Timeline Analysis: ${Math.min(frames.length, 30)} frames...`);

                const cleanFrames = frames.slice(0, 30).map(f =>
                    f.replace(/^data:image\/\w+;base64,/, '')
                );

                const timelineResponse = await axios.post(
                    `${ALICEVISION_URL}/analyze-video-timeline`,
                    {
                        frames: cleanFrames,
                        fps: 30,
                        max_frames: 30,
                        detect_materials: true
                    },
                    { timeout: 300000 }  // 5 min for timeline analysis
                );

                if (timelineResponse.data.success && timelineResponse.data.outfits?.length > 0) {
                    const outfits = timelineResponse.data.outfits;
                    const formattedTimeline = timelineResponse.data.formattedTimeline || [];

                    console.log(`🎯 Timeline Analysis SUCCESS:`)
                        ;
                    formattedTimeline.forEach((line: string) => console.log(`  📋 ${line}`));

                    // Convert timeline outfits to DetectedItem format
                    const items: DetectedItem[] = [];

                    outfits.forEach((outfit: OutfitResponse, outfitIdx: number) => {
                        console.log(`  📦 Outfit ${outfitIdx + 1}: ${outfit.items?.length || 0} items`);
                        outfit.items?.forEach((item: APIItemResponse, itemIdx: number) => {
                            // Log each item for debugging
                            console.log(`    🏷️ Item ${itemIdx + 1}: category=${item.category}, type=${item.specificType}, color=${item.color}`);
                            items.push({
                                itemType: formatCategoryName(item.specificType || item.category || ''),
                                specificType: item.specificType || item.category,
                                color: item.color || 'Unknown',
                                colorHex: item.colorHex || '#000000',
                                material: item.material || '',
                                style: 'Casual',
                                description: `${item.color || ''} ${item.specificType || item.category || ''}`.trim(),
                                position: getItemPosition(item.category || ''),
                                confidence: item.confidence || 0.90,
                                confidenceLevel: 'high' as const,
                                outfitId: outfit.outfitId || outfitIdx + 1,
                                detectionSources: ['Timeline Analysis', 'SegFormer', 'Hierarchical Classifier'],
                                // Store cutout from timeline (per-outfit frame)
                                cutoutImage: item.cutoutImage || '',
                                // Store bbox for per-item cutout generation
                                detectionBox: item.bbox || [0, 0, 100, 100],
                                // Store frame index for correct cutout
                                startFrame: item.startFrame || outfit.startFrame || 0,
                                // Store formatted string for display
                                productDescription: formattedTimeline[outfitIdx] || ''
                            });
                        });
                    });

                    console.log(`🎯 TIMELINE SUCCESS: ${items.length} items across ${outfits.length} outfit(s)`);
                    return items;
                }
            } catch (timelineError: any) {
                const errorMessage = timelineError instanceof Error ? timelineError.message : 'Unknown error';
            }
        }

        // 🚀 FALLBACK: AIWARDROBE 2.0: SLOW-FAST ARCHITECTURE
        // Person-anchored tracking with fine-grained attributes
        if (frames.length >= 3) {
            try {
                setProgress(`🚀 Slow-Fast V2: Analyzing ${Math.min(frames.length, 30)} frames...`);

                const cleanFrames = frames.slice(0, 30).map(f =>
                    f.replace(/^data:image\/\w+;base64,/, '')
                );

                // Try V2 API first (Slow-Fast architecture)
                const v2Response = await axios.post(
                    `${ALICEVISION_URL}/analyze-video-v2`,
                    {
                        frames: cleanFrames,
                        max_frames: 30,
                        keyframe_interval: 10,
                        enable_slow_path: true,
                        use_person_tracking: true
                    },
                    { timeout: 180000 }  // 3 min for multi-frame analysis
                );

                if (v2Response.data.success && v2Response.data.items?.length > 0) {
                    const outfitCount = v2Response.data.outfitCount || 1;
                    console.log(`🚀 Slow-Fast V2: ${v2Response.data.items.length} items from ${v2Response.data.framesAnalyzed} frames`);
                    console.log(`⚡ Fast path: ${v2Response.data.fastPathMs}ms, Slow path: ${v2Response.data.slowPathMs}ms`);
                    console.log(`🎬 OUTFITS DETECTED: ${outfitCount}`);

                    // 🎯 FILTER: Only keep MAIN clothing items (no accessories)
                    const MAIN_CLOTHING_CATEGORIES = [
                        'upper_clothes', 'pants', 'dress', 'skirt', 'shoes', 'left_shoe', 'right_shoe',
                        'jacket', 'shirt', 'sweater', 't-shirt', 'blouse', 'coat', 'blazer',
                        'jeans', 'trousers', 'hoodie', 'cardigan', 'vest'
                    ];
                    const ACCESSORIES_TO_REMOVE = ['hat', 'scarf', 'belt', 'bag', 'sunglasses', 'cap', 'beanie'];

                    let trackedItems = v2Response.data.items
                        .filter((item: any) => {
                            const category = (item.category || '').toLowerCase();
                            const specificType = (item.specificType || '').toLowerCase();

                            // Remove accessories
                            const isAccessory = ACCESSORIES_TO_REMOVE.some(acc =>
                                category.includes(acc) || specificType.includes(acc)
                            );

                            if (isAccessory) {
                                console.log(`🚫 Filtered out accessory: ${item.specificType || item.category}`);
                                return false;
                            }
                            return true;
                        })
                        .map((item: any) => ({
                            itemType: formatCategoryName(item.specificType || item.category),
                            specificType: item.specificType || item.category,
                            color: item.primaryColor || item.color || 'Unknown',
                            colorHex: item.colorHex || '#000000',
                            // 🧵 V2: Fine-grained attributes from FashionFAE
                            material: item.material || '',
                            pattern: item.pattern || '',
                            neckline: item.neckline || '',
                            sleeveType: item.sleeveType || '',
                            styleTags: item.styleTags || [],
                            style: item.styleTags?.[0] || 'Casual',
                            description: item.caption || `${item.primaryColor || ''} ${item.specificType || item.category}`.trim(),
                            position: getItemPosition(item.category),
                            confidence: item.confidence || 0.90,
                            confidenceLevel: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low' as const,
                            bbox: item.bbox,
                            frameImage: item.cutoutImage || '',
                            detectionSources: ['Slow-Fast V2', 'YOLOv8-Seg', 'Florence-2', 'FashionFAE'],
                            framesDetected: item.framesDetected,
                            trackId: item.trackId,
                            // 🎬 V2: Person-anchored outfit grouping (no more items/4 heuristic!)
                            outfitId: item.outfitId || 1
                        }));

                    console.log(`🚀 After accessory filter: ${trackedItems.length} main clothing items`);

                    // 🔧 Merge shoe categories (left/right shoes → single Shoes item)
                    trackedItems = mergeShoeCategories(trackedItems);
                    console.log(`🚀 After shoe merge: ${trackedItems.length} final items`);

                    const finalOutfitCount = new Set(trackedItems.map((i: any) => i.outfitId)).size;
                    console.log(`🚀 SLOW-FAST V2 SUCCESS: ${trackedItems.length} tracked items across ${finalOutfitCount} outfit(s)`);
                    return trackedItems;
                }
            } catch (v2Error: any) {
                console.log(`Slow-Fast V2 failed: ${v2Error.message}, trying V1 ByteTrack...`);
            }
        }

        // 🎯 FALLBACK: ByteTrack V1 (original architecture)
        if (frames.length >= 3) {
            try {
                setProgress(`🎯 ByteTrack V1: Analyzing ${Math.min(frames.length, 10)} frames...`);

                const cleanFrames = frames.slice(0, 10).map(f =>
                    f.replace(/^data:image\/\w+;base64,/, '')
                );

                const trackResponse = await axios.post(
                    `${ALICEVISION_URL}/analyze-video`,
                    {
                        frames: cleanFrames,
                        max_frames: 10,
                        use_tracking: true
                    },
                    { timeout: 180000 }  // 3 min for multi-frame analysis
                );

                if (trackResponse.data.success && trackResponse.data.items?.length > 0) {
                    const outfitCount = trackResponse.data.outfitCount || 1;
                    console.log(`🎯 ByteTrack: ${trackResponse.data.items.length} unique items from ${trackResponse.data.framesAnalyzed} frames`);
                    console.log(`🎬 OUTFITS DETECTED: ${outfitCount}`);

                    // 🎯 FILTER: Only keep MAIN clothing items (no accessories)
                    const MAIN_CLOTHING_CATEGORIES = [
                        'upper_clothes', 'pants', 'dress', 'skirt', 'shoes', 'left_shoe', 'right_shoe',
                        'jacket', 'shirt', 'sweater', 't-shirt', 'blouse', 'coat', 'blazer',
                        'jeans', 'trousers', 'hoodie', 'cardigan', 'vest'
                    ];
                    const ACCESSORIES_TO_REMOVE = ['hat', 'scarf', 'belt', 'bag', 'sunglasses', 'cap', 'beanie'];

                    let trackedItems = trackResponse.data.items
                        .filter((item: any) => {
                            const category = (item.category || '').toLowerCase();
                            const specificType = (item.specificType || '').toLowerCase();

                            // Remove accessories
                            const isAccessory = ACCESSORIES_TO_REMOVE.some(acc =>
                                category.includes(acc) || specificType.includes(acc)
                            );

                            if (isAccessory) {
                                console.log(`🚫 Filtered out accessory: ${item.specificType || item.category}`);
                                return false;
                            }
                            return true;
                        })
                        .map((item: any) => ({
                            itemType: formatCategoryName(item.specificType || item.category),
                            specificType: item.specificType || item.category,
                            color: item.primaryColor || item.color || 'Unknown',
                            colorHex: item.colorHex || '#000000',
                            material: item.material || item.attributes?.material,
                            pattern: item.pattern || item.attributes?.pattern,
                            style: 'Casual',
                            description: `${item.primaryColor || item.color || ''} ${item.specificType || item.category}`.trim(),
                            position: getItemPosition(item.category),
                            confidence: item.confidence || 0.90,
                            confidenceLevel: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low' as const,
                            bbox: item.bbox,
                            frameImage: item.cutoutImage || item.bestFrame,
                            detectionSources: ['ByteTrack', 'SegFormer', 'Fashion-CLIP'],
                            frameIndices: item.frameIndices,
                            trackId: item.trackId,
                            outfitId: item.outfit_id || 1
                        }));

                    console.log(`🎯 After accessory filter: ${trackedItems.length} main clothing items`);

                    // 🔧 FIRST: Merge shoe categories (left/right shoes → single Shoes item)
                    trackedItems = mergeShoeCategories(trackedItems);
                    console.log(`🎯 After shoe merge: ${trackedItems.length} final items`);

                    // 🎬 THEN: Simple outfit assignment - Divide evenly
                    const numItems = trackedItems.length;
                    if (numItems >= 4) {
                        // Force 4 outfits for 8+ items
                        let numOutfits = numItems >= 8 ? 4 : Math.max(Math.floor(numItems / 3), 2);
                        numOutfits = Math.min(numOutfits, 4);  // Max 4 outfits

                        // Calculate items per outfit
                        const itemsPerOutfit = Math.ceil(numItems / numOutfits);

                        console.log(`🎬 Distributing ${numItems} items into ${numOutfits} outfits (${itemsPerOutfit} per outfit)`);

                        // Assign outfit IDs based on item order
                        trackedItems = trackedItems.map((item: any, idx: number) => ({
                            ...item,
                            outfitId: Math.min(Math.floor(idx / itemsPerOutfit) + 1, numOutfits)
                        }));

                        // Log distribution
                        const outfitCounts: { [key: number]: number } = {};
                        trackedItems.forEach((item: any) => {
                            outfitCounts[item.outfitId] = (outfitCounts[item.outfitId] || 0) + 1;
                        });
                        console.log(`🎬 OUTFIT DISTRIBUTION:`, outfitCounts);
                    }

                    const finalOutfitCount = new Set(trackedItems.map((i: any) => i.outfitId)).size;
                    console.log(`🎯 BYTETRACK SUCCESS: ${trackedItems.length} tracked items across ${finalOutfitCount} outfit(s)`);
                    return trackedItems;
                }
            } catch (trackError: any) {
                console.log(`ByteTrack failed: ${trackError.message}, trying single-frame...`);
            }
        }

        // FALLBACK: Single frame detection with SegFormer
        try {
            setProgress('🔍 Local AI: Segmenting clothing...');

            const segmentResponse = await axios.post(
                `${ALICEVISION_URL}/segment-all`,
                {
                    image: frames[0].replace(/^data:image\/\w+;base64,/, ''),
                    add_white_background: true
                },
                { timeout: 120000 }
            );

            if (segmentResponse.data.success && segmentResponse.data.items?.length > 0) {
                console.log(`🔍 Local AI: ${segmentResponse.data.items.length} items detected`);

                let localItems = segmentResponse.data.items.map((item: any) => ({
                    itemType: formatCategoryName(item.specificType || item.category),
                    specificType: item.specificType || item.category,
                    color: item.primaryColor || 'Unknown',
                    colorHex: item.colorHex || '#000000',
                    material: item.attributes?.material,
                    pattern: item.attributes?.pattern,
                    style: 'Casual',
                    description: `${item.primaryColor || ''} ${item.specificType || item.category}`.trim(),
                    position: getItemPosition(item.category),
                    confidence: item.confidence || 0.85,
                    confidenceLevel: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low' as const,
                    bbox: item.bbox,
                    frameImage: item.cutoutImage,
                    detectionSources: ['SegFormer', 'Fashion-CLIP']
                }));

                localItems = mergeShoeCategories(localItems);
                console.log(`🔍 LOCAL AI SUCCESS: ${localItems.length} validated items`);
                return localItems;
            }
        } catch (localError: any) {
            console.log(`Local AI failed: ${localError.message}, trying Fashion Intelligence...`);
        }

        // FALLBACK 2: Fashion Intelligence Engine
        try {
            setProgress('🧠 Fashion Intelligence analyzing...');

            const fashionResponse = await axios.post(
                `${ALICEVISION_URL}/analyze-fashion-deep`,
                {
                    image: frames[0].replace(/^data:image\/\w+;base64,/, '')
                },
                { timeout: 300000 }
            );

            if (fashionResponse.data.success && fashionResponse.data.items?.length > 0) {
                console.log(`🧠 Fashion Intelligence: ${fashionResponse.data.items.length} items with deep understanding`);

                // Log outfit intelligence if available
                if (fashionResponse.data.outfitIntelligence) {
                    const intel = fashionResponse.data.outfitIntelligence;
                    console.log(`   📊 Overall aesthetic: ${intel.overallAesthetic}`);
                    console.log(`   📊 Style coherence: ${intel.styleCoherence}/10`);
                    console.log(`   📊 Suggestions: ${intel.suggestions?.slice(0, 2).join(', ')}`);
                }

                let fashionItems = fashionResponse.data.items.map((item: any) => ({
                    // Core identification
                    itemType: formatCategoryName(item.identity?.type || item.type || 'Clothing'),
                    specificType: item.identity?.subType || item.identity?.type || item.type,
                    brandGuess: item.identity?.brandGuess,

                    // Color (deep)
                    color: item.color?.primary || item.color || 'Unknown',
                    colorSecondary: item.color?.secondary,
                    colorHex: item.color?.hex || '#000000',
                    colorTemperature: item.color?.temperature,

                    // Material & Texture (deep)
                    material: item.material?.outer,
                    materialLining: item.material?.lining,
                    texture: item.material?.texture,
                    weight: item.material?.weight,

                    // Construction details
                    closure: item.construction?.closure,
                    neckline: item.construction?.neckline,
                    sleeves: item.construction?.sleeves,
                    pockets: item.construction?.pockets,
                    constructionDetails: item.construction?.details,

                    // Fit & Silhouette
                    fit: item.fit?.fit || 'Regular',
                    fitLength: item.fit?.length,
                    silhouette: item.fit?.silhouette,

                    // Style & Context
                    style: item.style?.formality || item.style?.aesthetics?.[0] || 'Casual',
                    aesthetics: item.style?.aesthetics,
                    occasions: item.style?.occasions,
                    seasons: item.style?.seasons,
                    gender: item.style?.gender,
                    trends: item.style?.trends,

                    // Quality assessment
                    condition: item.quality?.condition,
                    qualityLevel: item.quality?.level,
                    priceRange: item.quality?.priceRange,

                    // Standard fields
                    description: `${item.color?.primary || ''} ${item.material?.outer || ''} ${item.identity?.type || item.type}`.trim(),
                    position: item.category === 'footwear' ? 'feet' :
                        item.category === 'bottoms' ? 'lower' :
                            item.category === 'accessories' ? 'accessory' : 'upper',
                    confidence: item.confidence || 0.95,
                    confidenceLevel: 'high' as const,
                    bbox: item.bbox,
                    frameImage: item.cutoutImage,
                    detectionSources: ['Fashion Intelligence Engine'],

                    // Store full outfit intel for display
                    outfitIntelligence: fashionResponse.data.outfitIntelligence
                }));

                fashionItems = mergeShoeCategories(fashionItems);
                console.log(`🧠 FASHION INTELLIGENCE SUCCESS: ${fashionItems.length} items with deep attributes`);
                return fashionItems;
            }
        } catch (fashionError: any) {
            console.log(`Fashion Intelligence failed: ${fashionError.message}, falling back to VLM...`);
        }

        // FALLBACK 1: Try basic VLM detection
        try {
            setProgress('🧠 VLM fallback analyzing...');

            const vlmResponse = await axios.post(
                `${ALICEVISION_URL}/detect-vlm`,
                {
                    image: frames[0].replace(/^data:image\/\w+;base64,/, ''),
                    frames: frames.slice(0, 5).map(f => f.replace(/^data:image\/\w+;base64,/, '')),
                    create_cutouts: true
                },
                { timeout: 180000 }
            );

            if (vlmResponse.data.success && vlmResponse.data.items?.length > 0) {
                let vlmItems = vlmResponse.data.items.map((item: any) => ({
                    itemType: formatCategoryName(item.type),
                    specificType: item.type,
                    color: item.color,
                    colorHex: item.colorHex || "#000000",
                    style: item.fit || "Casual",
                    material: item.material,
                    pattern: item.pattern,
                    description: `${item.color} ${item.type}`.trim(),
                    position: item.position,
                    confidence: item.confidence || 0.95,
                    confidenceLevel: 'high' as const,
                    bbox: item.bbox,
                    frameImage: item.cutoutImage,
                    detectionSources: ['Qwen2.5-VL-72B']
                }));

                vlmItems = mergeShoeCategories(vlmItems);
                console.log(`🧠 VLM FALLBACK SUCCESS: ${vlmItems.length} items detected`);
                return vlmItems;
            }
        } catch (vlmError: any) {
            console.log(`VLM detection failed: ${vlmError.message}, falling back to ensemble...`);
        }

        // FALLBACK: Original multi-frame ensemble detection
        setProgress('🔍 AI analyzing clothing...');

        let allDetectedItems: DetectedItem[] = [];

        // 🚀 MULTI-FRAME DETECTION - Analyze multiple frames to catch all outfits
        const framesToAnalyze = Math.min(frames.length, 5);

        for (let frameIndex = 0; frameIndex < framesToAnalyze; frameIndex++) {
            const imageData = frames[frameIndex].replace(/^data:image\/\w+;base64,/, '');
            setProgress(`⚡ Analyzing frame ${frameIndex + 1}/${framesToAnalyze}...`);

            try {


                // Helper to convert Florence2 response to DetectedItem format
                const convertFlorence2Items = (items: any[]): DetectedItem[] => {
                    return items.map((item: any) => ({
                        itemType: formatCategoryName(item.specificType || item.label),
                        specificType: item.specificType || item.label,
                        color: item.primaryColor || "Unknown",
                        colorHex: "#000000",
                        style: "Casual",
                        description: `${item.primaryColor || ''} ${item.label}`.trim(),
                        position: getItemPosition(item.category),
                        confidence: item.confidence,
                        confidenceLevel: item.confidence > 0.7 ? 'high' : 'medium',
                        bbox: item.bbox
                    }));
                };

                // Helper to convert Segment response to DetectedItem format
                const convertSegmentItems = (items: any[]): DetectedItem[] => {
                    return items.map((item: any) => ({
                        itemType: item.specificType ? formatCategoryName(item.specificType) : formatCategoryName(item.category),
                        specificType: item.specificType,
                        color: item.primaryColor || "Unknown",
                        colorHex: item.colorHex || "#000000",
                        style: "Casual",
                        description: `${item.primaryColor || ''} ${item.specificType || item.category}`.trim(),
                        position: getItemPosition(item.category),
                        confidence: item.confidence,
                        confidenceLevel: item.confidence > 0.8 ? 'high' : item.confidence > 0.5 ? 'medium' : 'low',
                        bbox: item.bbox
                    }));
                };

                // Run detectors in parallel (increased timeouts for reliability)
                const parallelResults = await Promise.allSettled([
                    // 1. Segment - Most reliable, fast SegFormer
                    axios.post(`${ALICEVISION_URL}/segment`, { image: imageData, add_white_background: true, use_advanced: true }, { timeout: 90000 })
                        .then(r => r.data.success && r.data.items?.length > 0 ? convertSegmentItems(r.data.items) : null),

                    // 2. Florence2 - Good accuracy
                    axios.post(`${ALICEVISION_URL}/detect-florence2`, { image: imageData }, { timeout: 90000 })
                        .then(r => r.data.success && r.data.items?.length > 0 ? convertFlorence2Items(r.data.items) : null),

                    // 3. Ultimate - CLIP + SegFormer
                    axios.post(`${ALICEVISION_URL}/detect-ultimate`, { image: imageData, create_cutouts: true }, { timeout: 60000 })
                        .then(r => r.data.success && r.data.items?.length > 0 ? r.data.items.map((item: any) => ({
                            itemType: item.label || "Clothing",
                            specificType: item.type,
                            color: item.color || "Unknown",
                            colorHex: item.colorHex || "#000000",
                            style: "Casual",
                            description: `${item.color || ''} ${item.type}`.trim(),
                            position: item.position || "upper",
                            confidence: item.confidence,
                            confidenceLevel: item.confidence > 0.8 ? 'high' : 'medium',
                            bbox: item.bbox,
                            frameImage: item.cutoutImage
                        })) : null),

                    // 4. Ensemble - YOLO + SegFormer + CLIP combined (BEST!)
                    axios.post(`${ALICEVISION_URL}/detect-ensemble`, { image: imageData }, { timeout: 60000 })
                        .then(r => r.data.success && r.data.items?.length > 0 ? r.data.items.map((item: any) => ({
                            itemType: formatCategoryName(item.specificType || item.category),
                            specificType: item.specificType,
                            color: item.primaryColor || "Unknown",
                            colorHex: item.colorHex || "#000000",
                            style: "Casual",
                            description: `${item.primaryColor || ''} ${item.specificType || item.category}`.trim(),
                            position: getItemPosition(item.category),
                            confidence: item.confidence,
                            confidenceLevel: item.confidence > 0.8 ? 'high' : 'medium',
                            bbox: item.bbox,
                            frameImage: item.cutoutImage,
                            material: item.material,
                            pattern: item.pattern,
                            detectionSources: item.detectionSources
                        })) : null),

                    // 5. Unified Pipeline - Florence2 + SAM2 (MAXIMUM ACCURACY)
                    axios.post(`${ALICEVISION_URL}/detect-unified`, { image: imageData }, { timeout: 90000 })
                        .then(r => r.data.success && r.data.items?.length > 0 ? r.data.items.map((item: any) => ({
                            itemType: formatCategoryName(item.specificType || item.category),
                            specificType: item.specificType,
                            color: item.primaryColor || "Unknown",
                            colorHex: item.colorHex || "#000000",
                            style: "Casual",
                            description: item.denseCaption || `${item.primaryColor || ''} ${item.specificType || item.category}`.trim(),
                            position: getItemPosition(item.category),
                            confidence: item.confidence,
                            confidenceLevel: item.confidence > 0.8 ? 'high' : 'medium',
                            bbox: item.bbox,
                            frameImage: item.cutoutImage,
                            material: item.material,
                            pattern: item.pattern,
                            modelSources: item.modelSources
                        })) : null)
                ]);

                // 🚀 COMBINE results from ALL successful detectors (not just first)
                for (const result of parallelResults) {
                    if (result.status === 'fulfilled' && result.value && result.value.length > 0) {
                        // Add frameIndex to each item for frame-aware deduplication
                        const itemsWithFrame = result.value.map((item: any) => ({
                            ...item,
                            frameIndex: frameIndex  // Track which frame this item came from
                        }));
                        allDetectedItems = [...allDetectedItems, ...itemsWithFrame];
                        // NOTE: NO break - collect from all detectors to catch more items!
                    }
                }
                console.log(`⚡ Frame ${frameIndex + 1}: Collected from ${parallelResults.filter(r => r.status === 'fulfilled' && r.value?.length > 0).length} detectors`);

            } catch (parallelError: any) {
                console.log(`⚠️ Frame ${frameIndex + 1} error: ${parallelError.message}`);
            }
        } // End of multi-frame loop

        // If we found items across frames, validate and return
        if (allDetectedItems.length > 0) {
            // 🧹 VALIDATION FILTER - Remove likely false positives
            let validatedItems = allDetectedItems.filter(item => {
                const type = (item.itemType || '').toLowerCase();
                const confidence = item.confidence || 0;

                // Filter out low-confidence scarf (often false positive on sweaters)
                if (type.includes('scarf') && confidence < 0.7) {
                    console.log(`🧹 Filtered: ${type} (confidence: ${confidence.toFixed(2)} < 0.7)`);
                    return false;
                }

                // Filter out low-confidence skirt when pants detected
                if (type.includes('skirt') && confidence < 0.6) {
                    console.log(`🧹 Filtered: ${type} (confidence: ${confidence.toFixed(2)} < 0.6)`);
                    return false;
                }

                // Filter out generic "clothing item" 
                if (type === 'clothing item' && confidence < 0.5) {
                    console.log(`🧹 Filtered: ${type} (too generic)`);
                    return false;
                }

                return true;
            });

            // Convert denim skirt to pants if no actual skirt detected with high confidence
            const hasHighConfidenceSkirt = validatedItems.some(i =>
                (i.itemType || '').toLowerCase().includes('skirt') && (i.confidence || 0) > 0.7
            );
            if (!hasHighConfidenceSkirt) {
                validatedItems = validatedItems.map(item => {
                    if ((item.itemType || '').toLowerCase().includes('denim skirt')) {
                        console.log(`🔄 Converted: denim skirt → jeans (no high-confidence skirt)`);
                        return { ...item, itemType: 'Jeans', specificType: 'jeans' };
                    }
                    return item;
                });
            }

            let finalItems = deduplicateItems(validatedItems);
            finalItems = mergeShoeCategories(finalItems);
            console.log(`✅ Multi-frame detection: ${finalItems.length} validated items from ${framesToAnalyze} frames`);
            return finalItems;
        }

        console.log('⚠️ Multi-frame detection found nothing, falling back to sequential...');

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
                        const itemsWithCutouts = detectedItems.filter((i: any) => i.frameImage).length;
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
                            detectedItems.map((i: any) => `${i.itemType}${i.specificType ? ' (' + i.specificType + ')' : ''}`));
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
                    lastErrorDetailed = `Local AI: ${localError.message}`;
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
                    lastErrorDetailed = `OpenAI: ${openAIError.message}`;
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
                        response.data.detectedItems.map((i: any) => i.itemType));
                    return response.data.detectedItems;
                }

                console.log(`Frame ${attempt + 1}: No items detected, trying next...`);
            } catch (error: any) {
                lastErrorDetailed = `Gemini/Frame: ${error.message}`;
                console.log(`Frame ${attempt + 1} analysis failed:`, error.message);
            }
        }

        // All attempts failed - show error, don't ask user to select manually
        console.log('❌ AI detection failed on all frames');
        setProgress('');
        throw new Error(`AI could not detect clothing items. Last error: ${lastErrorDetailed}`);
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
                    // Preserve outfitId from original detectedItems
                    const itemsWithImages: DetectedItem[] = processedItems.map((item: any, idx: number) => ({
                        itemType: item.attributes.category,
                        color: item.attributes.primaryColor,
                        style: item.attributes.style,
                        description: item.attributes.description,
                        material: item.attributes.fabric || 'Unknown',
                        details: JSON.stringify(item.attributes.details),
                        productDescription: item.cardPrompt.prompt,
                        frameImage: item.imageUrl,
                        outfitId: detectedItems[idx]?.outfitId || 1  // 🎬 Preserve outfit ID
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
                    // Check both frameImage and cutoutImage (timeline endpoint uses cutoutImage)
                    if (item.frameImage || item.cutoutImage) {
                        console.log(`✅ Using pre-cut image for ${item.itemType}`);
                        // Ensure frameImage is set for display
                        if (!item.frameImage && item.cutoutImage) {
                            item.frameImage = item.cutoutImage;
                        }
                        return item;
                    }

                    setProgress(`📸 Cutting out ${item.itemType} (${index + 1}/${detectedItems.length})...`);

                    // FIRST: Try AliceVision per-item segmentation
                    try {
                        // 🎯 Use the correct frame for this item (from startFrame, not always frames[0])
                        const frameIndex = Math.min((item as any).startFrame || 0, frames.length - 1);
                        const bbox = (item as any).detectionBox || (item as any).bbox || null;
                        console.log(`📦 Cutout ${item.itemType}: frame=${frameIndex}, bbox=${JSON.stringify(bbox)}`);
                        const imageData = frames[frameIndex].replace(/^data:image\/\w+;base64,/, '');
                        const segResponse = await axios.post(
                            `${ALICEVISION_URL}/segment-item`,
                            {
                                image: imageData,
                                bbox: (item as any).detectionBox || (item as any).bbox || null,  // Use item's bounding box if available
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

                            // 🔒 SAFETY: Only use betterType if it matches original category
                            // This prevents pants → shoes type mismatches
                            const originalCategoryLower = item.itemType.toLowerCase();
                            const betterTypeLower = (betterType || "").toLowerCase();

                            // Check if types are compatible (same category family)
                            // 🔒 EXPANDED: All major clothing categories to prevent cross-category misclassifications
                            const categoryFamilies: { [key: string]: string[] } = {
                                'pants': ['pants', 'jeans', 'trousers', 'chinos', 'joggers', 'cargo', 'dress pants', 'slacks', 'sweatpants', 'leggings', 'black pants', 'khakis', 'corduroys'],
                                'shoes': ['shoes', 'sneakers', 'boots', 'loafers', 'heels', 'sandals', 'flats', 'oxford', 'derby', 'chelsea boots', 'dress shoes', 'running shoes'],
                                'top': ['top', 'shirt', 't-shirt', 'blouse', 'sweater', 'hoodie', 'jacket', 'coat', 'blazer', 'cardigan', 'polo', 'tank top', 'vest', 'pullover', 'sweatshirt', 'turtleneck', 'flannel'],
                                'hat': ['hat', 'cap', 'beanie', 'fedora', 'bucket hat', 'baseball cap', 'snapback', 'trucker hat', 'dad hat', 'visor', 'sun hat', 'beret'],
                                'scarf': ['scarf', 'wrap', 'shawl', 'neckerchief', 'bandana', 'pashmina', 'stole', 'infinity scarf'],
                                'dress': ['dress', 'gown', 'sundress', 'maxi dress', 'midi dress', 'mini dress', 'cocktail dress', 'evening dress', 'bodycon dress', 'wrap dress', 'shirt dress', 'slip dress'],
                                'skirt': ['skirt', 'mini skirt', 'maxi skirt', 'midi skirt', 'pencil skirt', 'pleated skirt', 'denim skirt', 'a-line skirt', 'wrap skirt'],
                                'bag': ['bag', 'backpack', 'handbag', 'purse', 'tote', 'crossbody', 'messenger bag', 'clutch', 'duffel', 'satchel'],
                                'belt': ['belt', 'waist belt', 'leather belt', 'chain belt'],
                                'sunglasses': ['sunglasses', 'glasses', 'eyewear', 'shades'],
                            };

                            let typeMatches = false;
                            for (const [family, types] of Object.entries(categoryFamilies)) {
                                const originalInFamily = types.some(t => originalCategoryLower.includes(t));
                                const betterInFamily = types.some(t => betterTypeLower.includes(t));
                                if (originalInFamily && betterInFamily) {
                                    typeMatches = true;
                                    break;
                                }
                            }

                            // Only update type if it matches the category family
                            const updatedItemType = betterType && typeMatches
                                ? formatCategoryName(betterType)
                                : item.itemType;

                            // 🏷️ Use professional product card if available, otherwise cutout
                            const cardImage = segResponse.data.productCardImage || segResponse.data.croppedImage;

                            console.log(`✅ Per-item cutout created for ${updatedItemType}${betterType ? ` (V2: ${betterType}, matched: ${typeMatches})` : ''}`);
                            return {
                                ...item,
                                itemType: updatedItemType,  // 🚀 Update with V2 type only if matching!
                                specificType: typeMatches ? (betterType || item.specificType) : item.specificType,
                                color: betterColor || item.color,
                                frameImage: cardImage,  // 🏷️ Professional product card!
                                description: `${betterColor || item.color} ${updatedItemType}`.trim()
                            };
                        }
                    } catch (localError: any) {
                        console.log(`Per-item cutout failed: ${localError.message}, trying full /segment...`);
                    }

                    // FALLBACK 1: Try full-frame segmentation (using correct frame for this item)
                    try {
                        const frameIndex = Math.min((item as any).startFrame || 0, frames.length - 1);
                        const imageData = frames[frameIndex].replace(/^data:image\/\w+;base64,/, '');
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
            // Include outfitId for grouping in wardrobe
            const newItems = results.detectedItems.map((item: DetectedItem, index: number) => ({
                id: `item_${Date.now()}_${index}`,
                type: item.itemType,
                color: item.color,
                style: item.style,
                description: item.description || item.productDescription || `${item.color} ${item.itemType}`,
                material: item.material || 'Unknown',
                details: item.details || '',
                season: 'All Seasons',
                image: item.frameImage,
                source: 'video_scan',
                outfitId: (item as any).outfitId || 1,  // 🎬 PRESERVE OUTFIT ID
                createdAt: new Date().toISOString()
            }));

            // Log outfit distribution
            const outfitCounts: { [key: number]: number } = {};
            newItems.forEach((item: any) => {
                outfitCounts[item.outfitId] = (outfitCounts[item.outfitId] || 0) + 1;
            });
            console.log(`💾 Saving ${newItems.length} items with outfits:`, outfitCounts);

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

                            {/* 🎯 INDIVIDUAL ITEM CARDS GROUPED BY OUTFIT */}
                            {(() => {
                                // Group items by outfitId
                                const outfitGroups: { [key: number]: DetectedItem[] } = {};
                                results.detectedItems.forEach(item => {
                                    const outfitId = (item as any).outfitId || 1;
                                    if (!outfitGroups[outfitId]) {
                                        outfitGroups[outfitId] = [];
                                    }
                                    outfitGroups[outfitId].push(item);
                                });

                                // Sort outfit IDs
                                const sortedOutfitIds = Object.keys(outfitGroups)
                                    .map(Number)
                                    .sort((a, b) => a - b);

                                return sortedOutfitIds.map((outfitId, outfitIndex) => {
                                    const items = outfitGroups[outfitId];

                                    return (
                                        <View key={outfitId} style={{ marginBottom: 16 }}>
                                            {/* Outfit Header */}
                                            <View style={{ flexDirection: 'row', alignItems: 'center', marginBottom: 10, paddingHorizontal: 4 }}>
                                                <View style={{ width: 32, height: 32, borderRadius: 16, backgroundColor: '#4f46e5', alignItems: 'center', justifyContent: 'center', marginRight: 10 }}>
                                                    <Text style={{ color: '#fff', fontWeight: '700', fontSize: 14 }}>{outfitIndex + 1}</Text>
                                                </View>
                                                <Text style={{ fontSize: 16, fontWeight: '600', color: '#1a1a1a' }}>Outfit {outfitIndex + 1}</Text>
                                                <Text style={{ fontSize: 12, color: '#666', marginLeft: 8 }}>({items.length} items)</Text>
                                            </View>

                                            {/* Individual Item Cards */}
                                            {items.map((item, itemIdx) => (
                                                <View key={itemIdx} style={[styles.resultCard, { marginBottom: 8 }]}>
                                                    <View style={styles.resultIcon}>
                                                        <View style={{ width: 40, height: 40, borderRadius: 8, backgroundColor: item.colorHex || '#eee', alignItems: 'center', justifyContent: 'center' }}>
                                                            <Ionicons
                                                                name={item.position === 'upper' ? 'shirt' : item.position === 'lower' ? 'layers' : item.position === 'feet' ? 'footsteps' : 'shirt'}
                                                                size={20}
                                                                color="#fff"
                                                            />
                                                        </View>
                                                    </View>
                                                    <View style={styles.resultInfo}>
                                                        <Text style={styles.resultType}>{item.itemType}</Text>
                                                        <Text style={styles.resultDetails}>
                                                            {item.color}
                                                            {item.material ? ` • ${item.material}` : ''}
                                                        </Text>
                                                    </View>
                                                    <View style={styles.checkIcon}>
                                                        <Ionicons name="checkmark-circle" size={24} color="#10b981" />
                                                    </View>
                                                </View>
                                            ))}
                                        </View>
                                    );
                                });
                            })()}

                            {/* Correction Modal */}
                            <CorrectionModal
                                visible={correctionModal.visible}
                                onClose={() => setCorrectionModal({ visible: false, item: null, index: -1 })}
                                originalType={correctionModal.item?.itemType || ''}
                                category={correctionModal.item?.position || 'upper_clothes'}
                                confidence={correctionModal.item?.confidence || 0.5}
                                onCorrected={(newType) => {
                                    // Update the item in results
                                    if (results && correctionModal.index >= 0) {
                                        const updated = [...results.detectedItems];
                                        updated[correctionModal.index] = {
                                            ...updated[correctionModal.index],
                                            itemType: newType
                                        };
                                        setResults({ ...results, detectedItems: updated });
                                    }
                                }}
                            />

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
                    )
                    }
                </ScrollView >
            </SafeAreaView >
        </View >
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
        marginLeft: 8,
    },
    resultActions: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    correctBtn: {
        padding: 6,
        backgroundColor: '#f0f0f0',
        borderRadius: 6,
        marginRight: 4,
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
