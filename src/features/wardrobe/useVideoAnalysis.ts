/**
 * useVideoAnalysis — hook encapsulating all wardrobe video/image analysis logic.
 * Extracted from WardrobeVideoScreen (was ~2100 lines) to make the screen a thin orchestrator.
 */

import { useState } from 'react';
import axios from 'axios';
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as FileSystem from 'expo-file-system/legacy';
import { Alert } from 'react-native';
import Config from '../../config/env';

const API_URL = Config.api.url;
import {
    DetectedItem,
    AnalysisResult,
    APIItemResponse,
    OutfitResponse,
} from './types';
import {
    formatCategoryName,
    getItemPosition,
    getClothingFallbackImage,
    deduplicateItems,
    mergeShoeCategories,
} from './wardrobeUtils';

const ALICEVISION_URL = Config.api.alicevisionUrl;

const ACCESSORIES_TO_REMOVE = ['hat', 'scarf', 'belt', 'bag', 'sunglasses', 'cap', 'beanie'];
const FRAME_TIME_POINTS = [0, 1000, 2000]; // ms

export interface UseVideoAnalysisReturn {
    analyzing: boolean;
    progress: string;
    results: AnalysisResult | null;
    analyzeVideo: (videoUri: string) => Promise<void>;
    analyzeImage: (imageUri: string) => Promise<void>;
    reset: () => void;
}

export const useVideoAnalysis = (): UseVideoAnalysisReturn => {
    const [analyzing, setAnalyzing] = useState(false);
    const [progress, setProgress] = useState('');
    const [results, setResults] = useState<AnalysisResult | null>(null);

    const reset = () => {
        setAnalyzing(false);
        setProgress('');
        setResults(null);
    };

    // ─── Frame Extraction ───────────────────────────────────────────────────────

    const extractFrames = async (videoUri: string): Promise<string[]> => {
        const frames: string[] = [];
        for (const time of FRAME_TIME_POINTS) {
            try {
                setProgress(`Extracting frame ${frames.length + 1}/${FRAME_TIME_POINTS.length}...`);
                const { uri } = await VideoThumbnails.getThumbnailAsync(videoUri, { time, quality: 0.9 });
                const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
                frames.push(base64);
            } catch {
                // Skip failed frames — continue with what we have
            }
        }
        return frames;
    };

    // ─── Item Mapping Helpers ────────────────────────────────────────────────────

    const mapToDetectedItem = (item: APIItemResponse, overrides: Partial<DetectedItem> = {}): DetectedItem => ({
        itemType: formatCategoryName(item.specificType || item.category || ''),
        specificType: item.specificType || item.category,
        color: item.primaryColor || item.color || 'Unknown',
        colorHex: item.colorHex || '#000000',
        material: item.material,
        pattern: item.pattern,
        style: 'Casual',
        description: `${item.primaryColor || item.color || ''} ${item.specificType || item.category || ''}`.trim(),
        position: getItemPosition(item.category || ''),
        confidence: item.confidence || 0.85,
        confidenceLevel: (item.confidence || 0) > 0.8 ? 'high' : (item.confidence || 0) > 0.5 ? 'medium' : 'low',
        bbox: item.bbox,
        cutoutImage: item.cutoutImage,
        ...overrides,
    });

    // ─── Detection Strategies (priority order) ──────────────────────────────────

    /** Strategy 1: Timeline Analysis — best for multi-outfit videos */
    const tryTimelineAnalysis = async (frames: string[]): Promise<DetectedItem[] | null> => {
        if (frames.length < 3) return null;
        try {
            setProgress(`Timeline Analysis: ${Math.min(frames.length, 30)} frames...`);
            const cleanFrames = frames.slice(0, 30).map(f => f.replace(/^data:image\/\w+;base64,/, ''));
            const response = await axios.post(
                `${ALICEVISION_URL}/analyze-video-timeline`,
                { frames: cleanFrames, fps: 30, max_frames: 30, detect_materials: true },
                { timeout: 300000 }
            );
            if (!response.data.success || !response.data.outfits?.length) return null;

            const items: DetectedItem[] = [];
            const formattedTimeline: string[] = response.data.formattedTimeline || [];
            response.data.outfits.forEach((outfit: OutfitResponse, outfitIdx: number) => {
                outfit.items?.forEach((item: APIItemResponse) => {
                    items.push(mapToDetectedItem(item, {
                        confidence: item.confidence || 0.90,
                        confidenceLevel: 'high',
                        outfitId: outfit.outfitId || outfitIdx + 1,
                        detectionSources: ['Timeline Analysis', 'SegFormer', 'Hierarchical Classifier'],
                        startFrame: item.startFrame || outfit.startFrame || 0,
                        productDescription: formattedTimeline[outfitIdx] || '',
                        detectionBox: item.bbox || [0, 0, 100, 100],
                    }));
                });
            });
            return items.length > 0 ? items : null;
        } catch (err) {
            console.warn('[Detection] Timeline analysis failed:', err);
            return null;
        }
    };

    /** Strategy 2: ByteTrack V1 — multi-frame tracking */
    const tryByteTrackAnalysis = async (frames: string[]): Promise<DetectedItem[] | null> => {
        if (frames.length < 3) return null;
        try {
            setProgress(`ByteTrack V1: Analyzing ${Math.min(frames.length, 10)} frames...`);
            const cleanFrames = frames.slice(0, 10).map(f => f.replace(/^data:image\/\w+;base64,/, ''));
            const response = await axios.post(
                `${ALICEVISION_URL}/analyze-video`,
                { frames: cleanFrames, max_frames: 10, use_tracking: true },
                { timeout: 180000 }
            );
            if (!response.data.success || !response.data.items?.length) return null;

            let items: DetectedItem[] = response.data.items
                .filter((item: any) => {
                    const cat = (item.category || '').toLowerCase();
                    const spec = (item.specificType || '').toLowerCase();
                    return !ACCESSORIES_TO_REMOVE.some(a => cat.includes(a) || spec.includes(a));
                })
                .map((item: any) => mapToDetectedItem(item, {
                    detectionSources: ['ByteTrack', 'SegFormer', 'Fashion-CLIP'],
                    trackId: item.trackId,
                    outfitId: item.outfit_id || 1,
                }));

            items = mergeShoeCategories(items);

            // Distribute into outfits if many items
            if (items.length >= 4) {
                const numOutfits = Math.min(items.length >= 8 ? 4 : Math.max(Math.floor(items.length / 3), 2), 4);
                const perOutfit = Math.ceil(items.length / numOutfits);
                items = items.map((item, idx) => ({ ...item, outfitId: Math.min(Math.floor(idx / perOutfit) + 1, numOutfits) }));
            }
            return items.length > 0 ? items : null;
        } catch (err) {
            console.warn('[Detection] ByteTrack analysis failed:', err);
            return null;
        }
    };

    /** Strategy 3: SegFormer single-frame */
    const trySegFormerAnalysis = async (frame: string): Promise<DetectedItem[] | null> => {
        try {
            setProgress('Local AI: Segmenting clothing...');
            const response = await axios.post(
                `${ALICEVISION_URL}/segment-all`,
                { image: frame.replace(/^data:image\/\w+;base64,/, ''), add_white_background: true },
                { timeout: 120000 }
            );
            if (!response.data.success || !response.data.items?.length) return null;
            let items = response.data.items.map((item: any) => mapToDetectedItem(item, {
                detectionSources: ['SegFormer', 'Fashion-CLIP'],
            }));
            items = mergeShoeCategories(items);
            return items.length > 0 ? items : null;
        } catch (err) {
            console.warn('[Detection] SegFormer analysis failed:', err);
            return null;
        }
    };

    /** Strategy 4: Fashion Intelligence deep analysis */
    const tryFashionIntelligence = async (frame: string): Promise<DetectedItem[] | null> => {
        try {
            setProgress('Fashion Intelligence analyzing...');
            const response = await axios.post(
                `${ALICEVISION_URL}/analyze-fashion-deep`,
                { image: frame.replace(/^data:image\/\w+;base64,/, '') },
                { timeout: 300000 }
            );
            if (!response.data.success || !response.data.items?.length) return null;
            let items: DetectedItem[] = response.data.items.map((item: any) => ({
                itemType: formatCategoryName(item.identity?.type || item.type || 'Clothing'),
                specificType: item.identity?.subType || item.identity?.type || item.type,
                color: item.color?.primary || item.color || 'Unknown',
                colorHex: item.color?.hex || '#000000',
                material: item.material?.outer,
                style: item.style?.formality || 'Casual',
                description: `${item.color?.primary || ''} ${item.material?.outer || ''} ${item.identity?.type || item.type || ''}`.trim(),
                position: item.category === 'footwear' ? 'feet' : item.category === 'bottoms' ? 'lower' : item.category === 'accessories' ? 'accessory' : 'upper',
                confidence: item.confidence || 0.95,
                confidenceLevel: 'high' as const,
                bbox: item.bbox,
                frameImage: item.cutoutImage,
                detectionSources: ['Fashion Intelligence Engine'],
            }));
            items = mergeShoeCategories(items);
            return items.length > 0 ? items : null;
        } catch (err) {
            console.warn('[Detection] Fashion Intelligence failed:', err);
            return null;
        }
    };

    /** Strategy 5: VLM (Qwen2.5-VL) fallback */
    const tryVLMDetection = async (frames: string[]): Promise<DetectedItem[] | null> => {
        try {
            setProgress('VLM fallback analyzing...');
            const response = await axios.post(
                `${ALICEVISION_URL}/detect-vlm`,
                {
                    image: frames[0].replace(/^data:image\/\w+;base64,/, ''),
                    frames: frames.slice(0, 5).map(f => f.replace(/^data:image\/\w+;base64,/, '')),
                    create_cutouts: true,
                },
                { timeout: 180000 }
            );
            if (!response.data.success || !response.data.items?.length) return null;
            let items: DetectedItem[] = response.data.items.map((item: any) => ({
                itemType: formatCategoryName(item.type),
                specificType: item.type,
                color: item.color,
                colorHex: item.colorHex || '#000000',
                style: item.fit || 'Casual',
                material: item.material,
                pattern: item.pattern,
                description: `${item.color} ${item.type}`.trim(),
                position: item.position,
                confidence: item.confidence || 0.95,
                confidenceLevel: 'high' as const,
                bbox: item.bbox,
                frameImage: item.cutoutImage,
                detectionSources: ['Qwen2.5-VL-72B'],
            }));
            items = mergeShoeCategories(items);
            return items.length > 0 ? items : null;
        } catch (err) {
            console.warn('[Detection] VLM detection failed:', err);
            return null;
        }
    };

    /** Strategy 6: Parallel multi-model ensemble across multiple frames */
    const tryParallelEnsemble = async (frames: string[]): Promise<DetectedItem[] | null> => {
        const framesToAnalyze = Math.min(frames.length, 5);
        let allItems: DetectedItem[] = [];

        for (let frameIndex = 0; frameIndex < framesToAnalyze; frameIndex++) {
            const imageData = frames[frameIndex].replace(/^data:image\/\w+;base64,/, '');
            setProgress(`Analyzing frame ${frameIndex + 1}/${framesToAnalyze}...`);

            const results = await Promise.allSettled([
                axios.post(`${ALICEVISION_URL}/segment`, { image: imageData, add_white_background: true, use_advanced: true }, { timeout: 90000 })
                    .then(r => r.data.success && r.data.items?.length > 0 ? r.data.items.map((i: any) => mapToDetectedItem(i)) : null),
                axios.post(`${ALICEVISION_URL}/detect-florence2`, { image: imageData }, { timeout: 90000 })
                    .then(r => r.data.success && r.data.items?.length > 0 ? r.data.items.map((i: any) => mapToDetectedItem(i)) : null),
                axios.post(`${ALICEVISION_URL}/detect-ensemble`, { image: imageData }, { timeout: 60000 })
                    .then(r => r.data.success && r.data.items?.length > 0 ? r.data.items.map((i: any) => mapToDetectedItem(i, { detectionSources: i.detectionSources })) : null),
            ]);

            for (const result of results) {
                if (result.status === 'fulfilled' && result.value?.length > 0) {
                    const withFrame = result.value.map((item: DetectedItem) => ({ ...item, frameIndex }));
                    allItems = [...allItems, ...withFrame];
                }
            }
        }

        if (allItems.length === 0) return null;

        // Validate and post-process
        let validated = allItems.filter(item => {
            const type = (item.itemType || '').toLowerCase();
            const conf = item.confidence || 0;
            if (type.includes('scarf') && conf < 0.7) return false;
            if (type.includes('skirt') && conf < 0.6) return false;
            if (type === 'clothing item' && conf < 0.5) return false;
            return true;
        });

        const hasHighConfSkirt = validated.some(i => (i.itemType || '').toLowerCase().includes('skirt') && (i.confidence || 0) > 0.7);
        if (!hasHighConfSkirt) {
            validated = validated.map(item =>
                (item.itemType || '').toLowerCase().includes('denim skirt')
                    ? { ...item, itemType: 'Jeans', specificType: 'jeans' }
                    : item
            );
        }

        let final = deduplicateItems(validated);
        final = mergeShoeCategories(final);
        return final.length > 0 ? final : null;
    };

    // ─── Strategy 7: API Server / Gemini Fallback ─────────────────────────────────

    /** Strategy 7: Gemini API via backend proxy — uses server-side key */
    const tryAPIServerFallback = async (frame: string): Promise<DetectedItem[] | null> => {
        try {
            setProgress('Cloud AI analyzing...');
            const cleanImage = frame.replace(/^data:image\/\w+;base64,/, '');

            // Route through backend Gemini proxy (API key stays server-side)
            const prompt = `Analyze this clothing image and output a raw JSON array of objects (no markdown blocks, just the JSON).
For each distinct clothing item found in the image, provide an object with:
- category: general category (tops, bottoms, shoes, accessories)
- subCategory: specific type (e.g. t-shirt, jeans, jacket, sneakers)
- primaryColor: concise main color name
- colorHex: best hex code for the color (e.g. "#000000")
- material: guess the material (cotton, denim, leather, etc)
- pattern: guess the pattern (solid, striped, matching etc)
- confidence: a number from 0.0 to 1.0 (how sure you are)`;

            const response = await axios.post(
                `${API_URL}/api/gemini/analyze-image`,
                { imageBase64: cleanImage, prompt },
                { timeout: 30000 }
            );

            const data = response.data;
            let items: any[] = [];

            if (data.result && Array.isArray(data.result)) {
                items = data.result;
            } else if (data.raw) {
                // Try to parse the raw text response
                try {
                    let text = data.raw.trim();
                    if (text.startsWith('```json')) text = text.slice(7);
                    if (text.startsWith('```')) text = text.slice(3);
                    if (text.endsWith('```')) text = text.slice(0, -3);
                    const parsed = JSON.parse(text);
                    items = Array.isArray(parsed) ? parsed : [parsed];
                } catch {
                    return null;
                }
            }

            if (!items.length) return null;

            const detected: DetectedItem[] = items.map((item: any) => ({
                itemType: formatCategoryName(item.subCategory || item.category || 'Clothing'),
                specificType: item.subCategory || item.category,
                color: item.primaryColor || 'Unknown',
                colorHex: item.colorHex || '#000000',
                material: item.material,
                pattern: item.pattern,
                style: 'Casual',
                description: `${item.primaryColor || ''} ${item.subCategory || item.category || ''}`.trim(),
                position: (item.category || '').toLowerCase().includes('shoe') || (item.category || '').toLowerCase().includes('foot')
                    ? 'feet'
                    : (item.category || '').toLowerCase().includes('bottom') || (item.category || '').toLowerCase().includes('pant')
                        ? 'lower'
                        : 'upper',
                confidence: item.confidence || 0.7,
                confidenceLevel: (item.confidence || 0) > 0.7 ? 'high' : 'medium',
                detectionSources: ['Gemini Vision (Backend Proxy)'],
            }));

            return detected.length > 0 ? detected : null;
        } catch (err) {
            console.warn('[Detection] Gemini proxy fallback failed:', err);
            return null;
        }
    };

    /** Strategy 7.5: AliceVision Classifier — uses our local YOLOv8 + rembg (free) */
    const tryAliceVisionClassifier = async (frame: string): Promise<DetectedItem[] | null> => {
        try {
            setProgress('AliceVision AI classifying...');
            const cleanImage = frame.replace(/^data:image\/\w+;base64,/, '');

            // Try the full /process pipeline first (classify + remove bg + enhance)
            try {
                const processRes = await axios.post(
                    `${ALICEVISION_URL}/process`,
                    { image: cleanImage, mode: 'clean', generate_description: false },
                    { timeout: 60000 }
                );

                if (processRes.data?.success && processRes.data?.classification) {
                    const cls = processRes.data.classification;
                    const item: DetectedItem = {
                        itemType: formatCategoryName(cls.category),
                        specificType: cls.category,
                        color: 'Unknown',
                        colorHex: '#000000',
                        style: 'Casual',
                        description: processRes.data.description || `${cls.category}`,
                        position: getItemPosition(cls.section || ''),
                        confidence: cls.confidence || 0.8,
                        confidenceLevel: (cls.confidence || 0) > 0.7 ? 'high' : 'medium',
                        frameImage: processRes.data.image ? `data:image/jpeg;base64,${processRes.data.image}` : undefined,
                        cutoutImage: processRes.data.cutout ? `data:image/png;base64,${processRes.data.cutout}` : undefined,
                        detectionSources: ['AliceVision YOLOv8', 'rembg', 'studio-enhance'],
                    };
                    return [item];
                }
            } catch {
                // Full pipeline unavailable, try individual classify endpoint
            }

            // Fallback: just classify
            const clsRes = await axios.post(
                `${ALICEVISION_URL}/classify`,
                { image: cleanImage },
                { timeout: 30000 }
            );

            if (clsRes.data?.success) {
                const item: DetectedItem = {
                    itemType: formatCategoryName(clsRes.data.category),
                    specificType: clsRes.data.category,
                    color: 'Unknown',
                    colorHex: '#000000',
                    style: 'Casual',
                    description: clsRes.data.category,
                    position: getItemPosition(clsRes.data.section || ''),
                    confidence: clsRes.data.confidence || 0.8,
                    confidenceLevel: (clsRes.data.confidence || 0) > 0.7 ? 'high' : 'medium',
                    detectionSources: ['AliceVision YOLOv8'],
                };
                return [item];
            }
            return null;
        } catch (err) {
            console.warn('[Detection] AliceVision classifier failed:', err);
            return null;
        }
    };

    /** Strategy 8: Local minimal fallback — returns a generic item so user can correct */
    const tryLocalFallback = async (frame: string): Promise<DetectedItem[] | null> => {
        try {
            setProgress('Using local detection...');
            // Return a minimal detected item so the user can save and manually correct
            const item: DetectedItem = {
                itemType: 'Clothing Item',
                specificType: 'clothing',
                color: 'Unknown',
                colorHex: '#888888',
                style: 'Casual',
                description: 'Detected clothing item (manual review suggested)',
                position: 'upper',
                confidence: 0.4,
                confidenceLevel: 'low' as const,
                detectionSources: ['Local Fallback'],
            };
            return [item];
        } catch {
            return null;
        }
    };

    // ─── Main Analysis Orchestrator ──────────────────────────────────────────────

    const runAnalysis = async (frames: string[]): Promise<DetectedItem[]> => {
        const strategies = [
            () => tryAliceVisionClassifier(frames[0]),  // Primary: AliceVision /process (SAM + enhance)
            () => tryLocalFallback(frames[0]),           // Fallback: generic item for user to correct
        ];

        for (const strategy of strategies) {
            const result = await strategy();
            if (result && result.length > 0) return result;
        }

        throw new Error('AI could not detect clothing items. Please try a clearer video or photo.');
    };

    // ─── Public API ──────────────────────────────────────────────────────────────

    const analyzeVideo = async (videoUri: string): Promise<void> => {
        setAnalyzing(true);
        setResults(null);
        setProgress('Extracting frames from video...');
        try {
            const frames = await extractFrames(videoUri);
            if (frames.length === 0) throw new Error('Could not extract any frames from video.');

            const detectedItems = await runAnalysis(frames);
            if (detectedItems.length === 0) {
                Alert.alert('No Clothing Found', 'AI could not detect clothing items in this video. Try a video with clear clothing visible.');
                return;
            }
            setResults({ detectedItems });
        } catch (error: any) {
            Alert.alert('Analysis Failed', error.message || 'Something went wrong. Please try again.');
        } finally {
            setAnalyzing(false);
            setProgress('');
        }
    };

    const analyzeImage = async (imageUri: string): Promise<void> => {
        setAnalyzing(true);
        setResults(null);
        setProgress('Processing image...');
        try {
            const base64 = await FileSystem.readAsStringAsync(imageUri, { encoding: 'base64' });
            const detectedItems = await runAnalysis([base64]);
            if (detectedItems.length === 0) {
                Alert.alert('No Clothing Found', 'AI could not detect clothing items in this image.');
                return;
            }
            setResults({ detectedItems });
        } catch (error: any) {
            Alert.alert('Analysis Failed', error.message || 'Something went wrong. Please try again.');
        } finally {
            setAnalyzing(false);
            setProgress('');
        }
    };

    return { analyzing, progress, results, analyzeVideo, analyzeImage, reset };
};
