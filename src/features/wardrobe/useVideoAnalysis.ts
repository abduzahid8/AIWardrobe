/**
 * useVideoAnalysis — Serverless version using External AI APIs directly
 * No backend server needed - calls HuggingFace/Replicate from mobile
 */

import { useState } from 'react';
import { Platform } from 'react-native';
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as FileSystem from 'expo-file-system/legacy';
import { Alert } from 'react-native';
import { ExternalAIService } from '../../services/externalAIService';
import useUploadQueueStore from '../../store/uploadQueueStore';
import { createLogger } from '../../utils/logger';
import { useTranslation } from 'react-i18next';

const logger = createLogger('useVideoAnalysis');

const FRAME_TIME_POINTS = [0, 1000, 2000]; // ms

export interface DetectedItem {
  itemType: string;
  specificType?: string;
  color: string;
  colorHex?: string;
  material?: string;
  pattern?: string;
  style: string;
  description: string;
  productDescription?: string;
  position: 'upper' | 'lower' | 'full' | 'feet' | 'accessory';
  confidence: number;
  confidenceLevel: 'high' | 'medium' | 'low';
  frameImage?: string;
  cutoutImage?: string;
  detectionSources: string[];
  outfitId?: number;
}

export interface AnalysisResult {
  detectedItems: DetectedItem[];
}

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

  // Extract frames from video
  const extractFrames = async (videoUri: string): Promise<string[]> => {
    const frames: string[] = [];
    
    if (Platform.OS === 'web') {
      logger.warn('Web: Video frame extraction not fully implemented');
      return [];
    }
    
    for (const time of FRAME_TIME_POINTS) {
      try {
        setProgress(`Extracting frame ${frames.length + 1}/${FRAME_TIME_POINTS.length}...`);
        const { uri } = await VideoThumbnails.getThumbnailAsync(videoUri, { time, quality: 0.9 });
        const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
        frames.push(base64);
      } catch (error) {
        console.warn(`Failed to extract frame at ${time}ms:`, error);
      }
    }
    
    return frames;
  };

  // Analyze a single frame using External AI
  const analyzeFrame = async (base64Image: string): Promise<DetectedItem[]> => {
    setProgress('Analyzing with AI...');
    
    try {
      const result = await ExternalAIService.processClothingImage(base64Image);
      
      if (!result.success || !result.classification) {
        // Return fallback item
        return [{
          itemType: 'Clothing Item',
          color: 'Unknown',
          style: 'Casual',
          description: 'Detected clothing item',
          position: 'upper',
          confidence: 0.5,
          confidenceLevel: 'low',
          detectionSources: ['ExternalAI Fallback'],
          frameImage: `data:image/jpeg;base64,${base64Image}`,
        }];
      }

      const item: DetectedItem = {
        itemType: result.classification.category,
        specificType: result.classification.category,
        color: 'Unknown', // Color detection would need additional model
        colorHex: '#888888',
        style: result.classification.attributes.style,
        description: result.description || `${result.classification.category}`,
        position: result.classification.section as any,
        confidence: result.classification.confidence,
        confidenceLevel: result.classification.confidence > 0.8 ? 'high' : 
                        result.classification.confidence > 0.5 ? 'medium' : 'low',
        detectionSources: result.steps,
        frameImage: result.imageUrl,
        cutoutImage: result.cutoutUrl || undefined,
      };

      return [item];
    } catch (error) {
      console.error('Analysis error:', error);
      return [{
        itemType: 'Clothing Item',
        color: 'Unknown',
        style: 'Casual',
        description: 'Error during analysis',
        position: 'upper',
        confidence: 0.3,
        confidenceLevel: 'low',
        detectionSources: ['Error Fallback'],
        frameImage: `data:image/jpeg;base64,${base64Image}`,
      }];
    }
  };

  // Analyze video
  const analyzeVideo = async (videoUri: string): Promise<void> => {
    logger.info('analyzeVideo called', { videoUri });
    setAnalyzing(true);
    setResults(null);
    setProgress('Extracting frames...');
    
    try {
      const frames = await extractFrames(videoUri);
      
      if (frames.length === 0) {
        const { t } = useTranslation();
        Alert.alert(
          t('videoAnalysis.videoProcessingLimited'),
          t('videoAnalysis.videoProcessingLimitedMessage'),
          [{ text: t('common.ok') }]
        );
        setResults({
          detectedItems: [{
            itemType: 'Clothing Item',
            color: 'Unknown',
            style: 'Casual',
            description: 'Add details manually in the next step',
            position: 'upper',
            confidence: 0.3,
            confidenceLevel: 'low',
            detectionSources: ['Manual Fallback'],
            frameImage: undefined,
            cutoutImage: undefined,
          }],
        });
        return;
      }

      // Analyze the best frame (first one for now)
      const detectedItems = await analyzeFrame(frames[0]);
      
      if (detectedItems.length === 0) {
        const { t } = useTranslation();
        Alert.alert(t('videoAnalysis.noClothingFound'), t('videoAnalysis.noClothingFoundVideo'));
        return;
      }

      setResults({ detectedItems });
    } catch (error: any) {
      console.error('analyzeVideo error:', error);
      const isNetworkError = error.message?.toLowerCase().includes('network') || 
                            error.message?.toLowerCase().includes('timeout');

      if (isNetworkError) {
        useUploadQueueStore.getState().addUpload(videoUri, 'video');
        const { t } = useTranslation();
        Alert.alert(t('videoAnalysis.savedOffline'), t('videoAnalysis.savedOfflineVideo'));
      } else {
        const { t } = useTranslation();
        Alert.alert(t('videoAnalysis.analysisFailed'), error.message || t('videoAnalysis.somethingWrong'));
      }
    } finally {
      setAnalyzing(false);
      setProgress('');
    }
  };

  // Analyze image
  const analyzeImage = async (imageUri: string): Promise<void> => {
    setAnalyzing(true);
    setResults(null);
    setProgress('Processing image...');
    
    try {
      let base64: string;
      
      if (Platform.OS === 'web') {
        base64 = imageUri; // Already base64 from web file picker
      } else {
        base64 = await FileSystem.readAsStringAsync(imageUri, { encoding: 'base64' });
      }
      
      const detectedItems = await analyzeFrame(base64);
      
      if (detectedItems.length === 0) {
        const { t } = useTranslation();
        Alert.alert(t('videoAnalysis.noClothingFound'), t('videoAnalysis.noClothingFoundImage'));
        return;
      }

      setResults({ detectedItems });
    } catch (error: any) {
      const isNetworkError = error.message?.toLowerCase().includes('network') || 
                            error.message?.toLowerCase().includes('timeout');

      if (isNetworkError) {
        useUploadQueueStore.getState().addUpload(imageUri, 'image');
        const { t } = useTranslation();
        Alert.alert(t('videoAnalysis.savedOffline'), t('videoAnalysis.savedOfflinePhoto'));
      } else {
        const { t } = useTranslation();
        Alert.alert(t('videoAnalysis.analysisFailed'), error.message || t('videoAnalysis.somethingWrong'));
      }
    } finally {
      setAnalyzing(false);
      setProgress('');
    }
  };

  return { analyzing, progress, results, analyzeVideo, analyzeImage, reset };
};

export default useVideoAnalysis;
