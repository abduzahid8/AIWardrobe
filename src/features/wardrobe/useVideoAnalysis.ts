/**
 * useVideoAnalysis — image/video frame classification through authenticated Edge Functions.
 */

import { useEffect, useRef, useState } from 'react';
import { authEvents } from '../../../store/authEvents';
import { Platform } from 'react-native';
import * as VideoThumbnails from 'expo-video-thumbnails';
import * as FileSystem from 'expo-file-system/legacy';
import { Alert } from 'react-native';
import { ExternalAIService, type ProcessingResult } from '../../services/externalAIService';
import useUploadQueueStore, { flushUploadQueuePersistence } from '../../store/uploadQueueStore';
import { commitScanReview } from '../../services/scanReviews';
import { readVideoSampleTimes, readAccurateVideoFrames } from '../../services/videoFrameSampling';
import { createLogger } from '../../utils/logger';
import { useTranslation } from 'react-i18next';

const logger = createLogger('useVideoAnalysis');


export interface DetectedItem {
  reviewItemId?: string;
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
  reviewId?: string;
  sourceQueueId?: string;
  detectedItems: DetectedItem[];
}

export interface ScanOptions { queueOnFailure?: boolean; silent?: boolean; queueId?: string }

export interface UseVideoAnalysisReturn {
  analyzing: boolean;
  progress: string;
  results: AnalysisResult | null;
  analyzeVideo: (videoUri: string, options?: ScanOptions) => Promise<boolean>;
  analyzeImage: (imageUri: string, options?: ScanOptions) => Promise<boolean>;
  reset: () => void;
  restoreReview: (review: AnalysisResult) => void;
}

export const useVideoAnalysis = (): UseVideoAnalysisReturn => {
  const { t } = useTranslation();
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState('');
  const [results, setResults] = useState<AnalysisResult | null>(null);

  const requestVersion = useRef(0);
  const pendingMetadata = useRef<AbortController | null>(null);
  const mounted = useRef(true);

  const reset = () => {
    requestVersion.current++;
    pendingMetadata.current?.abort();
    setAnalyzing(false);
    setProgress('');
    setResults(null);
  };

  useEffect(() => {
    mounted.current = true;
    let identityRevision = authEvents.getRevision();
    const onIdentityChange = () => {
      const revision = authEvents.getRevision();
      if (identityRevision === revision) return;
      identityRevision = revision;
      reset();
    };
    const offLogin = authEvents.onLogin(onIdentityChange);
    const offLogout = authEvents.onLogout(onIdentityChange);
    return () => {
      mounted.current = false;
      pendingMetadata.current?.abort();
      requestVersion.current++;
      offLogin(); offLogout();
    };
  }, []);

  const beginRequest = () => {
    pendingMetadata.current?.abort();
    const controller = new AbortController();
    pendingMetadata.current = controller;
    const version = ++requestVersion.current;
    const revision = authEvents.getRevision();
    return Object.assign(() => mounted.current && version === requestVersion.current && revision === authEvents.getRevision(), { signal: controller.signal });
  };

  // Extract frames from video
  const extractFrames = async (videoUri: string, current: () => boolean, signal: AbortSignal): Promise<string[]> => {
    const frames: string[] = [];
    
    if (Platform.OS === 'web') {
      logger.warn('Web: Video frame extraction not fully implemented');
      return [];
    }
    
    if (Platform.OS === 'android') {
      const samples = await readAccurateVideoFrames(videoUri, signal);
      return current() ? samples.map(sample => sample.base64) : [];
    }
    const timePoints = await readVideoSampleTimes(videoUri, signal);
    if (!current()) return [];
    for (const time of timePoints) {
      if (!current()) return [];
      let thumbnailUri: string | undefined;
      try {
        setProgress(t('wardrobeVideo.extractingFrame', { index: frames.length + 1, count: timePoints.length }));
        const { uri } = await VideoThumbnails.getThumbnailAsync(videoUri, { time, quality: 0.9 });
        thumbnailUri = uri;
        if (!current()) return [];
        const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
        if (!current()) return [];
        frames.push(base64);
      } catch (error) {
        console.warn(`Failed to extract frame at ${time}ms:`, error);
      } finally {
        // Results retain bytes, never this temporary file. This also runs when
        // cancellation wins while thumbnail extraction or reading is pending.
        if (thumbnailUri) {
          try { await FileSystem.deleteAsync(thumbnailUri, { idempotent: true }); }
          catch { logger.warn('Unable to remove temporary scan thumbnail'); }
        }
      }
    }
    
    return frames;
  };

  // Analyze a single frame using External AI
  const mapFrameResult = (result: ProcessingResult): DetectedItem[] => {
      
    if (!result.success || !result.classification) {
      throw new Error(result.failureKind === 'network' || result.failureKind === 'timeout' ? 'network' : t('videoAnalysis.somethingWrong'));
    }
    const positions: Record<string, DetectedItem['position']> = {
      tops: 'upper', outerwear: 'upper', bottoms: 'lower', dresses: 'full',
      shoes: 'feet', accessories: 'accessory',
    };

    const position = positions[result.classification.section];
    if (!position) throw new Error(t('videoAnalysis.somethingWrong'));

    const item: DetectedItem = {
      itemType: result.classification.category,
      specificType: result.classification.category,
      color: result.classification.attributes.color || '',
      material: result.classification.attributes.material ?? undefined,
      style: result.classification.attributes.style,
      description: result.description || `${result.classification.category}`,
      position,
      confidence: result.classification.confidence,
      confidenceLevel: result.classification.confidence > 0.8 ? 'high' :
                      result.classification.confidence > 0.5 ? 'medium' : 'low',
      detectionSources: result.steps,
      frameImage: result.imageUrl,
      cutoutImage: result.cutoutUrl || undefined,
    };

    return [item];
  };

  const analyzeFrame = async (base64Image: string, current: () => boolean): Promise<DetectedItem[]> => {
    if (!current()) return [];
    setProgress('Analyzing with AI...');
    const result = await ExternalAIService.classifyOnly(base64Image);
    return current() ? mapFrameResult(result) : [];
  };

  const queueForRetry = async (uri: string, type: 'image' | 'video', current: () => boolean): Promise<boolean> => {
    const owner = authEvents.getUserId();
    if (!owner || !current()) return false;
    let staged = uri;
    let queued = false;
    try {
      if (Platform.OS !== 'web') {
        const { stagePendingScanMedia } = require('../../services/pendingScanMedia') as typeof import('../../services/pendingScanMedia');
        staged = await stagePendingScanMedia(owner, uri, type);
      }
      if (!current()) return false;
      queued = useUploadQueueStore.getState().addUpload(staged, type);
      if (!queued) return false;
      await flushUploadQueuePersistence();
      return current();
    } catch { return false; }
    finally {
      if (Platform.OS !== 'web' && !queued && staged !== uri) {
        const media = require('../../services/pendingScanMedia') as typeof import('../../services/pendingScanMedia');
        const referenced = useUploadQueueStore.getState().pendingUploads.some(entry => {
          if (entry.ownerId !== owner) return false;
          try { return media.resolvePendingScanMedia(entry.uri, owner) === staged; } catch { return false; }
        });
        if (!referenced) await media.removePendingScanMedia(owner, staged, () => useUploadQueueStore.getState().pendingUploads.filter(entry => entry.ownerId === owner).map(entry => entry.uri)).catch(() => undefined);
      }
    }
  };

  // Analyze video
  const analyzeVideo = async (videoUri: string, options: ScanOptions = {}): Promise<boolean> => {
    if (!mounted.current) return false;
    const current = beginRequest();
    setAnalyzing(true);
    setResults(null);
    setProgress('Extracting frames...');
    
    try {
      const frames = await extractFrames(videoUri, current, current.signal);
      if (!current()) return false;
      
      if (frames.length === 0) {
        if (!options.silent) Alert.alert(
          t('videoAnalysis.videoProcessingLimited'),
          t('videoAnalysis.videoProcessingLimitedMessage'),
          [{ text: t('common.ok') }]
        );
        return false;
      }

      setProgress(t('wardrobeVideo.analyzingVideo'));
      const frameResults = await ExternalAIService.processVideoFrames(frames);
      if (!current()) return false;
      const detectedItems = frameResults.flatMap((result, frameIndex) =>
        mapFrameResult(result).map(item => ({ ...item, outfitId: frameIndex + 1 })));

      
      if (detectedItems.length === 0) {
        if (!options.silent) Alert.alert(t('videoAnalysis.noClothingFound'), t('videoAnalysis.noClothingFoundVideo'));
        return false;
      }

      const owner = authEvents.getUserId();
      const review = Platform.OS === 'web' ? { detectedItems } : await commitScanReview(owner || '', { detectedItems }, options.queueId);
      if (!current()) return false;
      setResults(review);
      return true;
    } catch (error: any) {
      if (!current()) return false;
      console.error('analyzeVideo error:', error);
      const isNetworkError = error.name === 'FunctionsFetchError' || error.name === 'AbortError' || error.message?.toLowerCase().includes('network') || 
                            error.message?.toLowerCase().includes('timeout');

      if (isNetworkError && options.queueOnFailure !== false) {
        const queued = await queueForRetry(videoUri, 'video', current);
        if (!current()) return false;
        if (!options.silent) Alert.alert(t(queued ? 'videoAnalysis.savedOffline' : 'videoAnalysis.analysisFailed'),
          t(queued ? 'videoAnalysis.savedOfflineVideo' : 'videoAnalysis.somethingWrong'));
      } else {
        if (!options.silent) Alert.alert(t('videoAnalysis.analysisFailed'), error.message || t('videoAnalysis.somethingWrong'));
      }
      return false;
    } finally {
      if (current()) { setAnalyzing(false); setProgress(''); }
    }
  };

  // Analyze image
  const analyzeImage = async (imageUri: string, options: ScanOptions = {}): Promise<boolean> => {
    if (!mounted.current) return false;
    const current = beginRequest();
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
      
      if (!current()) return false;
      const detectedItems = await analyzeFrame(base64, current);
      if (!current()) return false;
      
      if (detectedItems.length === 0) {
        if (!options.silent) Alert.alert(t('videoAnalysis.noClothingFound'), t('videoAnalysis.noClothingFoundImage'));
        return false;
      }

      const owner = authEvents.getUserId();
      const review = Platform.OS === 'web' ? { detectedItems } : await commitScanReview(owner || '', { detectedItems }, options.queueId);
      if (!current()) return false;
      setResults(review);
      return true;
    } catch (error: any) {
      if (!current()) return false;
      const isNetworkError = error.name === 'FunctionsFetchError' || error.name === 'AbortError' || error.message?.toLowerCase().includes('network') || 
                            error.message?.toLowerCase().includes('timeout');

      if (isNetworkError && options.queueOnFailure !== false) {
        const queued = await queueForRetry(imageUri, 'image', current);
        if (!current()) return false;
        if (!options.silent) Alert.alert(t(queued ? 'videoAnalysis.savedOffline' : 'videoAnalysis.analysisFailed'),
          t(queued ? 'videoAnalysis.savedOfflinePhoto' : 'videoAnalysis.somethingWrong'));
      } else {
        if (!options.silent) Alert.alert(t('videoAnalysis.analysisFailed'), error.message || t('videoAnalysis.somethingWrong'));
      }
      return false;
    } finally {
      if (current()) { setAnalyzing(false); setProgress(''); }
    }
  };

  const restoreReview = (review: AnalysisResult) => { reset(); setResults(review); };
  return { analyzing, progress, results, analyzeVideo, analyzeImage, reset, restoreReview };
};

export default useVideoAnalysis;
