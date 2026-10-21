import readinessLogger from '../utils/logger';
/**
 * ExternalAIService - Secure version using Supabase Edge Functions
 * API keys are stored in Supabase, never exposed to mobile app
 */

import { supabase } from '../../lib/supabase';

export interface ProcessingResult {
  failureKind?: 'network' | 'timeout' | 'provider';
  success: boolean;
  imageUrl: string;
  cutoutUrl: string | null;
  enhancedUrl: string | null;
  normalizedUrl: string | null;
  classification: {
    category: string;
    section: string;
    confidence: number;
    attributes: { style: string; confidence?: number; color?: string; material?: string | null };
  } | null;
  description: string | null;
  steps: string[];
  processingTimeMs: number;
}

type ProcessingOperation = 'all' | 'classify' | 'studio_photo';

export const buildImageDataUrl = (imageBase64: string) =>
  imageBase64.startsWith('data:')
    ? imageBase64
    : `data:image/jpeg;base64,${imageBase64}`;

/** Bound the complete SDK call, including a stalled response body or auth refresh. */
async function invokeWithDeadline(body: Record<string, unknown>, timeoutMs: number) {
  const controller = new AbortController();
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => {
      const error = new Error('AI request timeout');
      error.name = 'TimeoutError';
      // Reject first so a transport-specific abort error cannot obscure the deadline.
      reject(error);
      controller.abort();
    }, timeoutMs);
  });
  try {
    return await Promise.race([
      Promise.resolve().then(() => supabase.functions.invoke('ai-process', { body, signal: controller.signal })),
      deadline,
    ]);
  } finally { if (timer !== undefined) clearTimeout(timer); }
}

async function invokeProcessingOperation(
  imageBase64: string,
  operation: ProcessingOperation,
  steps: string[],
  fallbackSteps: string[] = ['error_fallback']
): Promise<ProcessingResult> {
  const startTime = Date.now();
  const operationName = operation === 'studio_photo' ? 'AI Studio Photo' : operation;

  if (__DEV__) readinessLogger.debug(String(`[ExternalAI] Starting ${operationName}...`), []);

  try {
    const { data, error } = await invokeWithDeadline({ image: imageBase64, operation }, operation === 'all' ? 180_000 : 60_000);

    if (error) {
      console.error(`[ExternalAI] Edge Function error for ${operationName}:`, error);
      throw error;
    }

    if (__DEV__) readinessLogger.debug(String(`[ExternalAI] ${operationName} response:`), [{
      success: data?.success,
      hasCutoutUrl: !!data?.cutoutUrl,
      hasClassification: !!data?.classification,
      hasEnhancedUrl: !!data?.enhancedUrl,
      error: data?.error,
      localCutoutError: data?.localCutoutError,
      nvidiaError: data?._nvidiaError,
    }]);

    if (!data?.success) {
      throw new Error(data?.error || 'AI processing failed');
    }

    const processingTimeMs = Date.now() - startTime;
    
    // Always return the original uploaded image — the AI only classifies the
    // clothing type; it never modifies or removes the background.
    const finalImage = buildImageDataUrl(imageBase64);

    return {
      success: true,
      imageUrl: finalImage,
      cutoutUrl: data.cutoutUrl || null,
      enhancedUrl: data.enhancedUrl || null,
      normalizedUrl: data.normalizedUrl || null,
      classification: data.classification || null,
      description: data.description || null,
      steps,
      processingTimeMs,
    };
  } catch (error: any) {
    console.error(`[ExternalAI] ${operationName} processing failed:`, error?.message || error);

    return {
      success: false,
      failureKind: /timeout|abort/i.test(error?.name || '') || /timeout/i.test(error?.message || '') ? 'timeout'
        : error?.name === 'FunctionsFetchError' || /network/i.test(error?.message || '') ? 'network' : 'provider',
      imageUrl: buildImageDataUrl(imageBase64),  // original photo is always preserved
      cutoutUrl: null,
      enhancedUrl: null,
      normalizedUrl: null,
      classification: null,
      description: null,
      steps: fallbackSteps,
      processingTimeMs: Date.now() - startTime,
    };
  }
}

export const ExternalAIService = {
  async processVideoFrames(frames: string[]): Promise<ProcessingResult[]> {
    if (frames.length < 1 || frames.length > 3 || frames.some(frame => !frame.trim())) {
      throw new Error('Invalid video frames');
    }
    const started = Date.now();
    const { data, error } = await invokeWithDeadline({ operation: 'classify_frames', frames }, frames.length * 45_000 + 15_000);
    if (error) throw error;
    if (!data?.success || !Array.isArray(data.frames) || data.frames.length !== frames.length) {
      throw new Error('Incomplete video analysis');
    }
    return data.frames.map((frame: any, index: number) => {
      if (frame?.frameIndex !== index || !frame.classification) throw new Error('Invalid video analysis');
      return {
        success: true, imageUrl: buildImageDataUrl(frames[index]), cutoutUrl: null, enhancedUrl: null,
        normalizedUrl: null, classification: frame.classification, description: frame.description ?? null,
        steps: ['nvidia_classify'], processingTimeMs: Date.now() - started,
      };
    });
  },
  /**
   * Process clothing image via Supabase Edge Function
   * API keys stay secure in Supabase, never reach mobile app
   */
  async processClothingImage(imageBase64: string): Promise<ProcessingResult> {
    return invokeProcessingOperation(
      imageBase64,
      'all',
      ['nvidia_classify', 'replicate_normalize_angle', 'replicate_remove_bg', 'replicate_iron_enhance']
    );
  },

  /**
   * Upload photo flow: classify the clothing type only.
   * The original photo is kept as-is — no background removal is performed.
   * Classification result is used to correctly categorise the item in the wardrobe.
   */
  async processStudioPhoto(imageBase64: string): Promise<ProcessingResult> {
    return invokeProcessingOperation(
      imageBase64,
      'classify',            // classify-only — no bg removal, no cutout
      ['nvidia_classify'],
    );
  },

  /**
   * Quick classification only (faster, no background removal)
   */
  async classifyOnly(imageBase64: string): Promise<ProcessingResult> {
    return invokeProcessingOperation(
      imageBase64,
      'classify',
      ['nvidia_classify'],
      ['error']
    );
  },
};

export default ExternalAIService;
