/**
 * ExternalAIService - Secure version using Supabase Edge Functions
 * API keys are stored in Supabase, never exposed to mobile app
 */

import { supabase } from '../lib/supabase';

export interface ProcessingResult {
  success: boolean;
  imageUrl: string;
  cutoutUrl: string | null;
  enhancedUrl: string | null;
  normalizedUrl: string | null;
  classification: {
    category: string;
    section: string;
    confidence: number;
    attributes: { style: string; confidence: number };
  } | null;
  description: string | null;
  steps: string[];
  processingTimeMs: number;
}

export const ExternalAIService = {
  /**
   * Process clothing image via Supabase Edge Function
   * API keys stay secure in Supabase, never reach mobile app
   */
  async processClothingImage(imageBase64: string): Promise<ProcessingResult> {
    const startTime = Date.now();
    const steps: string[] = [];

    try {
      // Call Supabase Edge Function (keys are secure on server side)
      const { data, error } = await supabase.functions.invoke('ai-process', {
        body: { 
          image: imageBase64,
          operation: 'all' // classify + describe + remove_bg
        },
      });

      if (error) {
        console.error('[ExternalAI] Edge Function error:', error);
        throw error;
      }

      if (!data?.success) {
        throw new Error(data?.error || 'AI processing failed');
      }

      const processingTimeMs = Date.now() - startTime;

      // Build enhanced image URL (combine original + cutout if available)
      const base64WithPrefix = imageBase64.startsWith('data:')
        ? imageBase64
        : `data:image/jpeg;base64,${imageBase64}`;
      const finalImage = data.cutoutUrl || base64WithPrefix;

      return {
        success: true,
        imageUrl: finalImage,
        cutoutUrl: data.cutoutUrl || null,
        enhancedUrl: data.enhancedUrl || null,
        normalizedUrl: data.normalizedUrl || null,
        classification: data.classification || null,
        description: data.description || null,
        steps: ['nvidia_classify', 'replicate_normalize_angle', 'replicate_remove_bg', 'replicate_iron_enhance'],
        processingTimeMs,
      };
    } catch (error) {
      console.error('[ExternalAI] Processing failed:', error);
      
      // Return fallback on error
      return {
        success: false,
        imageUrl: imageBase64,
        cutoutUrl: null,
        enhancedUrl: null,
        normalizedUrl: null,
        classification: null,
        description: null,
        steps: ['error_fallback'],
        processingTimeMs: Date.now() - startTime,
      };
    }
  },

  /**
   * Quick classification only (faster, no background removal)
   */
  async classifyOnly(imageBase64: string): Promise<ProcessingResult> {
    const startTime = Date.now();

    try {
      const { data, error } = await supabase.functions.invoke('ai-process', {
        body: { 
          image: imageBase64,
          operation: 'classify' // Only classification
        },
      });

      if (error || !data?.success) {
        throw error || new Error('Classification failed');
      }

      return {
        success: true,
        imageUrl: imageBase64,
        cutoutUrl: null,
        enhancedUrl: null,
        normalizedUrl: null,
        classification: data.classification,
        description: data.description,
        steps: ['nvidia_classify'],
        processingTimeMs: Date.now() - startTime,
      };
    } catch (error) {
      return {
        success: false,
        imageUrl: imageBase64,
        cutoutUrl: null,
        enhancedUrl: null,
        normalizedUrl: null,
        classification: null,
        description: null,
        steps: ['error'],
        processingTimeMs: Date.now() - startTime,
      };
    }
  },
};

export default ExternalAIService;
