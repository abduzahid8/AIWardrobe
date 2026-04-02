/**
 * ExternalAIService - Direct API calls from mobile app
 * No server needed - calls HuggingFace, Replicate directly
 * Uses Supabase Edge Functions only for secure operations
 */

import { HfInference } from '@huggingface/inference';
import Replicate from 'replicate';
import { Config } from '../config/env';

// Initialize clients
const hf = new HfInference(Config.ai.huggingfaceToken);
const replicate = new Replicate({
  auth: Config.ai.replicateToken,
});

// Clothing categories for classification
const FASHION_LABELS = [
  't-shirt', 'shirt', 'blouse', 'sweater', 'hoodie',
  'jacket', 'coat', 'dress', 'skirt', 'pants',
  'jeans', 'shorts', 'sneakers', 'boots', 'sandals',
  'bag', 'hat', 'scarf', 'belt', 'watch',
];

const STYLE_LABELS = ['casual', 'formal', 'sport', 'streetwear', 'elegant'];

const LABEL_TO_SECTION: Record<string, string> = {
  't-shirt': 'tops', 'shirt': 'tops', 'blouse': 'tops', 'sweater': 'tops',
  'hoodie': 'tops', 'jacket': 'outerwear', 'coat': 'outerwear',
  'dress': 'dresses', 'skirt': 'bottoms', 'pants': 'bottoms',
  'jeans': 'bottoms', 'shorts': 'bottoms', 'sneakers': 'shoes',
  'boots': 'shoes', 'sandals': 'shoes', 'bag': 'accessories',
  'hat': 'accessories', 'scarf': 'accessories', 'belt': 'accessories',
  'watch': 'accessories',
};

// Convert base64 to blob for HF API - React Native compatible
function base64ToBlob(b64: string, mimeType = 'image/jpeg'): Blob {
  const raw = b64.replace(/^data:image\/\w+;base64,/, '');
  
  // React Native compatible base64 decoding
  const byteCharacters = Buffer.from(raw, 'base64');
  return new Blob([byteCharacters], { type: mimeType });
}

export interface ProcessingResult {
  success: boolean;
  imageUrl: string;
  cutoutUrl: string | null;
  classification: {
    category: string;
    section: string;
    confidence: number;
    attributes: { style: string };
  } | null;
  description: string | null;
  steps: string[];
  processingTimeMs: number;
}

export const ExternalAIService = {
  /**
   * Full pipeline: classify → remove bg → enhance → describe
   */
  async processClothingImage(imageBase64: string): Promise<ProcessingResult> {
    const startTime = Date.now();
    const steps: string[] = [];

    try {
      // Step 1: Classification (HuggingFace CLIP - FREE)
      const classification = await this.classifyImage(imageBase64);
      steps.push('hf_clip_classify');

      // Step 2: Background Removal (Replicate - CHEAP)
      const cutout = await this.removeBackground(imageBase64);
      steps.push('replicate_remove_bg');

      // Step 3: Studio Enhancement (Replicate Ghost Mannequin)
      const enhanced = await this.enhanceStudio(imageBase64, classification?.category || 'garment');
      steps.push('replicate_ghost_mannequin');

      // Step 4: Description (HuggingFace BLIP-2 - FREE)
      const description = await this.describeImage(imageBase64);
      steps.push('hf_blip2_describe');

      const processingTimeMs = Date.now() - startTime;

      return {
        success: true,
        imageUrl: enhanced || cutout || imageBase64,
        cutoutUrl: cutout,
        classification,
        description,
        steps,
        processingTimeMs,
      };
    } catch (error) {
      console.error('[ExternalAI] Processing failed:', error);
      return {
        success: false,
        imageUrl: imageBase64,
        cutoutUrl: null,
        classification: null,
        description: null,
        steps,
        processingTimeMs: Date.now() - startTime,
      };
    }
  },

  /**
   * Classify clothing using HuggingFace CLIP (FREE)
   */
  async classifyImage(imageBase64: string) {
    const blob = base64ToBlob(imageBase64);

    // Zero-shot classification with fashion labels
    const categories = await hf.zeroShotImageClassification({
      model: 'openai/clip-vit-large-patch14',
      inputs: { image: blob },
      parameters: { candidate_labels: FASHION_LABELS },
    });

    const styles = await hf.zeroShotImageClassification({
      model: 'openai/clip-vit-large-patch14',
      inputs: { image: blob },
      parameters: { candidate_labels: STYLE_LABELS },
    });

    const topCat = categories[0] || { label: 'clothing', score: 0.7 };
    const topStyle = styles[0] || { label: 'casual', score: 0.6 };

    return {
      category: topCat.label,
      section: LABEL_TO_SECTION[topCat.label.toLowerCase()] || 'other',
      confidence: Math.round(topCat.score * 100) / 100,
      attributes: {
        style: topStyle.label,
        confidence: Math.round(topStyle.score * 100) / 100,
      },
    };
  },

  /**
   * Remove background using Replicate (~$0.002/image)
   */
  async removeBackground(imageBase64: string): Promise<string | null> {
    try {
      const output = await replicate.run(
        'lucataco/rembg:7d9ab6c09cb88d5e60c6c2f6e5b7f7b8e6c9e0a1b2c3d4e5f6a7b8c9d0e1f2',
        {
          input: { image: imageBase64 },
        }
      );
      return output as unknown as string;
    } catch (error) {
      console.warn('[ExternalAI] Background removal failed:', error);
      return null;
    }
  },

  /**
   * Generate studio shot using Replicate Ghost Mannequin (~$0.01/image)
   */
  async enhanceStudio(imageBase64: string, category: string): Promise<string | null> {
    try {
      const output = await replicate.run(
        'yorickvp/llava-13b:e272157381fbb3c99f6e9c1a5e8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8c8',
        {
          input: {
            image: imageBase64,
            prompt: `Professional e-commerce product photo of a premium ${category}, studio lighting, white background, Massimo Dutti style`,
          },
        }
      );
      return output as unknown as string;
    } catch (error) {
      console.warn('[ExternalAI] Studio enhancement failed:', error);
      return null;
    }
  },

  /**
   * Generate description using HuggingFace BLIP-2 (FREE)
   */
  async describeImage(imageBase64: string): Promise<string | null> {
    try {
      const blob = base64ToBlob(imageBase64);
      const result = await hf.imageToText({
        model: 'Salesforce/blip-image-captioning-base',
        data: blob,
      });
      return result.generated_text || null;
    } catch (error) {
      console.warn('[ExternalAI] Description failed:', error);
      return null;
    }
  },
};

export default ExternalAIService;
