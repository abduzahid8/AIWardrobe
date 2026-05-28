/**
 * Google Gemini 2.0 Flash image generation client for virtual try-on.
 *
 * Much cheaper alternative to FLUX.1-Kontext-dev:
 * - FLUX: ~$0.20-0.50 per 1024x1024 image
 * - Gemini 2.0 Flash: ~$0.04 per image
 *
 * Supports image generation via Imagen 3 model under the hood.
 */

import sharp from 'sharp';
import logger from '../utils/logger.js';
import { loadImageBuffer, toDataUri, stripDataUri } from './tryonShared.js';

const GEMINI_API_BASE = 'https://generativelanguage.googleapis.com/v1beta';

// Prefer Flash for speed/cost, fallback to Pro if needed
// Image generation models (may require special access/whitelisting)
const GEMINI_FLASH_MODEL = 'gemini-flash-latest';
const GEMINI_FLASH_LITE_MODEL = 'gemini-flash-lite-latest';

/**
 * Get Gemini API key from environment or Supabase config.
 */
export async function getGeminiKey() {
  // Check environment first
  const envKey = process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (envKey) return envKey;

  // Fallback to Supabase config
  try {
    const { supabase } = await import('../lib/supabase.js');
    const { data } = await supabase
      .from('app_config')
      .select('value')
      .eq('key', 'gemini_api_key')
      .maybeSingle();
    return data?.value || null;
  } catch (e) {
    return null;
  }
}

/**
 * Build the side-by-side composite for Gemini (same format as FLUX).
 */
export async function buildGeminiComposite(personImage, garmentImage) {
  const [personBuf, garmentBuf] = await Promise.all([
    loadImageBuffer(personImage),
    loadImageBuffer(garmentImage),
  ]);

  const [left, right] = await Promise.all([
    sharp(personBuf)
      .resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
      .png()
      .toBuffer(),
    sharp(garmentBuf)
      .resize(768, 1024, { fit: 'contain', background: { r: 255, g: 255, b: 255, alpha: 1 } })
      .png()
      .toBuffer(),
  ]);

  const composite = await sharp({
    create: { width: 1536, height: 1024, channels: 4, background: { r: 255, g: 255, b: 255, alpha: 1 } },
  })
    .composite([
      { input: left, left: 0, top: 0 },
      { input: right, left: 768, top: 0 },
    ])
    .png()
    .toBuffer();

  return `data:image/png;base64,${composite.toString('base64')}`;
}

/**
 * Build the dressing prompt for Gemini (similar to FLUX but optimized for Gemini).
 */
export function buildGeminiDressingPrompt(label) {
  const garmentNoun =
    label === 'pants'
      ? 'pants / trousers'
      : label === 'shoes'
        ? 'pair of shoes'
        : label === 'layer'
          ? 'outer layer (jacket / coat / cardigan)'
          : 'top (shirt / t-shirt / sweater)';

  const garmentZone =
    label === 'pants'
      ? 'on the hips, thighs, knees, and legs of the mannequin'
      : label === 'shoes'
        ? 'on the feet of the mannequin'
        : label === 'layer'
          ? 'over the existing top, on the shoulders, chest, back, and sleeves of the mannequin'
          : 'on the torso, chest, shoulders, and arms of the mannequin';

  return [
    'Virtual try-on task: Dress a mannequin with a garment.',
    '',
    'INPUT IMAGE DESCRIPTION:',
    'The input image shows TWO halves side-by-side:',
    '- LEFT HALF: A smooth, headless, light-grey fashion mannequin on a clean white studio background.',
    '- RIGHT HALF: The product photo of a single garment that needs to be worn.',
    '',
    'YOUR TASK:',
    `Generate a photorealistic image of the mannequin from the LEFT HALF now wearing the ${garmentNoun} from the RIGHT HALF.`,
    `The garment should be worn ${garmentZone} with natural drape, realistic folds, and proper fit.`,
    '',
    'CRITICAL REQUIREMENTS (MUST FOLLOW):',
    '1. The mannequin must stay IDENTICAL: same pose, same proportions, same light-grey color, same headless silhouette.',
    '2. The white studio background must remain unchanged.',
    '3. Do NOT add human features (face, hair, skin, hands with fingers). Keep it a mannequin.',
    '4. Match the garment color, pattern, and details exactly from the product photo.',
    '5. Output ONLY the dressed mannequin on white background - no split screen, no side-by-side.',
    '',
    'STYLE: Studio fashion photography, soft even lighting, photorealistic, premium e-commerce quality.',
  ].join(' ');
}

/**
 * Call Gemini 2.0 Flash for image generation/editing.
 *
 * @param {object} opts
 * @param {string} opts.imageDataUri - Base64 data URI of input image
 * @param {string} opts.prompt - Text prompt describing the desired output
 * @param {string} [opts.apiKey] - Optional API key override
 * @param {string} [opts.model] - Model to use (default: gemini-2.0-flash-exp-image-generation)
 * @returns {Promise<string>} data:image/png;base64,... result
 */
export async function callGeminiFlash({
  imageDataUri,
  prompt,
  apiKey,
  model = GEMINI_FLASH_MODEL,
}) {
  const key = apiKey || (await getGeminiKey());
  if (!key) {
    throw new Error('Gemini API key is not configured. Set GEMINI_API_KEY or add to app_config.');
  }

  const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${key}`;

  // Prepare the image part
  const base64Data = stripDataUri(imageDataUri);
  const mimeType = imageDataUri.match(/^data:([^;]+);/)?.[1] || 'image/png';

  const body = {
    contents: [
      {
        role: 'user',
        parts: [
          { text: prompt },
          {
            inlineData: {
              mimeType,
              data: base64Data,
            },
          },
        ],
      },
    ],
    generationConfig: {
      responseModalities: ['Text', 'Image'],
      temperature: 0.3, // Lower temperature for more consistent results
      topP: 0.8,
      topK: 40,
    },
  };

  const startTime = Date.now();
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Gemini API failed HTTP ${res.status}: ${errorText}`);
  }

  const data = await res.json();

  // Extract the generated image from response
  const candidates = data.candidates || [];
  if (!candidates.length) {
    throw new Error('Gemini returned no candidates');
  }

  const parts = candidates[0]?.content?.parts || [];

  // Find the image part
  const imagePart = parts.find((p) => p.inlineData?.mimeType?.startsWith('image/'));
  if (imagePart?.inlineData?.data) {
    const mimeType = imagePart.inlineData.mimeType;
    const base64 = imagePart.inlineData.data;
    logger.info(`[geminiClient] Image generated in ${Date.now() - startTime}ms`);
    return `data:${mimeType};base64,${base64}`;
  }

  // If no image, check for text response (might be an error or refusal)
  const textPart = parts.find((p) => p.text);
  if (textPart?.text) {
    throw new Error(`Gemini returned text instead of image: ${textPart.text.slice(0, 200)}`);
  }

  throw new Error('Gemini response did not contain an image');
}

/**
 * Alternative: Use Gemini for text-only response (cost estimation, prompt analysis).
 */
export async function callGeminiText({ prompt, apiKey, model = GEMINI_FLASH_LITE_MODEL }) {
  const key = apiKey || (await getGeminiKey());
  if (!key) {
    throw new Error('Gemini API key is not configured');
  }

  const url = `${GEMINI_API_BASE}/models/${model}:generateContent?key=${key}`;

  const body = {
    contents: [
      {
        role: 'user',
        parts: [{ text: prompt }],
      },
    ],
    generationConfig: {
      temperature: 0.7,
      maxOutputTokens: 1024,
    },
  };

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const errorText = await res.text();
    throw new Error(`Gemini API failed HTTP ${res.status}: ${errorText}`);
  }

  const data = await res.json();
  return data.candidates?.[0]?.content?.parts?.[0]?.text || '';
}

/**
 * Health check for Gemini API.
 */
export async function checkGeminiHealth() {
  try {
    const key = await getGeminiKey();
    if (!key) {
      return { healthy: false, error: 'API key not configured' };
    }

    // Simple text generation to verify API is working
    await callGeminiText({
      prompt: 'Say "OK" if you are working.',
      apiKey: key,
    });

    return { healthy: true, model: GEMINI_FLASH_MODEL };
  } catch (err) {
    return { healthy: false, error: err.message };
  }
}
