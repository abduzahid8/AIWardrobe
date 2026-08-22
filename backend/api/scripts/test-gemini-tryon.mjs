#!/usr/bin/env node
/**
 * Test script for Gemini 2.0 Flash try-on
 *
 * This tests the new Gemini-based try-on endpoint which is ~6-12x cheaper than FLUX:
 * - FLUX.1-Kontext-dev: ~$0.20-0.50 per 1024x1024 image
 * - Gemini 2.0 Flash: ~$0.04 per image
 *
 * Usage:
 *   node backend/api/scripts/test-gemini-tryon.mjs [path-to-mannequin] [path-to-garment]
 *
 * Example:
 *   node backend/api/scripts/test-gemini-tryon.mjs backend/api/blank_mannequin.jpg backend/api/data/shirt.png
 */

import fs from 'fs/promises';
import path from 'path';

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const API_BASE_URL = process.env.API_URL || 'http://localhost:3000';

async function fileToDataUri(filePath) {
  const buffer = await fs.readFile(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const mimeType = ext === '.png' ? 'image/png' : ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' : 'image/png';
  return `data:${mimeType};base64,${buffer.toString('base64')}`;
}

async function testDirectGemini(mannequinPath, garmentPath) {
  console.log('\n🧪 Testing DIRECT Gemini API (no local server needed)...\n');

  const mannequinUri = await fileToDataUri(mannequinPath);
  const garmentUri = await fileToDataUri(garmentPath);

  // Build side-by-side composite
  console.log('Building composite...');
  const { buildGeminiComposite, buildGeminiDressingPrompt } = await import('../services/geminiClient.js');

  const composite = await buildGeminiComposite(mannequinUri, garmentUri);
  console.log(`✓ Composite built: ${composite.slice(0, 50)}... (${Math.round(composite.length / 1024)}KB)`);

  // Call Gemini
  console.log('\nCalling Gemini 2.0 Flash...');
  const startTime = Date.now();

  const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${GEMINI_API_KEY}`;

  const base64Data = composite.replace(/^data:image\/[^;]+;base64,/, '');

  const body = {
    contents: [
      {
        role: 'user',
        parts: [
          { text: buildGeminiDressingPrompt('top') },
          {
            inlineData: {
              mimeType: 'image/png',
              data: base64Data,
            },
          },
        ],
      },
    ],
    generationConfig: {
      responseModalities: ['Text', 'Image'],
      temperature: 0.3,
      topP: 0.8,
      topK: 40,
    },
  };

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const error = await res.text();
      throw new Error(`HTTP ${res.status}: ${error}`);
    }

    const data = await res.json();
    const elapsed = Date.now() - startTime;

    // Extract image
    const parts = data.candidates?.[0]?.content?.parts || [];
    const imagePart = parts.find((p) => p.inlineData?.mimeType?.startsWith('image/'));

    if (imagePart?.inlineData?.data) {
      const resultUri = `data:${imagePart.inlineData.mimeType};base64,${imagePart.inlineData.data}`;
      console.log(`✓ Image generated in ${elapsed}ms`);
      console.log(`✓ Result size: ${Math.round(resultUri.length / 1024)}KB`);

      // Save to file
      const outputPath = `/tmp/gemini-tryon-result-${Date.now()}.png`;
      const buffer = Buffer.from(imagePart.inlineData.data, 'base64');
      await fs.writeFile(outputPath, buffer);
      console.log(`✓ Saved to: ${outputPath}`);

      return { success: true, outputPath, elapsed };
    }

    // Check for text response (refusal/error)
    const textPart = parts.find((p) => p.text);
    if (textPart?.text) {
      throw new Error(`Gemini returned text: ${textPart.text}`);
    }

    throw new Error('No image in response');
  } catch (err) {
    console.error('❌ Gemini API failed:', err.message);
    return { success: false, error: err.message };
  }
}

async function testLocalServerEndpoint(mannequinPath, garmentPath) {
  console.log('\n🧪 Testing LOCAL SERVER endpoint...\n');

  const mannequinUri = await fileToDataUri(mannequinPath);
  const garmentUri = await fileToDataUri(garmentPath);

  const startTime = Date.now();

  try {
    const res = await fetch(`${API_BASE_URL}/api/tryon/gemini`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.TEST_TOKEN || 'test-token'}`,
      },
      body: JSON.stringify({
        mannequin_image: mannequinUri,
        garment_image: garmentUri,
        garment: { label: 'top' },
      }),
    });

    const data = await res.json();
    const elapsed = Date.now() - startTime;

    if (!data.success) {
      throw new Error(data.error || 'Unknown error');
    }

    console.log(`✓ Try-on complete in ${elapsed}ms`);
    console.log(`✓ Method: ${data.methodUsed}`);
    console.log(`✓ Coverage: ${(data.coverage * 100).toFixed(1)}%`);

    // Save result
    if (data.resultUrl) {
      const base64 = data.resultUrl.replace(/^data:image\/[^;]+;base64,/, '');
      const outputPath = `/tmp/gemini-tryon-server-${Date.now()}.png`;
      await fs.writeFile(outputPath, Buffer.from(base64, 'base64'));
      console.log(`✓ Saved to: ${outputPath}`);
    }

    return { success: true, elapsed };
  } catch (err) {
    console.error('❌ Server endpoint failed:', err.message);
    return { success: false, error: err.message };
  }
}

async function testHealthCheck() {
  console.log('\n🧪 Testing health check...\n');

  try {
    const res = await fetch(`${API_BASE_URL}/api/tryon/gemini/health`);
    const data = await res.json();
    console.log('Health:', data);
    return data.success;
  } catch (err) {
    console.error('❌ Health check failed:', err.message);
    return false;
  }
}

// Main
async function main() {
  if (!GEMINI_API_KEY) {
    console.error('GEMINI_API_KEY is required');
    process.exit(1);
  }

  const args = process.argv.slice(2);
  const mannequinPath = args[0] || 'backend/api/blank_mannequin.jpg';
  const garmentPath = args[1] || 'backend/api/data/shirt.png';

  console.log('═══════════════════════════════════════════════════════════');
  console.log('  Gemini 2.0 Flash Try-On Test');
  console.log('═══════════════════════════════════════════════════════════');
  console.log(`Mannequin: ${mannequinPath}`);
  console.log(`Garment: ${garmentPath}`);
  console.log(`API Key: ${GEMINI_API_KEY.slice(0, 10)}...`);
  console.log('');

  // Test 1: Direct API
  console.log('─'.repeat(50));
  const directResult = await testDirectGemini(mannequinPath, garmentPath);

  if (!directResult.success) {
    console.log('\n⚠️  Direct API test failed. Check API key and rate limits.');
    process.exit(1);
  }

  // Test 2: Health check (optional - needs running server)
  console.log('\n' + '─'.repeat(50));
  await testHealthCheck();

  console.log('\n' + '═'.repeat(50));
  console.log('✅ All tests completed!');
  console.log('');
  console.log('COST COMPARISON:');
  console.log('  FLUX.1-Kontext-dev: ~$0.20-0.50 per image');
  console.log('  Gemini 2.0 Flash:   ~$0.04 per image');
  console.log('  Savings: ~6-12x cheaper with Gemini!');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
