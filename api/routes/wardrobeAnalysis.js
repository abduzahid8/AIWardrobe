// Backend API Route for Clarifai Video Analysis
const express = require('express');
const router = express.Router();
const multer = require('multer');
const ffmpeg = require('fluent-ffmpeg');
const path = require('path');
const fs = require('fs');
const axios = require('axios');

// Configure multer for video upload
const upload = multer({
    dest: 'uploads/videos/',
    limits: { fileSize: 100 * 1024 * 1024 } // 100MB limit
});

// Clarifai API configuration
const CLARIFAI_API_KEY = process.env.CLARIFAI_API_KEY || 'YOUR_API_KEY_HERE';
const CLARIFAI_USER_ID = 'clarifai';
const CLARIFAI_APP_ID = 'main';
const CLARIFAI_MODEL_ID = 'apparel-detection';

// Check if ffmpeg is available
ffmpeg.getAvailableFormats(function (err, formats) {
    if (err) {
        console.error('❌ FFmpeg is NOT installed or not found in PATH. Video analysis will fail.');
    } else {
        console.log('✅ FFmpeg is installed and ready.');
    }
});

if (!process.env.CLARIFAI_API_KEY) {
    console.warn('⚠️ CLARIFAI_API_KEY is not set in environment variables.');
}

/**
 * Extract frames from video at 1 frame per second
 */
async function extractFrames(videoPath, outputDir) {
    return new Promise((resolve, reject) => {
        // Create output directory if it doesn't exist
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        ffmpeg(videoPath)
            .outputOptions('-vf fps=1') // Extract 1 frame per second
            .output(path.join(outputDir, 'frame-%04d.jpg'))
            .on('end', () => {
                const frames = fs.readdirSync(outputDir)
                    .filter(f => f.startsWith('frame-') && f.endsWith('.jpg'))
                    .map(f => path.join(outputDir, f));
                resolve(frames);
            })
            .on('error', (err) => {
                console.error('FFmpeg error:', err);
                reject(err);
            })
            .run();
    });
}

/**
 * Analyze a single frame using Clarifai Fashion API
 */
async function analyzeFrameWithClarifai(imagePath) {
    try {
        const imageBytes = fs.readFileSync(imagePath);
        const base64Image = imageBytes.toString('base64');

        const response = await axios.post(
            `https://api.clarifai.com/v2/users/${CLARIFAI_USER_ID}/apps/${CLARIFAI_APP_ID}/models/${CLARIFAI_MODEL_ID}/outputs`,
            {
                inputs: [
                    {
                        data: {
                            image: {
                                base64: base64Image
                            }
                        }
                    }
                ]
            },
            {
                headers: {
                    'Authorization': `Key ${CLARIFAI_API_KEY}`,
                    'Content-Type': 'application/json'
                }
            }
        );

        if (response.data.status.code !== 10000) {
            throw new Error(`Clarifai API error: ${response.data.status.description}`);
        }

        // Extract detected clothing items
        const regions = response.data.outputs[0].data.regions || [];
        const detectedItems = regions.map(region => {
            const concepts = region.data.concepts || [];
            const topConcept = concepts[0]; // Get highest confidence concept

            return {
                type: topConcept?.name || 'unknown',
                confidence: topConcept?.value || 0,
                boundingBox: region.region_info?.bounding_box
            };
        }).filter(item => item.confidence > 0.6); // Filter low confidence detections

        return detectedItems;
    } catch (error) {
        console.error('Clarifai analysis error:', error.response?.data || error.message);
        throw error;
    }
}

/**
 * Main endpoint: Analyze wardrobe video
 */
router.post('/analyze-wardrobe', upload.single('video'), async (req, res) => {
    let videoPath = null;
    let framesDir = null;

    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No video file uploaded' });
        }

        videoPath = req.file.path;
        framesDir = path.join('uploads', 'frames', Date.now().toString());

        console.log('📹 Video received:', req.file.originalname);
        console.log('📂 Extracting frames...');

        // Extract frames from video
        const frames = await extractFrames(videoPath, framesDir);
        console.log(`✅ Extracted ${frames.length} frames`);

        if (frames.length === 0) {
            throw new Error('No frames could be extracted from video');
        }

        // Analyze each frame with Clarifai
        console.log('🔍 Analyzing frames with Clarifai...');
        const allDetections = [];

        for (let i = 0; i < frames.length; i++) {
            console.log(`Analyzing frame ${i + 1}/${frames.length}...`);
            try {
                const detections = await analyzeFrameWithClarifai(frames[i]);
                allDetections.push(...detections);
            } catch (error) {
                console.error(`Error analyzing frame ${i + 1}:`, error.message);
                // Continue with other frames even if one fails
            }
        }

        console.log(`✅ Total detections: ${allDetections.length}`);

        // Aggregate and deduplicate results
        const itemCounts = {};
        allDetections.forEach(item => {
            const key = item.type.toLowerCase();
            if (!itemCounts[key]) {
                itemCounts[key] = {
                    type: item.type,
                    count: 0,
                    totalConfidence: 0,
                    maxConfidence: 0
                };
            }
            itemCounts[key].count++;
            itemCounts[key].totalConfidence += item.confidence;
            itemCounts[key].maxConfidence = Math.max(itemCounts[key].maxConfidence, item.confidence);
        });

        // Calculate final results
        const finalItems = Object.values(itemCounts)
            .map(item => ({
                type: item.type,
                detectedInFrames: item.count,
                avgConfidence: (item.totalConfidence / item.count),
                maxConfidence: item.maxConfidence
            }))
            .sort((a, b) => b.detectedInFrames - a.detectedInFrames); // Sort by frequency

        // Cleanup temporary files
        console.log('🧹 Cleaning up...');
        if (fs.existsSync(videoPath)) fs.unlinkSync(videoPath);
        if (fs.existsSync(framesDir)) fs.rmSync(framesDir, { recursive: true });

        console.log('✅ Analysis complete!');

        res.json({
            success: true,
            itemsDetected: finalItems,
            totalFramesAnalyzed: frames.length,
            totalDetections: allDetections.length
        });

    } catch (error) {
        console.error('❌ Analysis error:', error);

        // Cleanup on error
        if (videoPath && fs.existsSync(videoPath)) {
            fs.unlinkSync(videoPath);
        }
        if (framesDir && fs.existsSync(framesDir)) {
            fs.rmSync(framesDir, { recursive: true });
        }

        res.status(500).json({
            error: error.message,
            details: error.response?.data || null
        });
    }
});

/**
 * Health check endpoint
 */
router.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        clarifaiConfigured: !!CLARIFAI_API_KEY && CLARIFAI_API_KEY !== 'YOUR_API_KEY_HERE'
    });
});

module.exports = router;
