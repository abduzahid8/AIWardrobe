import express from "express";
import multer from "multer";
import fs from "fs";
import axios from "axios";
import Replicate from "replicate";
import { HfInference } from "@huggingface/inference";
import { GoogleAIFileManager } from "@google/generative-ai/server";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { createClient } from "@supabase/supabase-js";
import cosineSimilarity from "compute-cosine-similarity";
import Outfit from "../models/outfit.js";

const router = express.Router();

// Initialize services
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const fileManager = new GoogleAIFileManager(process.env.GEMINI_API_KEY);
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const hf = new HfInference(process.env.HF_TOKEN);
const replicate = new Replicate({
    auth: process.env.REPLICATE_API_TOKEN,
});

// Configure multer for video uploads
const upload = multer({ dest: "uploads/" });

// ============================================
// GEMINI VISION - Frame Analysis
// ============================================

/**
 * POST /api/analyze-frames
 * Analyze video frames using Gemini Vision
 */
router.post("/analyze-frames", async (req, res) => {
    try {
        const { frames } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "No frames provided" });
        }

        console.log(`🖼️ Received ${frames.length} frames for analysis`);

        // Check if API key is set
        if (!process.env.GEMINI_API_KEY) {
            console.log('❌ GEMINI_API_KEY not set in .env file!');
            return res.status(500).json({ error: "Gemini API key not configured" });
        }
        console.log(`🔑 Gemini API key: ${process.env.GEMINI_API_KEY.substring(0, 10)}...`);

        // Use gemini-pro-vision for image analysis (v1beta API)
        const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });
        console.log(`✅ Using model: gemini-pro-vision`);

        const imageParts = frames.slice(0, 5).map((base64Data) => ({
            inlineData: {
                data: base64Data.replace(/^data:image\/\w+;base64,/, ""),
                mimeType: "image/jpeg",
            },
        }));

        const prompt = `IMPORTANT: Identify EVERY SINGLE clothing item visible in these video frames.
    There are likely MULTIPLE items (2-5 or more). Check ALL body areas carefully:
    
    1. 👕 UPPER BODY: shirts, t-shirts, blouses, jackets, coats, hoodies, sweaters
    2. 👖 LOWER BODY: pants, jeans, shorts, skirts, trousers
    3. 👗 FULL BODY: dresses, jumpsuits, overalls
    4. 👟 FEET: shoes, sneakers, boots, sandals, heels
    5. 👜 ACCESSORIES: bags, hats, scarves, belts, watches, jewelry
    
    For EACH item found, provide:
    - itemType: specific type (e.g., "Denim Jacket", "V-neck T-shirt", "Slim-fit Jeans")
    - color: exact color(s)
    - style: Casual, Formal, Sport, or Streetwear
    - description: brief product description
    - position: where on body (upper, lower, feet, accessory, full)
    - confidence: your confidence level (high, medium, low)
    
    CRITICAL: Do NOT return just 1 item if multiple are visible!
    Return EVERY item as a JSON array:
    [{"itemType": "...", "color": "...", "style": "...", "description": "...", "position": "...", "confidence": "..."}]`;

        const result = await model.generateContent([prompt, ...imageParts]);
        const responseText = result.response.text();

        console.log("🤖 Gemini response:", responseText);

        let detectedItems = [];
        try {
            const jsonMatch = responseText.match(/\[[\s\S]*\]/);
            if (jsonMatch) {
                detectedItems = JSON.parse(jsonMatch[0]);
            }
        } catch (parseError) {
            console.error("Parse error:", parseError);
            detectedItems = [
                {
                    itemType: "Unknown Item",
                    color: "Unknown",
                    style: "Casual",
                    description: "Could not parse response",
                },
            ];
        }

        res.json({ detectedItems });
    } catch (error) {
        console.error("Frame analysis error:", error);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// OPENAI - Clothing Analysis & Image Generation
// ============================================

/**
 * POST /api/openai/analyze-clothing
 * Analyze clothing using OpenAI Vision API
 */
router.post("/openai/analyze-clothing", async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "Image base64 is required" });
        }

        const response = await axios.post(
            "https://api.openai.com/v1/chat/completions",
            {
                model: "gpt-4o-mini",
                messages: [
                    {
                        role: "user",
                        content: [
                            {
                                type: "text",
                                text: `You are a precise fashion AI. Analyze this image and identify ONLY DISTINCT, CLEARLY VISIBLE clothing items.

RULES:
- Count each item ONLY ONCE (don't count the same jacket as both "jacket" and "outerwear")
- Only include items you can clearly see (minimum 50% visible)
- DO NOT count partially hidden items behind other clothes
- DO NOT count accessories like watches, jewelry, or belts unless specifically asked
- If you see a layered outfit (shirt under jacket), count each as separate ONLY if both are clearly visible

For each DISTINCT item, provide:
1. itemType: specific type (e.g., "Denim Jacket", "Crew-neck T-shirt", "Slim-fit Jeans")
2. color: primary color(s)
3. style: Casual/Formal/Sport/Streetwear
4. material: fabric type if visible
5. description: 1-sentence product description
6. confidence: your confidence 0-100 (only include items with 70%+ confidence)
7. position: upper/lower/full/feet

Return ONLY items with 70%+ confidence as JSON array:
[{"itemType": "...", "color": "...", "style": "...", "material": "...", "description": "...", "confidence": 85, "position": "upper"}]

Be conservative - it's better to miss an item than to add a false one.`,
                            },
                            {
                                type: "image_url",
                                image_url: { url: `data:image/jpeg;base64,${imageBase64}` },
                            },
                        ],
                    },
                ],
                max_tokens: 1000,
            },
            {
                headers: {
                    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
                    "Content-Type": "application/json",
                },
                timeout: 45000,
            }
        );

        const text = response.data.choices?.[0]?.message?.content || "[]";
        console.log("OpenAI Clothing Analysis (raw):", text);

        const jsonMatch = text.match(/\[[\s\S]*\]/);
        let detectedItems = jsonMatch ? JSON.parse(jsonMatch[0]) : [];

        // Filter by confidence (only keep 70%+ confidence items)
        const originalCount = detectedItems.length;
        detectedItems = detectedItems.filter(item => {
            const confidence = item.confidence || 100;
            return confidence >= 70;
        });

        console.log(`✅ OpenAI detected ${originalCount} items, ${detectedItems.length} passed 70% confidence threshold`);
        console.log("Items:", detectedItems.map(i => `${i.itemType} (${i.confidence}%)`).join(", "));

        res.json({ detectedItems });
    } catch (error) {
        console.error("OpenAI analysis error:", error.response?.data || error.message);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/openai/generate-image
 * Generate product image using DALL-E
 */
router.post("/openai/generate-image", async (req, res) => {
    try {
        const { prompt } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: "Prompt is required" });
        }

        const response = await axios.post(
            "https://api.openai.com/v1/images/generations",
            {
                model: "dall-e-3",
                prompt: `${prompt}, professional product photography, clean white background, studio lighting, high quality, e-commerce style, centered, full garment visible`,
                n: 1,
                size: "1024x1024",
                quality: "standard",
            },
            {
                headers: {
                    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
                    "Content-Type": "application/json",
                },
                timeout: 60000,
            }
        );

        const imageUrl = response.data.data[0].url;
        res.json({ imageUrl });
    } catch (error) {
        console.error("DALL-E generation error:", error.response?.data || error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// REPLICATE - Image Generation & Processing
// ============================================

/**
 * POST /api/generate-product-image
 * Generate product photo using Replicate SDXL (TEMPORARILY DISABLED)
 */
router.post("/generate-product-image", async (req, res) => {
    try {
        const { description, itemType, color } = req.body;

        if (!description && !itemType) {
            return res.status(400).json({ error: "Description or item type required" });
        }

        // Replicate temporarily disabled - return placeholder or use AliceVision output
        console.log("⚠️ generate-product-image called but Replicate is disabled");
        console.log("   Use /api/product-photo/process instead for AliceVision processing");

        // Return a note that this endpoint is disabled
        res.json({
            imageUrl: null,
            message: "Replicate disabled. Use /api/product-photo/process for AliceVision-enhanced images",
            useAliceVision: true
        });

        /* REPLICATE CODE - Uncomment when credits available
        const prompt =
            description ||
            `A ${color || ""} ${itemType}, professional product photography, clean white background, studio lighting, high quality fashion catalog style, centered, full garment visible, no model, isolated on white`;

        console.log("🎨 Generating product image with SDXL:", prompt);

        const output = await replicate.run(
            "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
            {
                input: {
                    prompt: prompt,
                    negative_prompt: "human, person, model, mannequin, low quality, blurry, distorted",
                    width: 1024,
                    height: 1024,
                    num_inference_steps: 25,
                    guidance_scale: 7.5,
                },
            }
        );

        const imageUrl = Array.isArray(output) ? output[0] : output;
        console.log("✅ Generated image:", imageUrl);

        res.json({ imageUrl });
        */
    } catch (error) {
        console.error("Image generation error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/remove-background
 * Remove background from clothing image
 */
router.post("/remove-background", async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "Image base64 is required" });
        }

        console.log("🎨 Removing background from clothing image...");

        const output = await replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            {
                input: {
                    image: `data:image/jpeg;base64,${imageBase64}`,
                },
            }
        );

        console.log("✅ Background removed:", output);
        res.json({ imageUrl: output });
    } catch (error) {
        console.error("Background removal error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /api/product-photo/process
 * Professional product photo pipeline - Massimo Dutti style
 * Uses AliceVision comprehensive AI for intelligent frame selection,
 * advanced segmentation with full aspect preservation, and quality assessment
 */
router.post("/product-photo/process", async (req, res) => {
    try {
        const { frames, clothingType, clothingColor, clothingStyle, clothingDescription, useAliceVision = true } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        console.log(`📸 Processing ${frames.length} frames - ${useAliceVision ? 'AliceVision AI Enhanced' : 'Standard'} mode...`);
        console.log(`   Clothing: ${clothingType}, ${clothingColor}, ${clothingStyle}`);

        const steps = [];
        let bestFrameIndex = Math.floor(frames.length / 2);
        let bestFrame = frames[bestFrameIndex];
        let finalImageUrl;
        let analysisData = {};

        // STEP 1: Try AliceVision comprehensive analysis
        if (useAliceVision) {
            try {
                console.log("🔍 Step 1: AliceVision comprehensive AI analysis...");
                const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";

                // Use comprehensive analysis endpoint
                const comprehensiveResponse = await axios.post(`${aliceVisionUrl}/comprehensive-analysis`, {
                    image: frames[0], // Start with first frame, can upgrade to multi-frame later
                    include_detection: true,
                    include_segmentation: true,
                    include_attributes: true,
                    include_quality: true
                }, { timeout: 90000 }); // 90 seconds for complete analysis

                if (comprehensiveResponse.data && comprehensiveResponse.data.success) {
                    analysisData = comprehensiveResponse.data;
                    steps.push("comprehensive_ai_analysis");

                    console.log(`✅ AI Analysis complete:`);
                    console.log(`   - Detected items: ${analysisData.product?.detections?.length || 0}`);
                    console.log(`   - Segmented items: ${analysisData.segmentation?.itemCount || 0}`);
                    console.log(`   - Primary color: ${analysisData.attributes?.primaryColor || 'unknown'}`);
                    console.log(`   - Quality score: ${analysisData.quality?.overall || 'N/A'}`);
                    console.log(`   - E-commerce ready: ${analysisData.quality?.ecommerceReady || false}`);

                    // If quality is poor, try another frame
                    if (analysisData.quality && analysisData.quality.overall < 60 && frames.length > 1) {
                        console.log("⚠️ Quality score low, trying alternate frame...");
                        const altResponse = await axios.post(`${aliceVisionUrl}/comprehensive-analysis`, {
                            image: frames[Math.floor(frames.length / 2)],
                            include_detection: true,
                            include_segmentation: true,
                            include_attributes: true,
                            include_quality: true
                        }, { timeout: 90000 });

                        if (altResponse.data.quality && altResponse.data.quality.overall > analysisData.quality.overall) {
                            console.log(`✅ Better frame found (quality: ${altResponse.data.quality.overall})`);
                            analysisData = altResponse.data;
                        }
                    }
                }
            } catch (aiError) {
                console.log("⚠️ AliceVision comprehensive analysis unavailable:", aiError.message);
                console.log("   Falling back to basic segmentation...");
            }
        }

        // STEP 2: If comprehensive analysis failed, fallback to basic segmentation
        if (!analysisData.segmentation) {
            console.log("✂️ Step 2: Advanced clothing segmentation...");

            if (useAliceVision) {
                try {
                    const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";

                    // Use advanced segmentation with edge refinement
                    const segmentResponse = await axios.post(`${aliceVisionUrl}/segment`, {
                        image: bestFrame,
                        add_white_background: true,
                        use_advanced: true  // Use SegFormer for 18-category detection
                    }, { timeout: 60000 });

                    if (segmentResponse.data && segmentResponse.data.success) {
                        finalImageUrl = segmentResponse.data.segmentedImage;
                        analysisData.segmentation = {
                            confidence: segmentResponse.data.confidence,
                            itemCount: segmentResponse.data.itemCount,
                            items: segmentResponse.data.items
                        };
                        steps.push("advanced_segmentation");
                        console.log(`✅ Advanced segmentation complete (confidence: ${segmentResponse.data.confidence.toFixed(3)})`);
                        console.log(`   - Detected ${segmentResponse.data.itemCount} clothing items`);
                    }
                } catch (segError) {
                    console.log("⚠️ Advanced segmentation failed:", segError.message);
                }
            }

            // Fallback to basic Replicate rembg if both failed
            if (!finalImageUrl) {
                console.log("✂️ Fallback: Basic background removal...");
                try {
                    const imageDataUrl = `data:image/jpeg;base64,${bestFrame}`;
                    finalImageUrl = await replicate.run(
                        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
                        {
                            input: {
                                image: imageDataUrl,
                            },
                        }
                    );
                    steps.push("basic_background_removal");
                    console.log("✅ Basic background removal complete");
                } catch (rembgError) {
                    console.log("Background removal failed:", rembgError.message);
                    // Last resort: return original frame
                    finalImageUrl = `data:image/jpeg;base64,${bestFrame}`;
                    steps.push("no_processing");
                }
            }
        } else {
            // Use the segmented image from comprehensive analysis
            // The comprehensive endpoint returns the full analysis but not the image
            // So we need to call segmentation separately
            console.log("✂️ Step 2: Extracting segmented image from AI analysis...");

            try {
                const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";
                const segmentResponse = await axios.post(`${aliceVisionUrl}/segment`, {
                    image: bestFrame,
                    add_white_background: true,
                    use_advanced: true
                }, { timeout: 60000 });

                if (segmentResponse.data && segmentResponse.data.success) {
                    finalImageUrl = segmentResponse.data.segmentedImage;
                    steps.push("ai_segmentation");
                    console.log(`✅ AI-guided segmentation extracted`);
                }
            } catch (err) {
                console.log("Segmentation extraction failed:", err.message);
            }
        }

        // STEP 3: Apply lighting normalization to enhance the product photo
        if (finalImageUrl && useAliceVision) {
            try {
                console.log("💡 Step 3: Studio lighting normalization...");
                const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";

                // Extract base64 from the segmented image
                let imageB64 = finalImageUrl;
                if (imageB64.includes(',')) {
                    imageB64 = imageB64.split(',')[1];
                }

                const lightingResponse = await axios.post(`${aliceVisionUrl}/lighting`, {
                    image: imageB64,
                    target_brightness: 0.6,  // Slightly brighter for e-commerce
                    target_temperature: 6500, // Cool white for product photos
                    add_vignette: false
                }, { timeout: 30000 });

                if (lightingResponse.data && lightingResponse.data.success) {
                    finalImageUrl = lightingResponse.data.normalizedImage;
                    steps.push("lighting_normalization");
                    console.log("✅ Studio lighting applied");
                }
            } catch (lightError) {
                console.log("⚠️ Lighting normalization skipped:", lightError.message);
            }
        }

        // Build comprehensive response
        const response = {
            success: true,
            imageUrl: finalImageUrl,
            bestFrameIndex: bestFrameIndex,
            steps: steps,
            aiEnhanced: steps.some(s => s.includes("ai_") || s.includes("comprehensive")),
            analysis: {
                colors: analysisData.attributes?.colors || [],
                primaryColor: analysisData.attributes?.primaryColor || clothingColor || "unknown",
                pattern: analysisData.attributes?.pattern?.type || "solid",
                material: analysisData.attributes?.material?.type || "unknown",
                detectedCategory: analysisData.product?.primaryProduct?.category || clothingType || "clothing",
                confidence: analysisData.segmentation?.confidence || 0.85,
                itemCount: analysisData.segmentation?.itemCount || 1,
                quality: {
                    overall: analysisData.quality?.overall || 75,
                    ecommerceReady: analysisData.quality?.ecommerceReady || false,
                    issues: analysisData.quality?.issues || [],
                    recommendations: analysisData.quality?.recommendations || []
                }
            },
            preservedFullAspect: true,
            cleanBackground: true,
            processing: {
                totalTimeMs: analysisData.totalProcessingTimeMs || 0,
                steps: steps
            }
        };

        console.log(`✅ Product photo pipeline complete:`);
        console.log(`   - Final image: ${finalImageUrl ? 'Generated' : 'Failed'}`);
        console.log(`   - Processing steps: ${steps.join(' → ')}`);
        console.log(`   - Quality: ${response.analysis.quality.overall}/100`);
        console.log(`   - E-commerce ready: ${response.analysis.quality.ecommerceReady ? 'YES' : 'NO'}`);

        res.json(response);
    } catch (error) {
        console.error("Product photo pipeline error:", error.message);
        res.status(500).json({ error: error.message, details: error.stack });
    }
});

/**
 * POST /try-on
 * Virtual try-on using IDM-VTON
 */
router.post("/try-on", async (req, res) => {
    const { human_image, garment_image, description } = req.body;

    console.log("🎨 Starting virtual try-on...");
    console.log("Human:", human_image);
    console.log("Garment:", garment_image);

    try {
        const output = await replicate.run(
            "cuuupid/idm-vton:906425dbca90663ff54276248397db52027860a241f03fad3e5a04127a7570c8",
            {
                input: {
                    human_img: human_image,
                    garm_img: garment_image,
                    garment_des: description || "clothing",
                    crop: false,
                    seed: 42,
                    steps: 30,
                },
            }
        );

        console.log("✅ Try-on complete:", output);
        res.json({ image: output });
    } catch (error) {
        console.error("Replicate error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /scan-wardrobe
 * Scan wardrobe from video using Video-LLaVA
 */
router.post("/scan-wardrobe", upload.single("video"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No video file uploaded" });
        }

        console.log("🎥 Video received:", req.file.path);

        // Read file from disk
        const fileBuffer = fs.readFileSync(req.file.path);
        const fileName = `scan_${Date.now()}.mp4`;

        const BUCKET_NAME = "AIWARDROBE";

        console.log(`🔍 [DEBUG] Uploading to bucket: "${BUCKET_NAME}"`);

        // Upload to Supabase
        const { data: uploadData, error: uploadError } = await supabase.storage
            .from(BUCKET_NAME)
            .upload(fileName, fileBuffer, {
                contentType: "video/mp4",
                upsert: false,
            });

        if (uploadError) {
            console.error("❌ SUPABASE ERROR:", JSON.stringify(uploadError, null, 2));
            throw new Error(`Supabase upload failed: ${uploadError.message}`);
        }

        // Get public URL
        const { data: publicUrlData } = supabase.storage.from("AIWARDROBE").getPublicUrl(fileName);

        const videoUrl = publicUrlData.publicUrl;
        console.log(`🔗 Video URL: ${videoUrl}`);

        // Send to Replicate (Video-LLaVA)
        console.log("🧠 Sending to Replicate (Video-LLaVA)...");

        const input = {
            video_path: videoUrl,
            text_prompt: `List the clothing items in this video. 
      Format the output EXACTLY as a JSON list of objects.
      Each object must have: "itemType", "color", "style" (Casual/Formal), "description".
      Example: [{"itemType": "Shirt", "color": "Blue", "style": "Casual", "description": "Denim shirt"}]
      Do NOT include any other text, markdown, or explanations. ONLY the JSON array.`,
        };

        const output = await replicate.run(
            "lucataco/video-llava:16922da8774708779c3b9b9409549eb936307373322bc69c3bb9da40d42630e5",
            { input }
        );

        console.log("🤖 Replicate response:", output);

        // Parse response
        const rawText = Array.isArray(output) ? output.join("") : String(output);

        let items = [];
        try {
            const firstBracket = rawText.indexOf("[");
            const lastBracket = rawText.lastIndexOf("]");

            if (firstBracket !== -1 && lastBracket !== -1) {
                const jsonStr = rawText.substring(firstBracket, lastBracket + 1);
                items = JSON.parse(jsonStr);
            } else {
                console.log("⚠️ Could not find JSON, using raw text");
                items = [
                    {
                        itemType: "Detected Item",
                        color: "Mixed",
                        style: "Casual",
                        description: rawText.substring(0, 100).replace(/\n/g, " "),
                    },
                ];
            }
        } catch (parseErr) {
            console.error("Parse error:", parseErr);
            items = [
                {
                    itemType: "Unknown Item",
                    color: "Unknown",
                    style: "Casual",
                    description: "Item from video",
                },
            ];
        }

        // Clean up temp file
        if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);

        res.json({ detectedItems: items });
    } catch (error) {
        console.error("Video Scan Error:", error);
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// HUGGING FACE - Chat & Search
// ============================================

/**
 * Generate text embedding using HuggingFace
 */
const generateEmbedding = async (text) => {
    const response = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: text,
    });
    return response;
};

/**
 * Normalize search query
 */
const normalizeQuery = (query) => {
    const synonyms = {
        "coffee date": "coffee date",
        "dinner date": "date",
        "job interview": "interview",
        work: "interview",
        casual: "casual",
        formal: "formal",
        outfit: "",
        "give me": "",
        a: "",
        an: "",
        for: "",
    };

    let normalized = query.toLowerCase();
    Object.keys(synonyms).forEach((key) => {
        normalized = normalized.replace(new RegExp(`\\b${key}\\b`, "gi"), synonyms[key]);
    });
    return [...new Set(normalized.trim().split(/\s+/).filter(Boolean))].join(" ");
};

/**
 * GET /smart-search
 * AI-powered outfit search
 */
router.get("/smart-search", async (req, res) => {
    const { query } = req.query;
    if (!query) return res.status(400).json({ error: "Query required" });

    try {
        const normalizedQuery = normalizeQuery(query);
        const queryEmbedding = await generateEmbedding(normalizedQuery);
        const outfits = await Outfit.find();

        const MIN_SIMILARITY = query.length > 20 ? 0.3 : 0.4;

        let scored = outfits
            .map((o) => {
                const score = cosineSimilarity(queryEmbedding, o.embedding);
                return { ...o.toObject(), score };
            })
            .filter((o) => o.score >= MIN_SIMILARITY)
            .sort((a, b) => b.score - a.score);

        if (scored.length === 0) {
            const queryTerms = normalizedQuery.split(" ");
            scored = outfits
                .filter((o) =>
                    queryTerms.some(
                        (term) =>
                            (o.occasion || "").toLowerCase().includes(term) ||
                            (o.style || "").toLowerCase().includes(term) ||
                            (o.items || []).some((item) => (item || "").toLowerCase().includes(term))
                    )
                )
                .map((o) => ({ ...o.toObject(), score: 0.1 }));
        }

        res.json(scored.slice(0, 5));
    } catch (err) {
        console.error("🔴 AI ERROR:", err);
        res.status(500).json({ error: err.message });
    }
});

/**
 * POST /ai-chat
 * AI fashion stylist chat
 */
router.post("/ai-chat", async (req, res) => {
    const { query } = req.body;
    console.log("📨 Chat request:", query);

    try {
        const result = await hf.chatCompletion({
            model: "meta-llama/Meta-Llama-3-8B-Instruct",
            messages: [
                {
                    role: "system",
                    content: "You are a helpful fashion stylist. Keep answers short and fun with emojis.",
                },
                { role: "user", content: query },
            ],
            max_tokens: 500,
            temperature: 0.7,
        });

        if (result && result.choices && result.choices.length > 0) {
            console.log("🤖 Response:", result.choices[0].message.content);
            res.json({ text: result.choices[0].message.content });
        } else {
            throw new Error("AI returned empty response");
        }
    } catch (err) {
        console.error("❌ HF Error:", err.message);
        res.status(500).json({ error: "AI model is busy, try again later." });
    }
});

// ============================================
// AI OUTFIT GENERATOR - Curated Recommendations
// ============================================

/**
 * POST /api/generate-outfits
 * Generate outfit recommendations with AI-powered matching
 */
router.post("/generate-outfits", async (req, res) => {
    try {
        const { occasion, stylePreferences, limit = 5 } = req.body;

        if (!occasion && !stylePreferences) {
            return res.status(400).json({
                error: "Please provide an occasion or style preferences"
            });
        }

        console.log("🎨 Generating outfits with AI for:", { occasion, stylePreferences });

        // Import outfit database
        const { curatedOutfits } = await import("../data/curatedOutfits.js");

        // Try AI-powered matching first, fallback to keyword matching
        let scoredOutfits;

        try {
            // Use Gemini for semantic understanding
            const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

            const prompt = `You are a fashion AI stylist. Analyze these outfit requests and score each outfit from 0-100.

USER REQUEST:
Occasion: ${occasion || 'any'}
Style Preferences: ${stylePreferences || 'none specified'}

AVAILABLE OUTFITS:
${curatedOutfits.map((outfit, idx) => `
${idx + 1}. ${outfit.description}
   - Occasions: ${outfit.occasion.join(', ')}
   - Styles: ${outfit.style.join(', ')}
`).join('\n')}

Return ONLY a JSON array of scores, one number (0-100) for each outfit in order. Example: [95, 70, 85, 60, ...]`;

            const result = await model.generateContent(prompt);
            const responseText = result.response.text();

            // Extract numbers from response
            const scores = JSON.parse(responseText.match(/\[[\d,\s]+\]/)?.[0] || '[]');

            if (scores.length === curatedOutfits.length) {
                console.log("✅ Using AI-powered matching");
                scoredOutfits = curatedOutfits.map((outfit, idx) => ({
                    ...outfit,
                    matchScore: scores[idx] / 100
                }));
            } else {
                throw new Error("AI returned invalid scores");
            }
        } catch (aiError) {
            console.log("⚠️ AI matching failed, using keyword fallback:", aiError.message);

            // Fallback: Enhanced keyword matching
            const styleKeywords = stylePreferences
                ? stylePreferences.toLowerCase().split(/[\s,]+/).filter(w => w.length > 2)
                : [];

            scoredOutfits = curatedOutfits.map(outfit => {
                let score = 0;

                // Exact occasion match (weight: 10)
                if (occasion && outfit.occasion.includes(occasion.toLowerCase())) {
                    score += 10;
                }

                // Partial occasion match (weight: 5)
                if (occasion && outfit.occasion.some(occ =>
                    occ.includes(occasion.toLowerCase()) || occasion.toLowerCase().includes(occ)
                )) {
                    score += 5;
                }

                // Style keyword matches
                styleKeywords.forEach(keyword => {
                    // Exact style match (weight: 4)
                    if (outfit.style.some(s => s === keyword)) {
                        score += 4;
                    }
                    // Partial style match (weight: 2)
                    else if (outfit.style.some(s => s.includes(keyword) || keyword.includes(s))) {
                        score += 2;
                    }
                    // Description match (weight: 1)
                    if (outfit.description.toLowerCase().includes(keyword)) {
                        score += 1;
                    }
                });

                return { ...outfit, matchScore: Math.min(score / 15, 1) }; // Normalize to 0-1
            });
        }

        // Sort by score and return top matches
        const topOutfits = scoredOutfits
            .filter(o => o.matchScore > 0)
            .sort((a, b) => b.matchScore - a.matchScore)
            .slice(0, limit);

        console.log(`✅ Found ${topOutfits.length} matching outfits (top score: ${topOutfits[0]?.matchScore?.toFixed(2)})`);

        res.json({
            success: true,
            outfits: topOutfits,
            query: {
                occasion,
                stylePreferences
            },
            aiPowered: topOutfits[0]?.matchScore > 0.5
        });

    } catch (error) {
        console.error("Outfit generation error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// V2 ENHANCED - Multi-Item Detection & Card Generation
// ============================================

/**
 * POST /api/v2/product-photo/process-multi
 * Enhanced product photo pipeline - Detects MULTIPLE items and creates separate cards
 * Uses Grounded SAM2 for intelligent multi-item detection + FashionCLIP attributes
 * Creates one professional Massimo Dutti-style card per detected item
 */
router.post("/v2/product-photo/process-multi", async (req, res) => {
    try {
        const { frames, prompts = null } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        console.log(`🎯 V2 Multi-Item Processing: ${frames.length} frames`);
        console.log(`   Detecting items: ${prompts || 'all clothing items'}`);

        const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";
        const itemCards = [];
        const steps = [];

        // STEP 1: Use Grounded SAM2 to detect ALL clothing items
        console.log("👁️ Step 1: Grounded SAM2 multi-item detection...");

        try {
            const detectionResponse = await axios.post(
                `${aliceVisionUrl}/api/v2/detect-clothing`,
                {
                    image: frames[0], // Use first frame
                    prompts: prompts || ["shirt", "pants", "dress", "jacket", "skirt", "shoes", "bag"],
                    return_masks: true
                },
                { timeout: 60000 }
            );

            if (!detectionResponse.data.success || !detectionResponse.data.detections || detectionResponse.data.detections.length === 0) {
                console.log("⚠️ No items detected with Grounded SAM2");
                return res.status(400).json({
                    error: "No clothing items detected in the image",
                    suggestion: "Make sure the image shows clothing items clearly"
                });
            }

            const detections = detectionResponse.data.detections;
            steps.push(`grounded_sam2_detected_${detections.length}_items`);

            console.log(`✅ Detected ${detections.length} items:`);
            detections.forEach((det, i) => {
                console.log(`   ${i + 1}. ${det.category} (confidence: ${(det.confidence * 100).toFixed(1)}%)`);
            });

            // STEP 2: For EACH detected item, extract attributes and create card
            for (let i = 0; i < detections.length; i++) {
                const detection = detections[i];
                console.log(`\n📦 Processing item ${i + 1}/${detections.length}: ${detection.category}`);

                try {
                    // 2a: Extract fashion attributes using FashionCLIP
                    console.log(`   🎨 Extracting attributes with FashionCLIP...`);

                    const attributesResponse = await axios.post(
                        `${aliceVisionUrl}/api/v2/extract-fashion-attributes`,
                        {
                            image: frames[0],
                            roi: detection.bbox // Focus on this specific item
                        },
                        { timeout: 30000 }
                    );

                    if (!attributesResponse.data.success) {
                        console.log(`   ⚠️ Attribute extraction failed for ${detection.category}`);
                        continue;
                    }

                    const attributes = attributesResponse.data;
                    steps.push(`fashion_clip_${detection.category}`);

                    console.log(`   ✅ Extracted attributes:`);
                    console.log(`      - Category: ${attributes.category}`);
                    console.log(`      - Colors: ${attributes.colors.map(c => c.name).join(', ')}`);
                    console.log(`      - Pattern: ${attributes.patterns[0]?.name || 'unknown'}`);
                    console.log(`      - Style: ${attributes.styles[0]?.name || 'unknown'}`);

                    // 2b: Generate Massimo Dutti style card prompt
                    console.log(`   🖼️ Generating Massimo Dutti style prompt...`);

                    const cardPromptResponse = await axios.post(
                        `${aliceVisionUrl}/api/v2/generate-card-prompt`,
                        {
                            attributes: attributes,
                            style: "massimo_dutti", // Use Massimo Dutti preset
                            include_model: false // No model, just the clothing
                        },
                        { timeout: 10000 }
                    );

                    if (!cardPromptResponse.data.success) {
                        console.log(`   ⚠️ Card prompt generation failed`);
                        continue;
                    }

                    const cardPrompt = cardPromptResponse.data;
                    steps.push(`card_prompt_${detection.category}`);

                    console.log(`   ✅ Generated card prompt`);
                    console.log(`      Prompt: ${cardPrompt.prompt.substring(0, 100)}...`);

                    // 2c: Get segmented image for this item
                    console.log(`   ✂️ Segmenting item with white background...`);

                    // Since we have the mask from Grounded SAM2, we can use it
                    // Or call the segmentation endpoint for a clean cut
                    const segmentResponse = await axios.post(
                        `${aliceVisionUrl}/segment`,
                        {
                            image: frames[0],
                            add_white_background: true,
                            use_advanced: true
                        },
                        { timeout: 30000 }
                    );

                    let itemImageUrl = null;
                    if (segmentResponse.data && segmentResponse.data.success) {
                        itemImageUrl = segmentResponse.data.segmentedImage;
                        console.log(`   ✅ Segmentation complete`);
                    } else {
                        console.log(`   ⚠️ Segmentation failed, using original frame`);
                        itemImageUrl = `data:image/jpeg;base64,${frames[0]}`;
                    }

                    // Build the card data for this item
                    const itemCard = {
                        itemNumber: i + 1,
                        detection: {
                            category: detection.category,
                            confidence: detection.confidence,
                            bbox: detection.bbox
                        },
                        attributes: {
                            category: attributes.category,
                            subcategory: attributes.subcategory,
                            primaryColor: attributes.colors[0]?.name || "unknown",
                            colors: attributes.colors.map(c => c.name),
                            pattern: attributes.patterns[0]?.name || "solid",
                            style: attributes.styles[0]?.name || "casual",
                            fabric: attributes.fabric,
                            details: attributes.details,
                            description: attributes.description
                        },
                        cardPrompt: {
                            prompt: cardPrompt.prompt,
                            negative_prompt: cardPrompt.negative_prompt,
                            tags: cardPrompt.tags
                        },
                        imageUrl: itemImageUrl,
                        style: "massimo_dutti",
                        massimoOutti: true,
                        whiteBackground: true,
                        frontFacing: true
                    };

                    itemCards.push(itemCard);
                    console.log(`   ✅✅ Card ${i + 1} complete!`);

                } catch (itemError) {
                    console.error(`   ❌ Error processing item ${i + 1}:`, itemError.message);
                    // Continue with next item
                }
            }

            if (itemCards.length === 0) {
                return res.status(500).json({
                    error: "Failed to process any items",
                    detectedCount: detections.length
                });
            }

            // Build final response
            const response = {
                success: true,
                totalItemsDetected: detections.length,
                totalCardsCreated: itemCards.length,
                items: itemCards,
                processing: {
                    steps: steps,
                    aiEnhanced: true,
                    model: "grounded_sam2_fashion_clip",
                    style: "massimo_dutti"
                },
                summary: {
                    categories: itemCards.map(item => item.attributes.category),
                    colors: itemCards.map(item => item.attributes.primaryColor),
                    styles: itemCards.map(item => item.attributes.style)
                }
            };

            console.log(`\n✅✅✅ V2 Multi-Item Processing Complete:`);
            console.log(`   - Detected: ${detections.length} items`);
            console.log(`   - Created: ${itemCards.length} Massimo Dutti cards`);
            console.log(`   - Categories: ${response.summary.categories.join(', ')}`);

            res.json(response);

        } catch (detectionError) {
            console.error("❌ Grounded SAM2 detection failed:", detectionError.message);

            // Fallback to old single-item processing
            console.log("⚠️ Falling back to single-item processing...");

            return res.status(503).json({
                error: "Multi-item detection service unavailable",
                message: "The AI vision service (Grounded SAM2) is not available. Please ensure it's running on port 5050.",
                fallback: "Use /api/product-photo/process for single-item processing"
            });
        }

    } catch (error) {
        console.error("V2 Multi-Item Pipeline error:", error.message);
        res.status(500).json({
            error: error.message,
            stack: error.stack
        });
    }
});

export default router;

