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

        const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });

        const imageParts = frames.slice(0, 5).map((base64Data) => ({
            inlineData: {
                data: base64Data.replace(/^data:image\/\w+;base64,/, ""),
                mimeType: "image/jpeg",
            },
        }));

        const prompt = `Analyze these video frames showing a person's wardrobe/clothes.
    List ALL clothing items you can identify across all frames.
    
    For each item, provide:
    - itemType: (e.g., T-Shirt, Jeans, Dress, Jacket, Sneakers, etc.)
    - color: Primary color(s)
    - style: Casual, Formal, Sport, or Streetwear
    - description: Brief description
    
    Return ONLY a valid JSON array, no other text:
    [{"itemType": "...", "color": "...", "style": "...", "description": "..."}]`;

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
                                text: `Analyze this image and list EACH clothing item separately. For EACH item provide:
1. itemType: specific type (e.g., "Denim Jacket", "V-neck T-shirt", "Slim-fit Jeans")
2. color: exact color(s)  
3. style: Casual/Formal/Sport/Streetwear
4. material: fabric type if visible (cotton, denim, leather, etc.)
5. productDescription: A short product description

Return JSON array: [{"itemType": "...", "color": "...", "style": "...", "material": "...", "productDescription": "..."}]
If no clothing visible, return [].`,
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
        console.log("OpenAI Clothing Analysis:", text);

        const jsonMatch = text.match(/\[[\s\S]*\]/);
        const detectedItems = jsonMatch ? JSON.parse(jsonMatch[0]) : [];

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
 * Generate product photo using Replicate SDXL
 */
router.post("/generate-product-image", async (req, res) => {
    try {
        const { description, itemType, color } = req.body;

        if (!description && !itemType) {
            return res.status(400).json({ error: "Description or item type required" });
        }

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
 * Uses AliceVision for intelligent frame selection and enhanced segmentation when available
 */
router.post("/product-photo/process", async (req, res) => {
    try {
        const { frames, clothingType, useAliceVision = true } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        console.log(`📸 Processing ${frames.length} frames - ${useAliceVision ? 'AliceVision Enhanced' : 'Standard'} mode...`);

        const steps = [];
        let bestFrameIndex = Math.floor(frames.length / 2);
        let bestFrame = frames[bestFrameIndex];
        let segmentedImageUrl;
        let keyframeScores = null;

        // STEP 1: Try AliceVision for intelligent frame selection
        if (useAliceVision) {
            try {
                console.log("🔍 Step 1: AliceVision intelligent frame selection...");
                const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";

                const keyframeResponse = await axios.post(`${aliceVisionUrl}/keyframe`, {
                    frames,
                    sharpness_weight: 0.4,
                    blur_penalty: 0.3,
                    centering_weight: 0.2
                }, { timeout: 30000 });

                if (keyframeResponse.data.success) {
                    bestFrameIndex = keyframeResponse.data.bestFrameIndex;
                    bestFrame = frames[bestFrameIndex];
                    keyframeScores = keyframeResponse.data.scores;
                    steps.push("alicevision_keyframe");
                    console.log(`✅ AliceVision selected frame ${bestFrameIndex} (score: ${keyframeScores.totalScore.toFixed(4)})`);
                }
            } catch (avError) {
                console.log("⚠️ AliceVision keyframe unavailable, using middle frame:", avError.message);
                steps.push("fallback_middle_frame");
            }
        } else {
            console.log("🔍 Step 1: Using middle frame...");
            steps.push("middle_frame");
        }

        const imageDataUrl = `data:image/jpeg;base64,${bestFrame}`;

        // STEP 2: Try AliceVision for enhanced segmentation, fallback to Replicate
        if (useAliceVision) {
            try {
                console.log("✂️ Step 2: AliceVision enhanced segmentation...");
                const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";

                const segmentResponse = await axios.post(`${aliceVisionUrl}/segment`, {
                    image: bestFrame,
                    add_white_background: true
                }, { timeout: 60000 });

                if (segmentResponse.data.success) {
                    segmentedImageUrl = segmentResponse.data.segmentedImage;
                    steps.push("alicevision_segment");
                    console.log(`✅ AliceVision segmentation complete (confidence: ${segmentResponse.data.confidence})`);
                }
            } catch (avSegError) {
                console.log("⚠️ AliceVision segment unavailable, using Replicate rembg:", avSegError.message);
            }
        }

        // Fallback to Replicate rembg if AliceVision segmentation failed
        if (!segmentedImageUrl) {
            console.log("✂️ Step 2: Replicate rembg segmentation...");
            try {
                segmentedImageUrl = await replicate.run(
                    "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
                    {
                        input: {
                            image: imageDataUrl,
                        },
                    }
                );
                steps.push("replicate_segment");
                console.log("✅ Replicate segmentation complete");
            } catch (segError) {
                console.log("Segmentation failed:", segError.message);
                segmentedImageUrl = imageDataUrl;
                steps.push("no_segmentation");
            }
        }

        // STEP 3: Try AliceVision lighting normalization
        let normalizedImageUrl = segmentedImageUrl;
        if (useAliceVision) {
            try {
                console.log("💡 Step 3: AliceVision lighting normalization...");
                const aliceVisionUrl = process.env.ALICEVISION_URL || "http://localhost:5050";

                // Extract base64 from the segmented image
                let imageB64 = segmentedImageUrl;
                if (imageB64.includes(',')) {
                    imageB64 = imageB64.split(',')[1];
                }

                const lightingResponse = await axios.post(`${aliceVisionUrl}/lighting`, {
                    image: imageB64,
                    target_brightness: 0.55,
                    target_temperature: 6000
                }, { timeout: 30000 });

                if (lightingResponse.data.success) {
                    normalizedImageUrl = lightingResponse.data.normalizedImage;
                    steps.push("alicevision_lighting");
                    console.log("✅ Lighting normalization complete");
                }
            } catch (lightError) {
                console.log("⚠️ AliceVision lighting unavailable:", lightError.message);
            }
        }

        // STEP 4: Use IP-Adapter to preserve clothing identity (Massimo Dutti style)
        console.log("🎨 Step 4: Transforming to Massimo Dutti style...");
        let finalImageUrl = normalizedImageUrl;

        try {
            const clothingDesc = clothingType || "clothing item";

            const output = await replicate.run(
                "lucataco/ip-adapter-sdxl:9ed17ca0dc62091449bf513afc32a4c7d0d38c8d1b485cc3e19b02cfe4ce6d31",
                {
                    input: {
                        image: normalizedImageUrl,
                        prompt: `professional e-commerce product photography of this exact ${clothingDesc}, 
              front facing view, flat lay on pure white background #FFFFFF, 
              ghost mannequin invisible mannequin style, 
              Massimo Dutti catalog aesthetic, studio lighting, 
              perfectly centered, symmetrical, high resolution 4K, 
              clean minimal premium luxury fashion photography`,
                        negative_prompt:
                            "person, human, model, body, mannequin visible, hanger, shadows, wrinkles, low quality, blurry, deformed, cropped",
                        strength: 0.6,
                        guidance_scale: 7.5,
                        num_inference_steps: 30,
                    },
                }
            );

            finalImageUrl = Array.isArray(output) ? output[0] : output;
            steps.push("ip_adapter_transform");
            console.log("✅ Transformed to Massimo Dutti style!");
        } catch (ipAdapterError) {
            console.log("IP-Adapter failed, trying SDXL fallback:", ipAdapterError.message);

            try {
                const clothingDesc = clothingType || "clothing item";

                const output = await replicate.run(
                    "stability-ai/sdxl:39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
                    {
                        input: {
                            prompt: `professional Massimo Dutti e-commerce photo of this exact ${clothingDesc}, 
                front facing flat lay on pure white background, ghost mannequin style, 
                studio lighting, fashion catalog quality, perfectly centered`,
                            image: normalizedImageUrl,
                            prompt_strength: 0.3,
                            negative_prompt: "person, human, model, mannequin visible, hanger, shadows, low quality",
                            width: 1024,
                            height: 1024,
                            num_inference_steps: 30,
                            guidance_scale: 7.5,
                        },
                    }
                );

                finalImageUrl = Array.isArray(output) ? output[0] : output;
                steps.push("sdxl_transform");
                console.log("✅ SDXL fallback succeeded");
            } catch (sdxlError) {
                console.log("SDXL also failed, using normalized image:", sdxlError.message);
                steps.push("no_transform");
            }
        }

        res.json({
            success: true,
            imageUrl: finalImageUrl,
            bestFrameIndex: bestFrameIndex,
            keyframeScores: keyframeScores,
            steps: steps,
            style: "massimo_dutti",
            aliceVisionEnhanced: steps.some(s => s.startsWith("alicevision_")),
            preservedOriginal: true,
        });
    } catch (error) {
        console.error("Product photo pipeline error:", error.message);
        res.status(500).json({ error: error.message });
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

export default router;
