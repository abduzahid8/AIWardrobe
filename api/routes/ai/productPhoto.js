/**
 * Product Photo Pipeline Routes
 * POST /product-photo/process           — Single-item Massimo Dutti pipeline
 * POST /v2/product-photo/process-multi  — Multi-item detection with Grounded SAM2
 */
import express from "express";
import axios from "axios";
import Replicate from "replicate";
import { authenticateToken } from "../../middleware/auth.js";
import { requireTier } from "../../middleware/subscriptionGuard.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import { ALICEVISION_URL } from "../../config.js";
import logger from "../../utils/logger.js";

const router = express.Router();
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// ── POST /product-photo/process ──
router.post("/product-photo/process", authenticateToken, requireTier('premium'), aiLimiter, async (req, res) => {
    try {
        const { frames, clothingType, clothingColor, clothingStyle, clothingDescription, useAliceVision = true } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        logger.info(` Processing ${frames.length} frames - ${useAliceVision ? 'AliceVision AI Enhanced' : 'Standard'} mode...`);

        const steps = [];
        const bestFrameIndex = Math.floor(frames.length / 2);
        const bestFrame = frames[bestFrameIndex];
        let finalImageUrl;
        let analysisData = {};

        // STEP 1: Try AliceVision comprehensive analysis
        if (useAliceVision) {
            try {
                logger.info(" Step 1: AliceVision comprehensive AI analysis...");
                const comprehensiveResponse = await axios.post(`${ALICEVISION_URL}/comprehensive-analysis`, {
                    image: frames[0],
                    include_detection: true,
                    include_segmentation: true,
                    include_attributes: true,
                    include_quality: true
                }, { timeout: 90000 });

                if (comprehensiveResponse.data && comprehensiveResponse.data.success) {
                    analysisData = comprehensiveResponse.data;
                    steps.push("comprehensive_ai_analysis");

                    logger.info(` AI Analysis complete`);

                    // If quality is poor, try another frame
                    if (analysisData.quality && analysisData.quality.overall < 60 && frames.length > 1) {
                        logger.warn(" Quality score low, trying alternate frame...");
                        const altResponse = await axios.post(`${ALICEVISION_URL}/comprehensive-analysis`, {
                            image: frames[Math.floor(frames.length / 2)],
                            include_detection: true,
                            include_segmentation: true,
                            include_attributes: true,
                            include_quality: true
                        }, { timeout: 90000 });

                        if (altResponse.data.quality && altResponse.data.quality.overall > analysisData.quality.overall) {
                            analysisData = altResponse.data;
                        }
                    }
                }
            } catch (aiError) {
                logger.warn(" AliceVision comprehensive analysis unavailable:", aiError.message);
            }
        }

        // STEP 2: Segmentation fallback
        if (!analysisData.segmentation) {
            logger.info(" Step 2: Advanced clothing segmentation...");

            if (useAliceVision) {
                try {
                    const segmentResponse = await axios.post(`${ALICEVISION_URL}/segment`, {
                        image: bestFrame,
                        add_white_background: true,
                        use_advanced: true
                    }, { timeout: 60000 });

                    if (segmentResponse.data && segmentResponse.data.success) {
                        finalImageUrl = segmentResponse.data.segmentedImage;
                        analysisData.segmentation = {
                            confidence: segmentResponse.data.confidence,
                            itemCount: segmentResponse.data.itemCount,
                            items: segmentResponse.data.items
                        };
                        steps.push("advanced_segmentation");
                    }
                } catch (segError) {
                    logger.warn(" Advanced segmentation failed:", segError.message);
                }
            }

            // Fallback: AliceVision local rembg (free) → Replicate rembg (paid)
            if (!finalImageUrl) {
                // Try AliceVision local rembg first (free)
                try {
                    logger.info(" Fallback: AliceVision local background removal...");
                    const imageDataForAV = bestFrame.includes(',') ? bestFrame.split(',')[1] : bestFrame;
                    const avBgRes = await axios.post(`${ALICEVISION_URL}/remove-bg`, {
                        image: imageDataForAV,
                    }, { timeout: 60000 });

                    if (avBgRes.data && avBgRes.data.success) {
                        finalImageUrl = `data:image/png;base64,${avBgRes.data.image}`;
                        steps.push("alicevision_background_removal");
                        logger.info(` ✅ Background removed via AliceVision (free, ${avBgRes.data.processingTimeMs}ms)`);
                    }
                } catch (avBgErr) {
                    logger.warn(" AliceVision bg removal failed:", avBgErr.message);
                }
            }

            // Last resort: Replicate rembg (paid)
            if (!finalImageUrl) {
                logger.info(" Last resort: Replicate background removal (paid)...");
                try {
                    const imageDataUrl = `data:image/jpeg;base64,${bestFrame}`;
                    finalImageUrl = await replicate.run(
                        "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
                        { input: { image: imageDataUrl } }
                    );
                    steps.push("replicate_background_removal");
                } catch (rembgError) {
                    logger.warn("Background removal failed:", rembgError.message);
                    finalImageUrl = `data:image/jpeg;base64,${bestFrame}`;
                    steps.push("no_processing");
                }
            }
        } else {
            // Extract segmented image from comprehensive analysis
            try {
                const segmentResponse = await axios.post(`${ALICEVISION_URL}/segment`, {
                    image: bestFrame,
                    add_white_background: true,
                    use_advanced: true
                }, { timeout: 60000 });

                if (segmentResponse.data && segmentResponse.data.success) {
                    finalImageUrl = segmentResponse.data.segmentedImage;
                    steps.push("ai_segmentation");
                }
            } catch (err) {
                logger.warn("Segmentation extraction failed:", err.message);
            }
        }

        // STEP 3: Apply lighting normalization
        if (finalImageUrl && useAliceVision) {
            try {
                let imageB64 = finalImageUrl;
                if (imageB64.includes(',')) {
                    imageB64 = imageB64.split(',')[1];
                }

                const lightingResponse = await axios.post(`${ALICEVISION_URL}/lighting`, {
                    image: imageB64,
                    target_brightness: 0.6,
                    target_temperature: 6500,
                    add_vignette: false
                }, { timeout: 30000 });

                if (lightingResponse.data && lightingResponse.data.success) {
                    finalImageUrl = lightingResponse.data.normalizedImage;
                    steps.push("lighting_normalization");
                }
            } catch (lightError) {
                logger.warn(" Lighting normalization skipped:", lightError.message);
            }
        }

        // Build response
        res.json({
            success: true,
            imageUrl: finalImageUrl,
            bestFrameIndex,
            steps,
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
        });
    } catch (error) {
        logger.error("Product photo pipeline error:", error.message);
        res.status(500).json({ error: "Product photo processing failed" });
    }
});

// ── POST /v2/product-photo/process-multi ──
router.post("/v2/product-photo/process-multi", authenticateToken, requireTier('premium'), aiLimiter, async (req, res) => {
    try {
        const { frames, prompts = null } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        logger.info(` V2 Multi-Item Processing: ${frames.length} frames`);

        const itemCards = [];
        const steps = [];

        try {
            // STEP 1: Grounded SAM2 detection
            const detectionResponse = await axios.post(
                `${ALICEVISION_URL}/api/v2/detect-clothing`,
                {
                    image: frames[0],
                    prompts: prompts || ["shirt", "pants", "dress", "jacket", "skirt", "shoes", "bag"],
                    return_masks: true
                },
                { timeout: 60000 }
            );

            if (!detectionResponse.data.success || !detectionResponse.data.detections?.length) {
                return res.status(400).json({
                    error: "No clothing items detected in the image",
                    suggestion: "Make sure the image shows clothing items clearly"
                });
            }

            const detections = detectionResponse.data.detections;
            steps.push(`grounded_sam2_detected_${detections.length}_items`);

            // STEP 2: Process each detected item
            for (let i = 0; i < detections.length; i++) {
                const detection = detections[i];

                try {
                    // Extract fashion attributes
                    const attributesResponse = await axios.post(
                        `${ALICEVISION_URL}/api/v2/extract-fashion-attributes`,
                        { image: frames[0], roi: detection.bbox },
                        { timeout: 30000 }
                    );

                    if (!attributesResponse.data.success) continue;

                    const attributes = attributesResponse.data;
                    steps.push(`fashion_clip_${detection.category}`);

                    // Generate card prompt
                    const cardPromptResponse = await axios.post(
                        `${ALICEVISION_URL}/api/v2/generate-card-prompt`,
                        { attributes, style: "massimo_dutti", include_model: false },
                        { timeout: 10000 }
                    );

                    if (!cardPromptResponse.data.success) continue;

                    const cardPrompt = cardPromptResponse.data;
                    steps.push(`card_prompt_${detection.category}`);

                    // Segment item
                    const segmentResponse = await axios.post(
                        `${ALICEVISION_URL}/segment`,
                        { image: frames[0], add_white_background: true, use_advanced: true },
                        { timeout: 30000 }
                    );

                    const itemImageUrl = segmentResponse.data?.success
                        ? segmentResponse.data.segmentedImage
                        : `data:image/jpeg;base64,${frames[0]}`;

                    itemCards.push({
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
                        whiteBackground: true,
                        frontFacing: true
                    });
                } catch (itemError) {
                    logger.error(`Error processing item ${i + 1}:`, itemError.message);
                }
            }

            if (itemCards.length === 0) {
                return res.status(500).json({
                    error: "Failed to process any items",
                    detectedCount: detections.length
                });
            }

            res.json({
                success: true,
                totalItemsDetected: detections.length,
                totalCardsCreated: itemCards.length,
                items: itemCards,
                processing: { steps, aiEnhanced: true, model: "grounded_sam2_fashion_clip", style: "massimo_dutti" },
                summary: {
                    categories: itemCards.map(item => item.attributes.category),
                    colors: itemCards.map(item => item.attributes.primaryColor),
                    styles: itemCards.map(item => item.attributes.style)
                }
            });

        } catch (detectionError) {
            logger.error(" Grounded SAM2 detection failed:", detectionError.message);
            return res.status(503).json({
                error: "Multi-item detection service unavailable",
                message: "The AI vision service (Grounded SAM2) is not available.",
                fallback: "Use /api/product-photo/process for single-item processing"
            });
        }

    } catch (error) {
        logger.error("V2 Multi-Item Pipeline error:", error.message);
        res.status(500).json({ error: "Multi-item processing failed" });
    }
});

export default router;
