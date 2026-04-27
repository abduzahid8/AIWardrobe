/**
 * Unified Image Processor Routes
 * POST /api/process-upload  — Full pipeline: remove bg → classify → enhance → describe
 *
 * Primary: AliceVision Python service (free, zero API cost)
 * Fallback: Replicate rembg + Gemini Vision + HuggingFace CLIP
 */
import express from "express";
import axios from "axios";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import { ALICEVISION_URL } from "../../config.js";
import logger from "../../utils/logger.js";
import { validateImageData } from "../../middleware/validators.js";
import hfService from "../../services/huggingface.js";
import geminiService from "../../services/gemini.js";
import replicateService from "../../services/replicate.js";
import openaiService from "../../services/openai.js";

const router = express.Router();

// Massimo Dutti-style prompts for fallback descriptions
const STYLE_DESCRIPTIONS = {
    tops: "Premium cotton piece with impeccable tailoring and a contemporary silhouette",
    bottoms: "Tailored trouser with precise fit and refined craftsmanship",
    shoes: "Sleek footwear combining artisan quality with understated luxury",
    outerwear: "Sophisticated outerwear with clean lines and a luxurious drape",
    dresses: "Fluid dress with effortless elegance and premium fabric selection",
    accessories: "Refined accessory with minimalist design and premium materials",
    other: "Fashion piece with considered design and quality construction",
};

// Map labels to sections
const LABEL_TO_SECTION = {
    "t-shirt": "tops", "shirt": "tops", "blouse": "tops", "sweater": "tops",
    "hoodie": "tops", "jacket": "outerwear", "coat": "outerwear",
    "dress": "dresses", "skirt": "bottoms", "pants": "bottoms",
    "jeans": "bottoms", "shorts": "bottoms", "sneakers": "shoes",
    "boots": "shoes", "sandals": "shoes", "bag": "accessories",
    "hat": "accessories", "scarf": "accessories", "belt": "accessories",
    "watch": "accessories",
};

/**
 * POST /api/process-upload
 */
router.post(
    "/process-upload",
    aiLimiter,
    validateImageData,
    async (req, res) => {
        try {
            const { imageBase64, generateDescription = true } = req.body;

            if (!imageBase64) {
                return res.status(400).json({ error: "imageBase64 is required" });
            }

            logger.info("🎨 Process-upload: Starting unified pipeline...");

            // ── Strategy 1: AliceVision full pipeline (preferred, free) ──
            // Only use if OpenAI is not available, because we want Node to orchestrate the OpenAI enhancement.
            if (!openaiService.isAvailable()) {
                try {
                    const response = await axios.post(
                        `${ALICEVISION_URL}/process`,
                        {
                            image: imageBase64,
                            mode: "creative",
                            generate_description: generateDescription,
                        },
                        { timeout: 120000 }
                    );

                    if (response.data?.success) {
                        logger.info(
                            `✅ AliceVision pipeline complete: ${response.data.classification?.category || "unknown"} (${response.data.totalProcessingMs}ms)`
                        );

                        return res.json({
                            success: true,
                            imageUrl: `data:image/jpeg;base64,${response.data.image}`,
                            cutoutUrl: `data:image/png;base64,${response.data.cutout}`,
                            classification: response.data.classification,
                            description: response.data.description,
                            style: response.data.style,
                            dimensions: response.data.dimensions,
                            steps: response.data.steps,
                            totalProcessingMs: response.data.totalProcessingMs,
                            provider: "alicevision",
                            costs: response.data.costs,
                        });
                    }
                } catch (avError) {
                    logger.warn(
                        `⚠️ AliceVision unavailable: ${avError.message}. Trying fallbacks...`
                    );
                }
            }

            // ── Strategy 2: Multi-provider fallback ──
            let cutoutB64 = null;
            let classification = null;
            let enhancedB64 = null;
            let description = null;
            const steps = [];

            // 2b. Precision Background Removal: Replicate Grounded SAM → AliceVision fallback
            // Now that we know WHAT the item is, we can mathematically isolate ONLY that item
            if (replicateService.isAvailable() && classification?.category) {
                try {
                    const itemPrompt = classification.category.toLowerCase();
                    logger.info(`✂️ Using Replicate Grounded SAM to isolate exactly: ${itemPrompt}...`);
                    const repResult = await replicateService.isolateItem(imageBase64, itemPrompt);
                    cutoutB64 = repResult.image;
                    steps.push("replicate_grounded_sam_cutout");
                    logger.info(`✅ Precision cutout complete via Replicate (${repResult.processingTimeMs}ms)`);
                } catch (repErr) {
                    logger.warn("Replicate Grounded SAM failed, falling back to basic rembg:", repErr.message);
                }
            }

            // Fallback to AliceVision basic generic rembg if Grounded SAM failed or is unavailable
            if (!cutoutB64) {
                try {
                    const bgRes = await axios.post(
                        `${ALICEVISION_URL}/remove-bg`,
                        { image: imageBase64 },
                        { timeout: 60000 }
                    );
                    if (bgRes.data?.success) {
                        cutoutB64 = bgRes.data.image;
                        steps.push("alicevision_remove_bg");
                    }
                } catch (e) {
                    logger.warn("AliceVision generic bg removal also failed");

                    // Fallback to Replicate basic rembg
                    if (replicateService.isAvailable()) {
                        try {
                            const repResult = await replicateService.removeBackground(imageBase64);
                            cutoutB64 = repResult.image;
                            steps.push("replicate_remove_bg");
                        } catch (err) { }
                    }
                }
            }

            // 2a. Classification: AliceVision → Gemini Vision → HF CLIP
            // We classify FIRST so we know exactly what item to isolate in complex photos
            try {
                const clsRes = await axios.post(
                    `${ALICEVISION_URL}/classify`,
                    { image: imageBase64 },
                    { timeout: 30000 }
                );
                if (clsRes.data?.success) {
                    classification = {
                        category: clsRes.data.category,
                        section: clsRes.data.section,
                        confidence: clsRes.data.confidence,
                        top5: clsRes.data.top5,
                    };
                    steps.push("alicevision_classify");
                }
            } catch (e) {
                logger.warn("AliceVision classification unavailable");
            }

            // Gemini Vision classification fallback
            if (!classification && geminiService.isAvailable()) {
                try {
                    logger.info("🔮 Using Gemini Vision for classification...");
                    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");
                    const analysis = await geminiService.analyzeClothingImage(cleanBase64);

                    const section = LABEL_TO_SECTION[analysis.category?.toLowerCase()] || "other";
                    classification = {
                        category: analysis.category,
                        section,
                        confidence: 0.95,
                        attributes: {
                            style: analysis.style,
                            color: analysis.primaryColor,
                            pattern: analysis.pattern,
                            material: analysis.material,
                        },
                    };
                    description = analysis.description;
                    steps.push("gemini_classify");
                } catch (geminiErr) {
                    logger.warn("Gemini Vision fallback failed:", geminiErr.message);
                }
            }

            // HuggingFace CLIP classification fallback
            if (!classification) {
                try {
                    logger.info("🤗 Using HuggingFace CLIP for classification...");
                    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

                    const fashionLabels = [
                        "T-shirt/top", "Shirt", "Pullover", "Coat", "Dress",
                        "Trouser", "Sandal", "Sneaker", "Ankle boot", "Bag",
                    ];
                    const categories = await hfService.zeroShotImageClassify(cleanBase64, fashionLabels);

                    const styleLabels = ["casual", "formal", "sport", "semi_classic"];
                    const styles = await hfService.zeroShotImageClassify(cleanBase64, styleLabels);

                    const topCat = categories[0] || { label: "unknown", score: 0.5 };
                    const topStyle = styles[0] || { label: "casual", score: 0.5 };
                    const section = LABEL_TO_SECTION[topCat.label.toLowerCase()] || "other";

                    classification = {
                        category: topCat.label,
                        section,
                        confidence: topCat.score,
                        attributes: {
                            style: topStyle.label,
                        },
                    };
                    steps.push("hf_classify");
                } catch (hfErr) {
                    logger.warn(`HuggingFace fallback failed: ${hfErr.message}`);
                }
            }

            // 2c. Enhancement: Ghost Mannequin (Primary) → OpenAI (Secondary) → AliceVision (Fallback)
            // The source for Ghost Mannequin is the ORIGINAL image (not the cutout),
            // because IP-Adapter needs the full garment context to resynthesize it.
            const categoryName = classification?.category || "garment";

            // ── Tier 1: Replicate Ghost Mannequin (generative flat-lay) ──
            if (replicateService.isAvailable()) {
                try {
                    logger.info(`📸 Generating Ghost Mannequin flat-lay for '${categoryName}'...`);
                    const gmResult = await replicateService.generateGhostMannequin(imageBase64, categoryName);
                    enhancedB64 = gmResult.image;
                    steps.push("replicate_ghost_mannequin");
                    logger.info(`✅ Ghost Mannequin complete (${gmResult.processingTimeMs}ms)`);
                } catch (e) {
                    logger.warn("Ghost Mannequin failed:", e.message);
                }
            }

            // ── Tier 2: OpenAI DALL-E studio edit (if we have a cutout) ──
            if (!enhancedB64 && cutoutB64 && openaiService.isAvailable()) {
                try {
                    logger.info("🎨 Sending cutout to OpenAI for Massimo Dutti studio generation...");
                    const prompt = `Professional e-commerce product photo of a premium ${categoryName}, studio product photography, light gray background, soft diffused studio lighting, subtle drop shadow, centered, crisp details, Massimo Dutti style, magazine quality, 8k`;
                    enhancedB64 = await openaiService.editImage(cutoutB64, prompt);
                    steps.push("openai_studio_enhance");
                } catch (e) {
                    logger.warn("OpenAI enhancement failed:", e.message);
                }
            }

            // ── Tier 3: AliceVision LOCAL Stable Diffusion flat-lay (SD 1.5 + ControlNet + IP-Adapter) ──
            if (!enhancedB64 && cutoutB64) {
                try {
                    logger.info(`🎨 AliceVision SD flat-lay for '${categoryName}'...`);
                    const flatRes = await axios.post(
                        `${ALICEVISION_URL}/flat-lay`,
                        { image: cutoutB64, category: categoryName },
                        { timeout: 180000 }  // SD generation can take 1-3 minutes on MPS
                    );
                    if (flatRes.data?.success) {
                        enhancedB64 = flatRes.data.image;
                        steps.push("alicevision_sd_flatlay");
                        logger.info("✅ AliceVision SD flat-lay complete");
                    }
                } catch (e) {
                    logger.warn("AliceVision SD flat-lay failed:", e.message);
                }
            }

            // ── Tier 4: AliceVision basic Pillow-based enhancement (absolute last resort) ──
            if (!enhancedB64 && cutoutB64) {
                try {
                    const enhRes = await axios.post(
                        `${ALICEVISION_URL}/enhance`,
                        { image: cutoutB64 },
                        { timeout: 30000 }
                    );
                    if (enhRes.data?.success) {
                        enhancedB64 = enhRes.data.image;
                        steps.push("alicevision_pillow_enhance");
                    }
                } catch (e) {
                    logger.warn("AliceVision Pillow enhancement also skipped");
                }
            }

            // ── Build description ──
            if (!description && classification) {
                const section = classification.section || "other";
                description = STYLE_DESCRIPTIONS[section] || STYLE_DESCRIPTIONS.other;
            }

            // ── Final response ──
            const finalImage = enhancedB64 || cutoutB64 || imageBase64;
            const finalFormat = enhancedB64 ? "jpeg" : cutoutB64 ? "png" : "jpeg";

            res.json({
                success: true,
                imageUrl: `data:image/${finalFormat};base64,${finalImage}`,
                cutoutUrl: cutoutB64
                    ? `data:image/png;base64,${cutoutB64}`
                    : null,
                classification,
                description,
                style: "massimo_dutti",
                steps,
                provider: steps.length > 0 ? "hybrid" : "passthrough",
                costs: { api_calls: 0, paid: false },
            });
        } catch (error) {
            logger.error("Process-upload error:", error.message);
            res.status(500).json({ error: "Image processing failed" });
        }
    }
);

export default router;
