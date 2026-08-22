/**
 * Replicate AI Service — Virtual Try-On & Background Removal
 *
 * Uses the existing REPLICATE_API_TOKEN from .env.
 *
 * Exports:
 *   - virtualTryOn(personImageB64, garmentImageB64, options)
 *   - removeBackground(imageBase64)
 */
import Replicate from "replicate";
import logger from "../utils/logger.js";

// ── Singleton ──
const REPLICATE_API_TOKEN = process.env.REPLICATE_API_TOKEN;
if (!REPLICATE_API_TOKEN) {
    logger.warn("⚠️  REPLICATE_API_TOKEN not set — Replicate features will be unavailable");
}

const replicate = REPLICATE_API_TOKEN ? new Replicate({ auth: REPLICATE_API_TOKEN }) : null;

// ── Model IDs ──
const TRYON_MODEL = "cuuupid/idm-vton:c871bb9b046c1b1f6571f934c750ab1d72e05e27c6f0184093e605e8c09c3dea";
const REMBG_MODEL = "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003";

// ═══════════════════════════════════════════
// VIRTUAL TRY-ON (IDM-VTON)
// ═══════════════════════════════════════════

/**
 * Run virtual try-on using IDM-VTON on Replicate.
 * @param {string} personImageB64 - Base64 person image
 * @param {string} garmentImageB64 - Base64 garment image
 * @param {object} options
 * @param {string} options.garmentType - "upper_body", "lower_body", or "dresses"
 * @param {string} options.description - Garment description
 * @param {number} options.steps - Denoising steps (default 30)
 * @param {number} options.seed - Random seed (default 42)
 * @returns {Promise<{image: string, elapsed_seconds: number}>}
 */
export async function virtualTryOn(personImageB64, garmentImageB64, options = {}) {
    if (!replicate) throw new Error("Replicate API token not configured");

    const {
        garmentType = "upper_body",
        description = "A stylish clothing item",
        steps = 30,
        seed = 42,
    } = options;

    logger.info(`👗 Replicate: Starting IDM-VTON try-on (${garmentType})...`);

    const startTime = Date.now();

    // Ensure proper data URI format for Replicate
    const personUri = personImageB64.startsWith("data:")
        ? personImageB64
        : `data:image/jpeg;base64,${personImageB64}`;
    const garmentUri = garmentImageB64.startsWith("data:")
        ? garmentImageB64
        : `data:image/jpeg;base64,${garmentImageB64}`;

    const output = await replicate.run(TRYON_MODEL, {
        input: {
            human_img: personUri,
            garm_img: garmentUri,
            garment_des: description,
            category: garmentType,
            num_inference_steps: steps,
            seed,
        },
    });

    const elapsed = (Date.now() - startTime) / 1000;

    // Output is a URL to the generated image — download and convert to base64
    if (output) {
        const resultUrl = typeof output === "string" ? output : output[0] || output;
        logger.info(`✅ Replicate try-on complete in ${elapsed.toFixed(1)}s`);

        // Fetch the image and convert to base64
        const response = await fetch(resultUrl);
        const arrayBuffer = await response.arrayBuffer();
        const base64 = Buffer.from(arrayBuffer).toString("base64");

        return {
            image: base64,
            elapsed_seconds: elapsed,
        };
    }

    throw new Error("Replicate returned no output");
}

// ═══════════════════════════════════════════
// BACKGROUND REMOVAL & SEGMENTATION
// ═══════════════════════════════════════════

/**
 * Remove background using Replicate's rembg model.
 * @param {string} imageBase64 - Base64 image
 * @returns {Promise<{image: string, processingTimeMs: number}>}
 */
export async function removeBackground(imageBase64) {
    if (!replicate) throw new Error("Replicate API token not configured");

    logger.info("✂️ Replicate: Removing background with rembg...");
    const startTime = Date.now();

    const imageUri = imageBase64.startsWith("data:")
        ? imageBase64
        : `data:image/jpeg;base64,${imageBase64}`;

    const output = await replicate.run(REMBG_MODEL, {
        input: {
            image: imageUri,
        },
    });

    const elapsed = Date.now() - startTime;

    if (output) {
        const resultUrl = typeof output === "string" ? output : output[0] || output;

        // Fetch and convert to base64
        const response = await fetch(resultUrl);
        const arrayBuffer = await response.arrayBuffer();
        const base64 = Buffer.from(arrayBuffer).toString("base64");

        logger.info(`✅ Replicate bg removal complete (${elapsed}ms)`);
        return {
            image: base64,
            processingTimeMs: elapsed,
        };
    }

    throw new Error("Replicate rembg returned no output");
}

/**
 * Precisely isolate a specific item using Grounded SAM.
 * @param {string} imageBase64 - Base64 image
 * @param {string} itemLabel - Text prompt of what to isolate (e.g. "Jacket", "Shirt")
 * @returns {Promise<{image: string, processingTimeMs: number}>}
 */
export async function isolateItem(imageBase64, itemLabel) {
    if (!replicate) throw new Error("Replicate API token not configured");

    logger.info(`✂️ Replicate: Isolating '${itemLabel}' with Grounded SAM...`);
    const startTime = Date.now();

    const imageUri = imageBase64.startsWith("data:")
        ? imageBase64
        : `data:image/jpeg;base64,${imageBase64}`;

    // idea-research/grounded-sam:
    // Returns a masked image based on the text prompt
    const output = await replicate.run(
        "idea-research/grounded-sam:973dfecab61d76332ec484e5c830db92ce5ef8bb194c776fb083fb83883a9030",
        {
            input: {
                image: imageUri,
                mask_prompt: itemLabel,
                background: "transparent",
                adjustment_factor: 0
            }
        }
    );

    const elapsed = Date.now() - startTime;

    if (output) {
        // Output can be a single URL string depending on the exact replicate model schema
        const resultUrl = typeof output === "string" ? output : output[0] || output;

        // Fetch and convert to base64
        const response = await fetch(resultUrl);
        const arrayBuffer = await response.arrayBuffer();
        const base64 = Buffer.from(arrayBuffer).toString("base64");

        logger.info(`✅ Replicate Grounded SAM complete (${elapsed}ms)`);
        return {
            image: base64,
            processingTimeMs: elapsed,
        };
    }

    throw new Error("Replicate Grounded SAM returned no output");
}

// ═══════════════════════════════════════════
// GENERATIVE PRODUCT PHOTOGRAPHY (GHOST MANNEQUIN)
// ═══════════════════════════════════════════

/**
 * Generate a pristine ghost mannequin / flat-lay catalog photo.
 * Uses Replicate IP-Adapter SDXL to synthesize the garment onto a perfect template.
 * @param {string} imageBase64 - Base64 image of the cutout garment
 * @param {string} category - e.g., "Jacket", "Shirt"
 * @returns {Promise<{image: string, processingTimeMs: number}>}
 */
export async function generateGhostMannequin(imageBase64, category = "upper_body") {
    if (!replicate) throw new Error("Replicate API token not configured");

    logger.info(`📸 Replicate: generating SOTA Invisible Mannequin via IDM-VTON for '${category}'...`);
    const startTime = Date.now();

    const imageUri = imageBase64.startsWith("data:")
        ? imageBase64
        : `data:image/jpeg;base64,${imageBase64}`;

    // A light grey, blank 3D mannequin silhouette generated specifically as a blank canvas for IDM-VTON
    const blankMannequinURI = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAQAAwADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKqzf61qtVVm/wa1ADKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAWP76/UVcqnH99fqKuUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABVWb/WtVqqszDzXoAZRS719RS719RQA2il3r6ijevoKAEopd6+oo3r6igBKKXevqKNy+tACUUu9fUUm9fWgAopfMX1pd6+tADaKd5if3hRvT1FADaKPMX1FJvX1FAC0Um9fWl3p60AAopN6+tLvj/vCgAopN8f94Ubk/vUALRSb4/7wo3p/eFAC0Um9fWk3r6igB1FN3r6il3r60ALRSb19aXevoKACiiigBY/vr9RVuqcf3x9RVygAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//Z";

    try {
        const output = await replicate.run(
            "cuuupid/idm-vton:0513734a452173b8173e907e3a59d19a36266e55b48528559432bd21c7d7e985",
            {
                input: {
                    garm_img: imageUri,
                    human_img: blankMannequinURI,
                    category: category,    // Uses upper_body, lower_body, dresses
                    garment_des: `Professional product catalog photo of a garment`,
                    crop: false,
                    steps: 30
                }
            }
        );

        const elapsed = Date.now() - startTime;

        if (output) {
            const resultUrl = typeof output === "string" ? output : output[0] || output;

            // Replicate outputs WebP/PNG URLs usually
            const response = await fetch(resultUrl);
            const arrayBuffer = await response.arrayBuffer();
            const base64 = Buffer.from(arrayBuffer).toString("base64");

            logger.info(`✅ Replicate Ghost Mannequin (IDM-VTON) complete (${elapsed}ms)`);
            return {
                image: base64,
                processingTimeMs: elapsed,
            };
        }
    } catch (e) {
        logger.error(`Replicate IDM-VTON failed: ${e.message}`);
        throw e;
    }

    throw new Error("Replicate Ghost Mannequin returned no output");
}

// ═══════════════════════════════════════════
// UTILITY
// ═══════════════════════════════════════════

export function isAvailable() {
    return !!replicate;
}

export default {
    virtualTryOn,
    removeBackground,
    isolateItem,
    generateGhostMannequin,
    isAvailable,
};
