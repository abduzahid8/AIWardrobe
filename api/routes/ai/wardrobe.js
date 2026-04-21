/**
 * Wardrobe Scanning Route
 * POST /scan-wardrobe — Analyze wardrobe video frames using HuggingFace
 * (Replaced Replicate Video-LLaVA)
 */
import express from "express";
import multer from "multer";
import fs from "fs";
import { createClient } from "@supabase/supabase-js";
import { authenticateToken } from "../../middleware/auth.js";
import logger from "../../utils/logger.js";
import hfService from "../../services/huggingface.js";

import { supabase } from "../../lib/supabase.js";

const router = express.Router();
const upload = multer({ dest: "uploads/" });

// ── POST /scan-wardrobe ──
router.post("/scan-wardrobe", authenticateToken, upload.single("video"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No video file uploaded" });
        }

        logger.info("📹 Video received:", req.file.path);

        const fileBuffer = fs.readFileSync(req.file.path);
        const fileName = `scan_${Date.now()}.mp4`;
        const BUCKET_NAME = "AIWARDROBE";

        // Upload to Supabase Storage
        const { data: uploadData, error: uploadError } = await supabase.storage
            .from(BUCKET_NAME)
            .upload(fileName, fileBuffer, {
                contentType: "video/mp4",
                upsert: false,
            });

        if (uploadError) {
            logger.error("SUPABASE ERROR:", JSON.stringify(uploadError, null, 2));
            throw new Error(`Supabase upload failed: ${uploadError.message}`);
        }

        const { data: publicUrlData } = supabase.storage.from(BUCKET_NAME).getPublicUrl(fileName);
        const videoUrl = publicUrlData.publicUrl;
        logger.info(`📎 Video URL: ${videoUrl}`);

        // Since HF doesn't have a video analysis API, we'll check if the client
        // sent frames alongside the video. If not, we return a meaningful message.
        const { frames } = req.body;

        let items = [];

        if (frames && Array.isArray(frames) && frames.length > 0) {
            // Analyze extracted frames with HuggingFace
            logger.info(`📸 Analyzing ${frames.length} frames with HuggingFace...`);

            const framesToAnalyze = frames.slice(0, 3);

            for (let i = 0; i < framesToAnalyze.length; i++) {
                try {
                    const frame = framesToAnalyze[i].replace(/^data:image\/\w+;base64,/, "");
                    const analysis = await hfService.analyzeClothingImage(frame);

                    // Avoid duplicates
                    const isDuplicate = items.some(
                        (item) => item.itemType.toLowerCase() === analysis.itemType.toLowerCase()
                    );

                    if (!isDuplicate) {
                        items.push({
                            itemType: analysis.itemType.charAt(0).toUpperCase() + analysis.itemType.slice(1),
                            color: "Detected from frame",
                            style: analysis.style.charAt(0).toUpperCase() + analysis.style.slice(1),
                            description: analysis.description,
                        });
                    }

                    // Add strong secondary detections
                    for (const cat of analysis.categories.slice(1, 3)) {
                        if (cat.score > 0.15) {
                            const catDup = items.some(
                                (item) => item.itemType.toLowerCase() === cat.label.toLowerCase()
                            );
                            if (!catDup) {
                                items.push({
                                    itemType: cat.label.charAt(0).toUpperCase() + cat.label.slice(1),
                                    color: "Detected from frame",
                                    style: analysis.style.charAt(0).toUpperCase() + analysis.style.slice(1),
                                    description: analysis.description,
                                });
                            }
                        }
                    }
                } catch (frameErr) {
                    logger.warn(`Frame ${i + 1} analysis failed:`, frameErr.message);
                }
            }
        } else {
            // No frames provided — use imageToText on any snapshot
            // The client should ideally extract frames before uploading
            logger.warn("⚠️ No frames provided with video upload. Please extract frames client-side.");
            items = [{
                itemType: "Clothing Item",
                color: "Unknown",
                style: "Casual",
                description: "Video uploaded but frame analysis requires extracted frames. Please retry with frames parameter.",
            }];
        }

        if (items.length === 0) {
            items = [{
                itemType: "Unknown Item",
                color: "Unknown",
                style: "Casual",
                description: "Could not identify items from video frames",
            }];
        }

        // Clean up temp file
        if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);

        logger.info(`✅ Wardrobe scan detected ${items.length} items`);
        res.json({ detectedItems: items, videoUrl });
    } catch (error) {
        logger.error("Video Scan Error:", error.message);
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        res.status(500).json({ error: "Video scan failed" });
    }
});

export default router;
