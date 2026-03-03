/**
 * Wardrobe Scanning Route
 * POST /scan-wardrobe — Video-LLaVA wardrobe scan via Replicate
 */
import express from "express";
import multer from "multer";
import fs from "fs";
import Replicate from "replicate";
import { createClient } from "@supabase/supabase-js";
import { authenticateToken } from "../../middleware/auth.js";
import logger from "../../utils/logger.js";

const router = express.Router();
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });
const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_KEY);
const upload = multer({ dest: "uploads/" });

// ── POST /scan-wardrobe ──
router.post("/scan-wardrobe", authenticateToken, upload.single("video"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No video file uploaded" });
        }

        logger.info(" Video received:", req.file.path);

        const fileBuffer = fs.readFileSync(req.file.path);
        const fileName = `scan_${Date.now()}.mp4`;
        const BUCKET_NAME = "AIWARDROBE";

        const { data: uploadData, error: uploadError } = await supabase.storage
            .from(BUCKET_NAME)
            .upload(fileName, fileBuffer, {
                contentType: "video/mp4",
                upsert: false,
            });

        if (uploadError) {
            logger.error(" SUPABASE ERROR:", JSON.stringify(uploadError, null, 2));
            throw new Error(`Supabase upload failed: ${uploadError.message}`);
        }

        const { data: publicUrlData } = supabase.storage.from(BUCKET_NAME).getPublicUrl(fileName);
        const videoUrl = publicUrlData.publicUrl;

        logger.info(` Video URL: ${videoUrl}`);

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

        const rawText = Array.isArray(output) ? output.join("") : String(output);

        let items = [];
        try {
            const firstBracket = rawText.indexOf("[");
            const lastBracket = rawText.lastIndexOf("]");

            if (firstBracket !== -1 && lastBracket !== -1) {
                const jsonStr = rawText.substring(firstBracket, lastBracket + 1);
                items = JSON.parse(jsonStr);
            } else {
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
            logger.error("Parse error:", parseErr);
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
        logger.error("Video Scan Error:", error);
        if (req.file && fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
        res.status(500).json({ error: "Video scan failed" });
    }
});

export default router;
