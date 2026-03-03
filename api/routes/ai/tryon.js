/**
 * Virtual Try-On Route
 * POST /try-on — IDM-VTON virtual try-on via Replicate
 */
import express from "express";
import Replicate from "replicate";
import { authenticateToken } from "../../middleware/auth.js";
import { requireTier } from "../../middleware/subscriptionGuard.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";

const router = express.Router();
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// ── POST /try-on ──
router.post("/try-on", authenticateToken, requireTier('premium'), aiLimiter, async (req, res) => {
    const { human_image, garment_image, description } = req.body;

    logger.info(" Starting virtual try-on...");

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

        logger.info(" Try-on complete");
        res.json({ image: output });
    } catch (error) {
        logger.error("Replicate error:", error);
        res.status(500).json({ error: "Virtual try-on failed" });
    }
});

export default router;
