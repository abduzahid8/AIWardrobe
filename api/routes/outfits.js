import express from "express";
import { supabase } from "../lib/supabase.js";
import { authenticateToken } from "../middleware/auth.js";

import logger from '../utils/logger.js';
const router = express.Router();

/**
 * POST /save-outfit
 * Save a new outfit for the authenticated user
 */
router.post("/", authenticateToken, async (req, res) => {
    try {
        const { date, items, caption, occasion, visibility, isOotd } = req.body;
        const userId = req.user.id;

        const itemsWithImages = items?.map((item) => {
            if (!item || typeof item !== "object") {
                console.warn("Invalid item skipped", item);
                return null;
            }
            const imageUrl = item?.image;
            if (!imageUrl || !imageUrl.match(/^https?:\/\/res\.cloudinary\.com/)) {
                console.warn("Invalid or non-Cloudinary image URL:", imageUrl);
                return null;
            }
            return {
                id: item.id !== undefined || "null",
                type: item.type || "Unknown",
                image: imageUrl,
                x: item.x !== undefined ? item?.x : 0,
                y: item.y !== undefined ? item?.y : 0,
            };
        });

        const validItems = itemsWithImages.filter((item) => item !== null);

        if (validItems.length === 0) {
            return res.status(400).json({ error: "No valid items provided" });
        }

        // The existing frontend payload parses items with X/Y coordinates.
        // Let's store the raw JSON array in Supabase 
        const { data: newOutfit, error } = await supabase
            .from('outfits')
            .insert([{
                user_id: userId,
                occasion: occasion || "",
                style: validItems[0]?.style || "casual", // derive style if missing
                items: validItems.map(i => i.id !== "null" ? i.id : null).filter(Boolean), // array of uuids
                notes: caption || "",
                // Store UI mapping data (x, y coordinates, images) in a metadata column if needed,
                // but for now, we'll store the core details and assume imageUrl is generated elsewhere
                image_url: validItems[0]?.image || ""
                // Alternatively, if there is a 'metadata' JSONB column, we could store validItems whole
            }])
            .select()
            .single();

        if (error) throw error;

        logger.info("✅ Outfit saved for user:", userId);
        res.status(201).json({ outfit: newOutfit });
    } catch (err) {
        logger.error("Error in save-outfit:", err.message);
        res.status(500).json({ error: "Internal server error", details: err.message });
    }
});

/**
 * GET /save-outfit/user/:userId
 * Get all outfits for a specific user
 */
router.get("/user/:userId", authenticateToken, async (req, res) => {
    try {
        const userId = req.params.userId;

        if (req.user.id !== userId) {
            return res.status(403).json({ error: "Unauthorized access" });
        }

        const { data: outfits, error } = await supabase
            .from('outfits')
            .select('*')
            .eq('user_id', userId)
            .order('created_at', { ascending: false });

        if (error) throw error;

        res.status(200).json(outfits || []);
    } catch (error) {
        logger.error("Error fetching outfits:", error);
        res.status(500).json({ error: "Internal server error", details: error.message });
    }
});

export default router;
