import express from "express";
import Replicate from "replicate";
import axios from "axios";
import { supabase } from "../lib/supabase.js";
import { authenticateToken } from "../middleware/auth.js";
import { validateClothingItem } from "../middleware/validators.js";
import logger from '../utils/logger.js';

const router = express.Router();

// Initialize Replicate
const replicate = new Replicate({
    auth: process.env.REPLICATE_API_TOKEN,
});

/**
 * POST /clothing-items
 * Save a single clothing item from video scan
 */
router.post("/", authenticateToken, validateClothingItem, async (req, res) => {
    try {
        const { type, color, style, description, season, imageUrl } = req.body;

        const userId = req.user.id; // From authenticateToken

        // Map to Postgres schema shape
        const itemData = {
            user_id: userId,
            type: type || "Unknown",
            category: "tops", // Default fallback, should ideally come from client
            sub_category: type || "Unknown",
            color: color || "Unknown",
            style: style || "Casual",
            description: description || "",
            // 'season' is an array in PG, cast it or wrap it
            season: season ? [season] : ["All Seasons"],
            image_url: imageUrl || "https://via.placeholder.com/150",
        };

        const { data: newItem, error } = await supabase
            .from('clothing_items')
            .insert([itemData])
            .select()
            .single();

        if (error) throw error;

        logger.info("✅ Saved clothing item:", newItem.type, "for user:", userId);
        res.status(201).json({ success: true, item: newItem });
    } catch (error) {
        logger.error("Error saving clothing item:", error.message || error);
        res.status(500).json({ error: "Failed to save clothing item" });
    }
});

/**
 * GET /clothing-items
 * Fetch user's clothing items
 */
router.get("/", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const { data: items, error } = await supabase
            .from('clothing_items')
            .select('*')
            .eq('user_id', userId)
            .order('created_at', { ascending: false });

        if (error) throw error;

        logger.info("📦 Found", items?.length || 0, "items for user:", userId);
        res.json({ items: items || [] });
    } catch (error) {
        logger.error("Error fetching clothing items:", error.message || error);
        res.status(500).json({ error: "Failed to fetch clothing items" });
    }
});

/**
 * POST /wardrobe/add-batch
 * Bulk add clothing items with AI-generated images
 */
router.post("/add-batch", authenticateToken, async (req, res) => {
    try {
        const { items } = req.body;
        const userId = req.user.id;

        if (!items || !Array.isArray(items) || items.length === 0) {
            return res.status(400).json({ error: "No items provided" });
        }

        logger.info(`🎨 Processing ${items.length} items via Supabase...`);

        const itemsWithImages = await Promise.all(
            items.map(async (item) => {
                let finalImageUrl = "https://via.placeholder.com/300?text=No+Image";

                try {
                    // Generate prompt for image
                    const prompt = `A professional studio photography of a ${item.color} ${item.style} ${item.itemType} (${item.description}), isolated on clean white background, flat lay, fashion catalog style, high quality, realistic, no shadows`;

                    // Generate image with Replicate
                    const output = await replicate.run("black-forest-labs/flux-schnell", {
                        input: {
                            prompt: prompt,
                            aspect_ratio: "1:1",
                            output_format: "jpg",
                            output_quality: 80,
                        },
                    });

                    // If image generated, upload to Supabase
                    if (output && output[0]) {
                        const replicateUrl = output[0];

                        // Download image
                        const imageResponse = await axios.get(replicateUrl, {
                            responseType: "arraybuffer",
                        });
                        const buffer = Buffer.from(imageResponse.data, "binary");

                        // Generate unique filename
                        const fileName = `${userId}/${Date.now()}_${Math.random().toString(36).substring(7)}.jpg`;

                        // Upload to Supabase Storage
                        const { data, error } = await supabase.storage
                            .from("AIWARDROBE")
                            .upload(fileName, buffer, {
                                contentType: "image/jpeg",
                                upsert: false,
                            });

                        if (error) {
                            logger.error("Supabase error:", error);
                            throw error;
                        }

                        // Get public URL
                        const { data: publicUrlData } = supabase.storage
                            .from("AIWARDROBE")
                            .getPublicUrl(fileName);

                        finalImageUrl = publicUrlData.publicUrl;
                    }
                } catch (genError) {
                    logger.error(`Error with item ${item.itemType}:`, genError.message);
                }

                // Return object mapped to Postgres schema
                return {
                    user_id: userId,
                    type: item.itemType || "Unknown",
                    category: "tops", // Fallback
                    color: item.color,
                    season: item.season ? [item.season] : [],
                    style: item.style,
                    description: item.description,
                    image_url: finalImageUrl,
                };
            })
        );

        // Save directly to Supabase
        const { data: savedItems, error: insertError } = await supabase
            .from('clothing_items')
            .insert(itemsWithImages)
            .select();

        if (insertError) throw insertError;

        logger.info(`✅ Successfully saved: ${savedItems?.length || 0} items`);
        res.status(201).json({ success: true, count: savedItems?.length || 0 });
    } catch (err) {
        logger.error("Critical Error:", err.message || err);
        res.status(500).json({ error: "Failed to save wardrobe items" });
    }
});

export default router;
